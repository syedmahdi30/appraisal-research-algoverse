"""EMOTIC loader (images, Stage B/C) — parses the original Annotations.mat.

GATED dataset (docs/datasets.md); this module never fabricates access. Pipeline:
    emotic.zip  -> data/raw/emotic/emotic/<subset>/images/*.jpg   (images)
    Annotations.mat -> convert_mat_to_parquet(...) -> data/processed/emotic_{split}.parquet

`Annotations.mat` (scipy.io.loadmat, squeeze_me + struct_as_record=False) has top-level
arrays `train` / `val` / `test`, each an array of image structs:
    image.filename, image.folder, image.image_size, image.person[]
    person.body_bbox, person.annotations_{categories,continuous},
           person.combined_{categories,continuous}  (val/test aggregate annotators)

We prefer `combined_*` when present (multi-annotator val/test), else `annotations_*`
(train). VAD is 1-10; categories are the EMOTIC-26 strings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import PROCESSED_DIR, RAW_DIR

EMOTIC_DIR = RAW_DIR / "emotic"
# emotic.zip nests everything under a top-level `emotic/`, so images live one level deeper.
IMAGES_ROOT = EMOTIC_DIR / "emotic"
SPLITS = ("train", "val", "test")

_ACCESS_MSG = (
    "EMOTIC not found. It is gated: (1) extract emotic.zip to {img}, (2) place "
    "Annotations.mat and run convert_mat_to_parquet(...). See docs/datasets.md."
)


# --------------------------------------------------------------------------- helpers
def _as_str_list(x) -> list[str]:
    """Normalize a categories field (str / ndarray of str / nested) to list[str]."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(v) for v in np.atleast_1d(x).ravel()]


def _scalar(x) -> float | None:
    """Reduce a continuous value (scalar or per-annotator array) to a float mean."""
    if x is None:
        return None
    arr = np.atleast_1d(x).astype(float).ravel()
    return float(np.mean(arr)) if arr.size else None


def _person_continuous(person):
    """VAD struct: `combined_continuous` (val/test) else `annotations_continuous` (train).

    Both are structs with valence/arousal/dominance. In val/test `annotations_continuous`
    is a per-annotator array, so we always prefer the aggregated `combined_continuous`.
    """
    combined = getattr(person, "combined_continuous", None)
    if combined is not None and getattr(combined, "_fieldnames", None):
        return combined
    return getattr(person, "annotations_continuous", None)


def _person_categories(person) -> list[str]:
    """EMOTIC-26 categories for a person, across the two mat conventions.

    val/test: `combined_categories` is a BARE array of category strings.
    train:    `annotations_categories` is a struct (single annotator) with `.categories`.
    """
    combined = getattr(person, "combined_categories", None)
    if combined is not None:
        return _as_str_list(combined)  # already the string list
    ann = getattr(person, "annotations_categories", None)
    if ann is None:
        return []
    cats: set[str] = set()  # union over annotator struct(s), each carrying .categories
    for s in np.atleast_1d(ann):
        cats.update(_as_str_list(getattr(s, "categories", None)))
    return sorted(cats)


def _person_record(image, person, images_root: Path) -> dict:
    folder = str(getattr(image, "folder", ""))
    filename = str(getattr(image, "filename", ""))
    cont_struct = _person_continuous(person)
    categories = _person_categories(person)
    return {
        "image_path": str((images_root / folder / filename).resolve()),
        "folder": folder,
        "filename": filename,
        "bbox": np.atleast_1d(getattr(person, "body_bbox", [])).astype(int).tolist(),
        "categories": categories,
        "valence": _scalar(getattr(cont_struct, "valence", None)) if cont_struct is not None else None,
        "arousal": _scalar(getattr(cont_struct, "arousal", None)) if cont_struct is not None else None,
        "dominance": _scalar(getattr(cont_struct, "dominance", None)) if cont_struct is not None else None,
    }


# --------------------------------------------------------------------------- convert
def convert_mat_to_parquet(
    mat_path: str | Path,
    images_root: Path = IMAGES_ROOT,
    out_dir: Path = PROCESSED_DIR,
    splits: tuple[str, ...] = SPLITS,
    verify_images: bool = True,
) -> dict[str, int]:
    """Parse Annotations.mat into per-person parquet tables, one per split.

    Returns {split: n_rows}. If `verify_images`, asserts the first image path in each
    split exists so a wrong `images_root` fails loudly rather than deep in Stage C.
    """
    import scipy.io as sio

    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for split in splits:
        if split not in mat:
            raise KeyError(f"split {split!r} not in {mat_path}; keys={[k for k in mat if not k.startswith('__')]}")
        rows: list[dict] = []
        for image in np.atleast_1d(mat[split]):
            for person in np.atleast_1d(getattr(image, "person")):
                rec = _person_record(image, person, images_root)
                rec["split"] = split
                rows.append(rec)
        df = pd.DataFrame(rows)

        if verify_images and len(df):
            first = Path(df.iloc[0]["image_path"])
            if not first.exists():
                raise FileNotFoundError(
                    f"resolved image path does not exist: {first}\n"
                    f"Fix `images_root` (currently {images_root}). The mat 'folder' field is "
                    f"{df.iloc[0]['folder']!r}; images_root/folder/filename must point at a real file."
                )

        out = out_dir / f"emotic_{split}.parquet"
        df.to_parquet(out)
        counts[split] = len(df)
        print(f"[emotic] {split}: {len(df)} persons -> {out}")

    return counts


# --------------------------------------------------------------------------- load
def load_split(split: str) -> pd.DataFrame:
    """Load a converted EMOTIC split (per-person rows) from data/processed/."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = PROCESSED_DIR / f"emotic_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            _ACCESS_MSG.format(img=IMAGES_ROOT)
            + f"\nMissing {path} — run emotic.convert_mat_to_parquet(mat_path)."
        )
    return pd.read_parquet(path)


def to_shared_single_label(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the lossy EMOTIC-26 -> shared-7 mapping; keep single-label rows only.

    Each row's `categories` list is mapped through EMOTIC_TO_SHARED (unmapped dropped) and
    deduplicated; rows not collapsing to exactly one shared label are removed. The number
    dropped is stored in `.attrs['n_dropped_multilabel']` — a first-class Stage C caveat.
    """
    from .labels import EMOTIC_TO_SHARED

    unknown: set[str] = set()

    def collapse(cats) -> str | None:
        mapped = set()
        for c in cats:
            key = str(c).strip()
            if key not in EMOTIC_TO_SHARED:
                unknown.add(key)  # spelling/whitespace mismatch — surface, don't crash
                continue
            if EMOTIC_TO_SHARED[key] is not None:
                mapped.add(EMOTIC_TO_SHARED[key])
        return next(iter(mapped)) if len(mapped) == 1 else None

    out = df.copy()
    out["shared_label"] = out["categories"].map(collapse)
    kept = out[out["shared_label"].notna()].reset_index(drop=True)
    kept.attrs["n_dropped_multilabel"] = len(out) - len(kept)
    kept.attrs["unknown_categories"] = sorted(unknown)
    return kept
