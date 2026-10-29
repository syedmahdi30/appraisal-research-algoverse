"""Build the public results bundle in release/results/ from the local results/ tree.

EMOTIC is distributed under a gated, non-commercial agreement, so its annotations are not ours to
redistribute. Model outputs are. This copies every result file, removes the columns that carry
EMOTIC's own annotations, and replaces each with `<column>_emotic_row`, the row position of the
source annotation in `data/processed/emotic_test.parquet`: an index, which a user with EMOTIC access
turns back into the column with `scripts/rehydrate_results.py`.

Every file is listed in release/MANIFEST.json with its sha256, whether the paper cites it (and
where), and which columns were stripped. Files the paper does not cite are released too, marked as
such: they are superseded, exploratory, or provenance for a retraction the paper describes.

    python scripts/package_results.py            # needs results/ and data/processed/emotic_test.parquet
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "results", ROOT / "release" / "results"
EMOTIC = ROOT / "data" / "processed" / "emotic_test.parquet"

# (annotation column, column holding the image path it belongs to)
STRIP = {"image_valence": "image_path", "donor_valence": "donor_path",
         "recipient_valence": "recipient_path"}
# Files whose plain `valence` column is EMOTIC's 1-10 rating rather than the model's score.
EMOTIC_VALENCE_FILES = {"stage_c/caption_readout.parquet", "stage_c/caption_readout_rich.parquet"}

# Where the camera-ready paper uses each file. Anything absent is released but marked not cited.
CITED = {
    "stage_a/metrics.json": "Table 11 (probe localization)",
    "stage_a/probes.npz": "frozen probe, Sec. 4.1 and App. E",
    "stage_a/steering_v2_metrics.json": "Table 12 (text-only steering)",
    "stage_c/metrics_hf.json": "Sec. 4.1, App. E.2 (rho = +0.510, random-direction null)",
    "stage_d/steering_metrics_hf.json": "Table 13 (cross-modal steering)",
    "stage_f/conflict_qwen.parquet": "Table 1, Tables 3 and 9 (Qwen3-VL-8B, varied set)",
    "stage_f/conflict_qwen3-vl-8b-instruct_minimal.parquet": "Table 2, Tables 4-5 (matched set)",
    "stage_f/matched_readouts.json": "Table 2 and Sec. 4.2.2 (every readout x contrast)",
    "stage_f/conflict_gemma-3-4b-it.parquet": "Tables 3 and 9 (Gemma-3-4B)",
    "stage_f/conflict_llava_sequence.parquet": "Tables 3 and 9 (LLaVA-1.5-7B, complete-label)",
    "stage_f/conflict_llava-v1-6-mistral-7b-hf_sequence.parquet":
        "Tables 3 and 9 (LLaVA-NeXT-7B, complete-label)",
    "stage_f/unbounded_crossed.json": "Table 9 (four models, three readouts)",
    "stage_f/label_balance_sensitivity.json": "Sec. 4.3 (balanced label subsets)",
    "stage_f/text_only_qwen3-vl-8b-instruct.parquet": "App. G (text-only control, Qwen)",
    "stage_f/text_only_minimal.parquet": "App. G (text-only control, matched pairs)",
    "stage_f/text_only_gemma-3-4b-it.parquet": "Sec. 4.3 (no-image prior, Gemma)",
    "stage_f/text_only_llava_sequence.parquet": "Sec. 4.3, App. G (LLaVA-1.5-7B)",
    "stage_f/text_only_llava-v1-6-mistral-7b-hf_sequence.parquet": "Sec. 4.3, App. G (LLaVA-NeXT)",
    "stage_f/conflict_qwen3-vl-8b-instruct_px448.parquet": "Table 10 (resolution sweep)",
    "stage_f/conflict_qwen3-vl-8b-instruct_px896.parquet": "Table 10 (resolution sweep)",
    "stage_f/conflict_qwen3-vl-8b-instruct_px1344.parquet": "Table 10 (resolution sweep)",
    "stage_f/controls_person_box_qwen.parquet": "App. A (person grounding, box)",
    "stage_f/controls_person_crop_qwen.parquet": "App. A (person grounding, crop)",
    "stage_f/controls_frame_qwen.parquet": "App. B, Table 6 (wording: frame)",
    "stage_f/controls_question_qwen.parquet": "App. B, Table 6 (wording: question)",
    "stage_f/controls_generate_qwen.parquet": "App. on generated answers (first, void run)",
    "stage_f/controls_generate_qwen_t32.parquet": "App. on generated answers (second, void run)",
    "stage_f/patching_hf.parquet": "Table 7 (same-image patching, pair 2)",
    "stage_f/patching_hf_pair1.parquet": "Table 7 (same-image patching, pair 1)",
    "stage_f/cross_patching_hf.parquet": "Table 8 (cross-image patching, 18-28)",
    "stage_f/cross_patching_hf_metrics.json": "Table 8 (cross-image patching)",
    "stage_f/patching_qwen.parquet": "mechanism replication table (Qwen3-VL-8B)",
    "stage_f/patching_llava_sequence.parquet": "mechanism replication table (LLaVA-1.5-7B)",
    "stage_f/layerwise_hf.parquet": "App. on where the context signal appears",
    "stage_f/arbitration_hf.parquet": "App. E.4 (steering under conflict)",
}
# Results of the camera-ready Colab runs; cited once they exist.
CITED_IF_PRESENT = {
    "stage_f/controls_neutral_bank_qwen.parquet": "App. on the neutral baseline (bank B)",
    "stage_f/equal_room_screen_qwen.parquet": "App. on the equal-room comparison (screen)",
    "stage_f/equal_room_qwen.parquet": "Table 2, App. on the equal-room comparison",
}


def photo_key(path) -> str:
    return "/".join(str(path).split("/")[-3:])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _emotic_index(emotic: pd.DataFrame) -> dict:
    """(photo key, annotation valence) -> first matching row position in emotic_test.parquet."""
    keys = emotic["image_path"].map(photo_key)
    index = {}
    for pos, (k, v) in enumerate(zip(keys, emotic["valence"].astype(float))):
        index.setdefault((k, v), pos)
    return index


def sanitize(df: pd.DataFrame, rel: str, index: dict) -> tuple[pd.DataFrame, list[str]]:
    strip = dict(STRIP)
    if rel in EMOTIC_VALENCE_FILES:
        strip["valence"] = "image_path"
    stripped = []
    out = df.copy()
    for col, path_col in strip.items():
        if col not in out.columns:
            continue
        rows = [index.get((photo_key(p), float(v))) for p, v in zip(out[path_col], out[col])]
        if any(r is None for r in rows):
            raise SystemExit(f"{rel}: {sum(r is None for r in rows)} rows of {col} match no EMOTIC "
                             f"annotation; refusing to release a column that cannot be rehydrated")
        out[f"{col}_emotic_row"] = rows
        out = out.drop(columns=[col])
        stripped.append(col)
    return out, stripped


def main() -> None:
    if not EMOTIC.exists():
        sys.exit(f"{EMOTIC} missing: the row index is built against it")
    index = _emotic_index(pd.read_parquet(EMOTIC))
    if DST.exists():
        shutil.rmtree(DST)
    manifest = []
    for src in sorted(p for p in SRC.rglob("*") if p.is_file() and p.name != ".DS_Store"):
        rel = src.relative_to(SRC).as_posix()
        dst = DST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        stripped = []
        if src.suffix == ".parquet":
            df, stripped = sanitize(pd.read_parquet(src), rel, index)
            df.to_parquet(dst)
        else:
            shutil.copy2(src, dst)
        where = CITED.get(rel) or CITED_IF_PRESENT.get(rel)
        manifest.append({"path": rel, "sha256": _sha256(dst), "bytes": dst.stat().st_size,
                         "cited_in_paper": where is not None, "where": where,
                         "emotic_columns_stripped": stripped})
    missing = sorted(set(CITED) - {m["path"] for m in manifest})
    if missing:
        sys.exit(f"cited files missing from results/: {missing}")
    (DST.parent / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    n_cited = sum(m["cited_in_paper"] for m in manifest)
    n_strip = sum(bool(m["emotic_columns_stripped"]) for m in manifest)
    print(f"{len(manifest)} files -> {DST} ({n_cited} cited, {n_strip} with EMOTIC columns stripped)")


if __name__ == "__main__":
    main()
