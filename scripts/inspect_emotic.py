#!/usr/bin/env python
"""Inspect EMOTIC Annotations.mat to confirm its structure before writing the loader.

Prints the top-level split keys, the field names of one image struct and one person
struct, and one concrete decoded example (image path, bbox, categories, VAD). This lets
us verify field names and the folder-path convention against the real file.

Usage (Colab, after uploading Annotations.mat to Drive):
    !python scripts/inspect_emotic.py --mat /content/drive/MyDrive/Annotations.mat
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.io as sio


def _fields(obj) -> list[str]:
    return list(getattr(obj, "_fieldnames", [])) or [n for n in dir(obj) if not n.startswith("_")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True, help="path to Annotations.mat")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    mat = sio.loadmat(args.mat, squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]
    print(f"top-level keys: {keys}")

    data = np.atleast_1d(mat[args.split])
    print(f"\nsplit '{args.split}': {len(data)} images")

    img = data[0]
    print(f"\nimage struct fields: {_fields(img)}")
    for f in _fields(img):
        v = getattr(img, f)
        if f != "person":
            print(f"  {f!r}: {v!r}")

    persons = np.atleast_1d(img.person)
    print(f"\nfirst image has {len(persons)} person(s)")
    p = persons[0]
    print(f"person struct fields: {_fields(p)}")
    for f in _fields(p):
        v = getattr(p, f)
        # categories/continuous are nested structs — show their fields too.
        if hasattr(v, "_fieldnames"):
            print(f"  {f!r} -> nested fields {_fields(v)}")
            for sub in _fields(v):
                print(f"      {sub!r}: {getattr(v, sub)!r}")
        else:
            print(f"  {f!r}: {v!r}")


if __name__ == "__main__":
    main()
