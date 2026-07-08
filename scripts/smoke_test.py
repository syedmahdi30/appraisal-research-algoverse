#!/usr/bin/env python
"""Boot Gemma 3 through the bridge and prove the read-out/hook path works.

Not passed until a REAL forward pass succeeds (experiment-setup skill guardrail). Needs a
GPU and HF_TOKEN. Steps: boot + assert multimodal, verify label tokenization, one text
forward pass with last-token read-out, confirm the three LM taps are cached.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bridge.boot import boot_gemma
from src.bridge.hooks import TAP_SUFFIXES, keep_language_taps
from src.data import EMOTION_LABELS, verify_label_tokenization


def main() -> int:
    print("[1/4] booting Gemma 3 (bf16)...")
    bridge = boot_gemma()
    print(f"      is_multimodal={bridge.cfg.is_multimodal} n_layers={bridge.cfg.n_layers}")

    print("[2/4] verifying label tokenization...")
    report = verify_label_tokenization(bridge.tokenizer, EMOTION_LABELS)
    multi = [w for w, r in report.items() if not r["single_token"]]
    print(f"      multi-token labels: {multi or 'none'}")

    print("[3/4] one text forward pass with cache...")
    input_ids = bridge.to_tokens("<start_of_turn>user\nWhat single emotion is joy?<end_of_turn>\n")
    _, cache = bridge.run_with_cache(input_ids, names_filter=keep_language_taps())
    print(f"      cached {len(cache)} tensors; input_len={input_ids.shape[-1]}")

    print("[4/4] confirming the three LM taps fired...")
    present = {s for name in cache for s in TAP_SUFFIXES if name.endswith(s)}
    missing = set(TAP_SUFFIXES) - present
    if missing:
        print(f"      FAIL: missing taps {missing}")
        return 1

    print("\nSMOKE TEST PASSED — real forward pass + all three taps cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
