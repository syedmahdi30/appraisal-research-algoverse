# Response to comments — VLM4RWD draft

_Written 2026-08-27. Covers Sneheel's Overleaf comments (25–26 Aug), his meeting
notes (26 Aug), and an external LLM review. Paper at `2c52946`; both venues build
(VLM4RWD 8pp, Interp4Discovery 5pp)._

**Read this first.** Five of the thirteen Overleaf comments are anchored to text that
no longer exists. The Overleaf project is several revisions behind `paper/neurips_2026.tex`.
Those five were already fixed by the restructure before the comments were written, so
they need no action, but the current build should be uploaded before the next review pass.

---

## Sneheel — Overleaf comments

| # | Comment | Status | Where |
|---|---|---|---|
| 1 | "what does strongest test mean? weird phrasing" | **Open** | Anchor was "The strongest test holds"; now reads "Our sharpest test". Still an unexplained superlative. |
| 2 | "reads fairly convoluted, need to simplify" (abstract) | **Done** | Opening sentence split in two; CI moved to body; deployment clause added. `8e7d0dc` |
| 3 | "positive vs negative in what dimension? spell it out" | **Open** | The intro question still reads "does positive text pull the model as strongly as negative text?" without naming the dimension. |
| 4 | "remove m-dashes" | **Done** | 4 → 0 in the rendered long build; checklist cleaned. `8e7d0dc` |
| 5 | "very LLM-like: get to the point" ("Our paradigm is simple.") | **Already gone** | Removed by the restructure. Stale anchor. `0278cfb` |
| 6 | "frame as contributions, not findings" | **Done** | "Four findings emerge" replaced by a three-item contributions list. `0278cfb` |
| 7 | "Why just 6??" | **Open — needs your input** | The paper concedes "Six pairs give little power" but never says why six. The honest answer is a construction constraint on token-matched single-word flips; only you can confirm. |
| 8 | "why in the conclusion? what does 'clean' imply?" | **Already gone** | "The behavioral result is cleanest…" removed. Stale anchor. `0278cfb` |
| 9 | "LLM-like sentence to end intro; belongs in Related Work" | **Done** | "We scope our claims carefully…" cut. Its substance already existed in Related work, so nothing was lost. `0278cfb` |
| 10 | "Verify if this is really true" (related-work claim) | **Partly** | Reworded to "None of it asks whether matched positive and negative text pull with equal force." `8e7d0dc`. **The underlying claim is still unverified** and an earlier version of it was retired twice as false. Needs the four cited papers checked. |
| 11 | "what does primary behavioral model / comparable weight mean?" | **Partly** | The label is gone; now "the main behavioral-evaluation model because image and text receive comparable regression weight." `0278cfb`. "Comparable weight" is still unglossed in the body (the β ratio of 1.14 appears later). |
| 12 | "why not fit the probe for each model? what is primary mechanism model?" | **Partly** | Label gone; the reason is now stated ("because we fit the text-trained probe on its states"). The Gemma-only question is answered below but is not yet in the paper. |
| 13 | "why are the probe readout images blank?" + "Stage F needs to be removed" | **Done** | Root cause: no probe exists for Qwen or LLaVA, so the column is all-NaN and the panel was drawn anyway. Panel now suppressed, internal stage name removed from the figure title. `9219c77` |

## Sneheel — meeting items

| Item | Status |
|---|---|
| Limitations to one paragraph | **Done** (`0278cfb`) |
| Structure: methods+experiments → results → discussion | **Done** — now 1 Intro, 2 Related work, 3 Experimental framework, 4 Results and analysis, 5 Discussion, 6 Conclusion (`0278cfb`) |
| Related work into main text | **Done** (`e6c3ce4`) |
| Related work still too long, combine paragraphs | **Done** — six paragraphs to three, 911 → 664 words, all 37 citation keys kept (`8e7d0dc`) |
| Three contributions max | **Done** (`0278cfb`) |
| "rather than" is LLM prose | **Done** — 25 → 5, keeping only cases where it is the precise contrast (`8e7d0dc`) |
| Readable to a newcomer, assumes some interp knowledge | **Partly** (`2c52946`) — see below |
| Full human end-to-end rewrite | **Not done — yours** |

### On the readability calibration

An audit of the long build found the vocabulary is mostly fine: the patching procedure is
already explained in plain language where it is performed (§4.3.1), and `residual stream`,
`logit lens` and `attn_out` appear nowhere in the main text. The problem was ordering.
"Activation patching" was used cold in the contributions list, about 5,000 characters before
§4.3 explains it, in the paragraph every reader reads. Now glossed there, and `semipartial`
is glossed in the body while keeping the term.

Still loose: `difference-of-means steering` and `readout` are each used shortly before being
defined, and `norm-matched random directions` is never explained as a control on direction
length.

### On mechanism analysis being Gemma-only

Worth answering carefully, because the tempting answer is wrong.

There **is** a Qwen replacement run (`patching_qwen.parquet`, 60 images): text positions
jointly restore **65.4%**, image tokens **0.0%**. It is tempting to promote that 0% as a
second-model replication of "image tokens are causally inert". **It is not.** That is a
same-image design, so the image tokens are identical between the two runs and patching them
is a no-op. The zero is guaranteed by construction, which the paper already says about the
Gemma rows.

What Qwen genuinely adds is the 65.4% text-position recovery, which does not depend on the
probe. So the defensible position is: *"text positions carry the context difference"* holds
on two models; *"which* text positions"* is Gemma-only, because the probe was fit on Gemma's
states and Qwen's position groups are non-additive (11.8 + 5.7 + 0 ≈ 17.5 against 65.4
jointly). That bounds the claim instead of undermining it.

## External LLM review

| Suggestion | Status |
|---|---|
| Deployment/faithfulness framing for the venue | **Done** — abstract clause, intro stakes, and a new Discussion paragraph using the corrected 57%/35% (`8e7d0dc`) |
| Three readouts explained before the numbers | **Done** — inline glosses; an itemised list cost a full page and was rejected |
| Conceptual opener before the §4.3 tables | **Done** (`8e7d0dc`) |
| Abstract opening sentence split | **Done** |
| "Fix the §?? broken reference" | **No action — claim is false.** Zero `??` in the build; no undefined references. |
| "Add sub-subheadings 4.2.1, 4.2.2" | **No action** — they already exist. |
| Flip checklist item 8 to Yes with "~14 A100-40GB GPU hours" | **Declined.** That number is invented. The checklist states wall-clock totals are unavailable and the compute appendix carries a TODO to extract them from session logs. |
| Flip item 12 with a supplied license list | **Declined as written.** The list is wrong at least for LLaVA-1.5-7B, which is Vicuna/Llama-2 derived and carries the Llama 2 Community License, not Apache 2.0. |

## Open items

1. **Manual end-to-end rewrite** (yours; Sneheel's primary request).
2. **Checklist item 12** — confirm licenses for EMOTIC, crowd-enVENT, Qwen3-VL-8B,
   Gemma-3-4B, LLaVA-1.5-7B, LLaVA-NeXT-7B from their model cards, then it flips honestly.
3. **Comment 10** — verify the related-work novelty claim against the four cited papers.
4. **Comments 1, 3, 7, 11** — small wording and justification gaps listed above.
5. **Upload `overleaf/vlm4rwd.zip`.** The project under review predates every change here.

## Note on the page budget

VLM4RWD sits at exactly 8pp with no slack. Every addition above was paid for by a cut,
mostly from Related work. Anything further needs a matching cut; §4 (2,268 words) and
Discussion (1,053) are the largest blocks.
