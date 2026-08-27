# Abstract and keywords — VLM4RWD (8pp build)

_Generated 2026-08-27 against `2c52946`. **The zh-TW abstract is not submittable to
VLM4RWD**, which is English-only; it is included because the mode produces it, and is
useful only for a bilingual CV, thesis chapter, or departmental record._

## English (revised, 257 words)

Vision-language systems increasingly read images of people alongside text that a user
supplies, and the two can disagree. When they disagree about emotional valence, does the
model treat a positive and a negative sentence as equally strong? We pair EMOTIC
photographs with one-sentence contexts and measure how a conflicting context shifts the
model's emotion judgment away from a neutral-context baseline. In the most controlled
condition we hold the photograph and the described event fixed and flip a single valence
word (won/lost, wonderful/devastating). On positive images, negative wording moves
Qwen3-VL-8B's judgment four to five times farther than positive wording; all six pairs
agree, and the within-item contrast is +1.148. Run without an image, the same six pairs
show no detectable difference, which argues against a large imbalance in sentence strength
but cannot exclude a small one. A set of six unrelated events points the same way but does
not survive resampling the events. In Gemma-3-4B, where a text-trained valence probe is
available, replacing downstream states at text positions restores 88 to 93% of the context
difference, and a direction estimated only from valenced text still moves the answer under
conflict. Across four models the conclusion depends on the summary score and on how
multi-token labels are scored: scoring only a label's first piece manufactures a null in
one model and reverses the categorical ordering in another. Negative text can therefore
exert disproportionate influence on image judgments, though the measured size depends on
model and measurement. Evaluations should test what each cue says, not only which modality
carries it.

**Changes from the current abstract.** Two of Sneheel's open comments are fixed here.
Comment 1 ("what does strongest test mean? weird phrasing"): the superlative is replaced by
"in the most controlled condition", which says what is controlled instead of asserting rank.
Comment 3 ("positive vs negative in what dimension?"): the opening question now names the
dimension, "disagree about emotional valence". The closing sentence is tightened and the
deployment framing folded into the opening, since stating it twice was redundant.

## 中文（繁體）

視覺語言系統愈來愈常在閱讀人物影像的同時，一併處理使用者提供的文字，而兩者可能互相矛盾。
當影像與文字在情緒效價上不一致時，模型是否會將正向語句與負向語句視為同等強度？本研究將
EMOTIC 影像與單句情境配對，測量矛盾情境如何使模型的情緒判斷偏離中性情境基線。在控制最嚴格
的條件下，我們固定影像與所描述的事件，僅更動一個效價詞（won/lost、wonderful/devastating）。
在正向影像上，負向措辭使 Qwen3-VL-8B 的判斷偏移幅度達正向措辭的四至五倍；六組配對方向一致，
項目內對比為 +1.148。若移除影像，同樣的六組配對未顯示可偵測的差異，這可排除語句強度存在大幅
失衡的可能，但無法排除小幅失衡。另一組六個不相關事件的結果方向相同，惟在對事件重新抽樣後未
能維持。在 Gemma-3-4B 上，由於可取得以文字訓練的效價探針，將文字位置的下游狀態進行替換可回復
88% 至 93% 的情境差異，且僅由帶效價文字估計出的方向，在衝突情況下仍能改變模型的答案。然而
在四個模型之間，結論取決於所採用的彙總指標，以及多詞元標籤的計分方式：僅計分標籤的第一個詞元
會在其中一個模型上製造出虛假的零效果，並在另一個模型上使類別排序反轉。因此，負向文字確實可能
對影像判斷產生不成比例的影響，但其效果量的大小取決於模型與測量方式。評估應檢驗每個線索所傳達
的內容，而不只是哪一個模態承載了訊號。

## Keywords

**EN:** vision-language models; multimodal conflict; emotional valence; negativity
dominance; activation patching; linear probes; evaluation validity

**zh-TW:** 視覺語言模型；多模態衝突；情緒效價；負向優勢；激活修補；線性探針；評估效度
