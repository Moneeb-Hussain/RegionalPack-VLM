# Is product identity preserved inside a VLM when its final answer is wrong?
Pre-specified analysis plan for a linear-probe study on RegionalPack-VLM (Qwen3-VL-2B-Instruct).
Commit this file BEFORE running any probe, so the hypotheses and decision rules are timestamped.

## 1. Background and question
In the original evaluation, the model sometimes read a product's name correctly (transcription) but failed to
identify the product (identification). Two readings are possible:
- (A) The information needed to identify the product is lost or never formed inside the model.
- (B) The information is present in the model's internal representation, but the final answer does not use it.

Question: when the model's identification answer is wrong, can a simple linear probe still recover the product's
identity from the model's hidden states, on photos taken under conditions the probe was not trained on?

A second question connects to representation geometry (nuisance vs semantic variation):
the three capture conditions (straight, tilt, occluded) are nuisance variation of the same product, and
different products are semantic variation. Do nuisance shifts move the representation along directions the
probe finds identity-sensitive, and does the direction of a shift explain lost margin better than its size?

## 2. Data
60 photos, 20 products, 3 conditions each. 18 primary products (9 international/local pairs) and 1 exploratory
pair (apple juice) which is included in probe training but excluded from the model-failure analyses.
Model correctness per photo comes from the FROZEN Set A scorer. Do not rescore. Failed answers include
abstentions ("declined to name the product line"); record failure_type if available.

## 3. Features (sites)
- `vision_out`: mean-pooled vision tower output (what the language model receives).
- `llm_img_L{l}`: mean hidden state over image tokens at LLM layer l (the "evidence").
- `llm_last_L{l}`: hidden state at the last prompt token at layer l (the "decision" position).
- Controls with no model: `pixel_thumb` (16x16 RGB), `color_hist`.
Primary (confirmatory) sites, fixed in advance: vision_out, llm_img at the middle and last layer, llm_last at the
middle and last layer. Everything else (the layer sweep) is exploratory and labelled as such.

## 4. Probe protocol
Multinomial logistic regression (L2, C = 1.0, standardised features), 20 classes.
Leave-one-condition-out: train on two conditions, test on the third. Photos of one condition share lighting and
background, so random photo splits would overstate performance. Report C in {0.1, 1, 10} as a sensitivity check;
do not tune C on test folds.
Null: shuffle training labels within each training condition, test on true labels, 1000 permutations.
Uncertainty: bootstrap over products (not photos), 2000 resamples.

## 5. Hypotheses and decision rules
H1 Decodability. LOCO accuracy at a primary site exceeds the permutation null (p < 0.01, Bonferroni over 5 sites).
   Interpretation guard: if `vision_out` is as decodable as the LLM sites, the information is simply visible in
   the image features, which says little about the language model. If pixel/colour controls are also high,
   the probe is reading trivial cues and H1 is not interesting.
H2 Recoverable when wrong. Among model-wrong photos (primary products), probe accuracy exceeds its null (p < 0.01),
   and the exact McNemar test shows probe and model errors differ. Report the raw count (e.g. 9 of 14 recovered).
H3 Direction over distance. For nuisance shifts (straight -> tilt/occluded), lost probe margin correlates more with
   alignment (cosine between the shift and the margin-reducing direction) than with shift size. Margin drop =
   |g| |d| alignment is an exact identity, so the empirical question is how large the alignment values are and
   whether they rank-order margin loss. Report the bootstrap CI of corr(align, drop) - corr(dist, drop).
H4 (exploratory, few failures) Alignment or a probe-defined Fisher-style sensitivity predicts which photos the
   model fails on, compared with distance (AUROC). Temperature is fixed by nested cross-validation inside the
   training conditions, and sensitivity to temperature is reported (this was unstable in the earlier check).

## 6. What each outcome allows you to say
| Outcome | Allowed claim | Not allowed |
| --- | --- | --- |
| H1 and H2 hold, vision_out lower than LLM sites | Identity is linearly decodable from the LM's own states even when the answer is wrong, beyond what the vision tower exposes | "The model knows the product" or "the model uses it" |
| H1 and H2 hold, vision_out equal | Identity is available in the visual features; the failure arises later or in readout | Anything specific to the language model |
| H1 holds, H2 fails | Decodable overall, but not for the cases that matter; failures look like lost information | "Information is preserved" |
| H1 fails | Identity is not linearly decodable across conditions at this scale | "Information is absent" (probes are one lens, n is small) |
| H3 holds / fails | State the effect with its CI, either way | Causal language |

## 7. Limits (state them in the write-up)
- 60 photos, 20 products, one model, one annotator, three conditions that mix viewpoint and occlusion.
- Closed-set: every test product also appears in training under other conditions. This is not open-set
  recognition and says nothing about unseen products.
- Linear decodability is evidence that information is present, not that the model uses it. A causal test would
  intervene on the representation (out of scope here).
- The probe-defined Fisher-style quantity is a proxy inspired by Fisher-IRG, not an implementation of it.
- Exploratory layers are exploratory. Do not pick the best layer and present it as confirmatory.

## 8. Optional extension (decide after the primary run, record the decision)
Add 2 conditions (low light, glare) for the same 20 products (100 photos) and replicate on a second small VLM.
Only do this if the primary effect sizes are interesting enough to justify it.

## 9. Outputs
results/decodability_by_site.csv, results/geometry_summary.json, results/geometry_<site>.csv,
results/layerwise_decodability.png, and a 1-page README with the table of outcomes in section 6.
