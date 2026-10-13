# RegionalPack-VLM benchmark analysis

One saved run of Qwen3-VL-2B-Instruct. Descriptive counts only. This is not a test that geography, or pretraining frequency, caused the misses.

## Headline rates

Brand and category use the frozen alias list and the frozen category phrases. A product is correct when at least 2 of its 3 photographs succeed. The apple-drink pair is excluded.

| Rate | Result | Wilson 95% interval |
|---|---|---|
| Primary brand, product majority | 13/18 (72%) | 49%–88% |
| Primary category, product majority | 14/18 (78%) | 55%–91% |
| Primary brand, image | 42/54 (78%) | not used |
| Primary category, image | 43/54 (80%) | not used |

The interval treats 18 products as independent draws. It describes uncertainty for this sample. It is not a significance test. Image-level intervals are omitted because the three photographs of one product are not independent.

International products, brand majority: 9/9 (100%). Pakistan-market products, brand majority: 4/9 (44%).

Five local product-brand majorities fail under the frozen rule. Zero international product-brand majorities fail. That split is a description of these 18 products. It is not evidence of a geographic mechanism. Three of the five local misses still contain the product-brand name, blocked by a comma or quotation marks, and are waiting for human adjudication.

## Pairs

| Category | International brand | Pakistan-market brand | International category | Pakistan-market category |
|---|---|---|---|---|
| cola | correct | correct | correct | correct |
| biscuit | correct | incorrect | correct | correct |
| tea | correct | correct | correct | correct |
| chocolate | correct | incorrect | correct | correct |
| chips | correct | incorrect | correct | correct |
| detergent | correct | incorrect | correct | correct |
| toothpaste | correct | correct | incorrect | incorrect |
| noodles | correct | incorrect | correct | correct |
| soap | correct | correct | incorrect | incorrect |

Exploratory apple drinks, not in the rates above: apple_fruita brand correct, category correct, apple_fruitien brand correct, category correct.

## Capture condition

straight, tilt, and occluded are phone conditions, not a controlled viewpoint experiment.

| Condition | Brand | Category |
|---|---|---|
| straight | 13/18 (72%) | 14/18 (78%) |
| tilt | 15/18 (83%) | 14/18 (78%) |
| occluded | 14/18 (78%) | 15/18 (83%) |

## Abstention and adjudication

Primary identification responses: {'asserted_alias': 42, 'punctuation_blocked_assertion': 6, 'abstention': 6}.

An explicit statement that the specific brand cannot be identified is an abstention. A name quoted inside that statement is not a brand success. On this run the frozen matcher already rejected those quoted names, because a quotation mark stays attached to the word. The abstention label records why. It did not flip any frozen hit to a miss.

Images waiting for a person: 6. The model names the accepted brand, but a comma or quotation mark stops the frozen word match. Those images stay misses in the headline rate.

If a reviewer later accepts those assertions, and still rejects abstentions, biscuit Rite, Bonus Tristar, and Shoop would become product-level brand successes. CandyLand YOU and Super Crisp would not. That sensitivity is not the reported rate.

## Category note

Soap and toothpaste fail the frozen category list when the answer is “cleaning product.” That phrase is an example in the category prompt, and it is an accepted phrase only for detergent. The list was not expanded after seeing these answers.

Chocolate answers that also say “dairy” still match, because the frozen list accepts “chocolate” and the word is present. The dairy/snack wording is flagged, not rescored.

## OCR

Normalization is NFC, collapsed whitespace, and Latin casefold. Punctuation stays. Urdu is not casefolded. Empty ground truth is not a miss. There is no Urdu word-error rate. Extra model text raises character error rate, and the rate can exceed 1.

| Script | Images | Mean CER | Median CER | Mean WER |
|---|---|---|---|---|
| Latin | 60 | 5.22 | 0.69 | 3.98 |
| Urdu | 9 | 4.91 | 2.14 | not computed |

Latin mean CER without the repeated Knorr line: 1.56 on 59 images. That exclusion is a sensitivity, not the headline OCR number.

Urdu ground truth with no Urdu characters in the response: 3 images, all three Shoop photographs.

Repeated-line transcriptions: noodles_intl/tilt.

Prices, badges, and regulatory lines were left out of the human transcription on purpose. When the model adds them, character error rate counts the extra characters. They were not removed after the run.

Figures: `reports/figures/brand_by_pair.svg`, `reports/figures/brand_by_capture.svg`.
