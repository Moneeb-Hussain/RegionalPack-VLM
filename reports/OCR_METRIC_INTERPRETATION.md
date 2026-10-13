# OCR metric interpretation

The Latin and Urdu character error rates in `results/scored_predictions.csv` are unchanged. Nothing in the model output was deleted, and the human transcriptions were not edited.

## Why character error rate can exceed 1

Character error rate is the edit distance from the human transcription to the model transcription, divided by the number of characters in the human transcription. Insertions add to the distance. They do not add to the denominator. A long extra string, a price, a nutrition line, or a repeated sentence can make the distance larger than the reference, so the rate is greater than 1.

Punctuation is kept. Latin text is casefolded after Unicode NFC, and whitespace is collapsed. Urdu is not casefolded. An empty human transcription is not scored. There is no Urdu word-error rate.

## Extra packaging text

The human transcription stops at the brand, the product or flavor, and the prominent tagline. Prices, badges, and regulatory lines were left out on purpose. If the model transcribes those lines, every extra character is an insertion. The original rate counts them. They were not stripped after the run.

## Repeated generation and the mean

Latin mean character error rate: 5.22 on 60 images. Median: 0.69.

The mean without images whose transcription repeats a line at least four times: 1.56 on 59 images. The Knorr tilt transcription is that repeated output. Removing it from the mean is a sensitivity. The original mean still includes it.

Highest Latin rates in this run:

| Image | Latin CER | Repeated-line count |
|---|---|---|
| `noodles_intl/tilt` | 221.00 | 46 |
| `noodles_intl/occluded` | 13.60 | 1 |
| `noodles_intl/straight` | 9.00 | 1 |
| `detergent_intl/tilt` | 4.80 | 1 |
| `detergent_intl/occluded` | 4.30 | 1 |

## Proposed secondary split

This split was written after the run, once extra text and the repeated Knorr line were visible. It is not a replacement endpoint. It is a uniform rule for a later reading of the same strings.

Using the same normalization as the character error rate:

1. Reading: a reference word is read if it occurs in the model transcription. Coverage is the share of reference words that are read. A reading error is a reference word that is absent or replaced.
2. Scope: extra words are model words that are not reference words. Prices, nutrition lines, and repeated sentences land here. A scope error can exist even when every reference word was read.

The two counts stay separate. Incorrect text is not deleted to improve either count, and the human reference is not shortened to match the model.

The original character error rates remain the OCR record for this submission until that secondary count is computed on every image under the rule above.
