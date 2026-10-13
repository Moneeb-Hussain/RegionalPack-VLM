# Publication ready checklist

Version 0.1. 12 October 2026. This folder is the release synced to `moneebhussain/RegionalPack-VLM`.

The folder is `release/v0.1/publication/`.

This release is a Hugging Face Bucket. The Bucket README is plain Markdown. It does not use dataset-card metadata.

| Requirement | Result | Evidence |
|---|---|---|
| Bucket CLI | PASS | The Bucket is `moneebhussain/RegionalPack-VLM`, updated with `hf buckets sync`. |
| Required-file build | PASS | `scripts/assemble_v01_bucket.py` raises before writing a folder or `SHA256SUMS` if any required file or the 60 photographs are missing. |
| GitHub README paths | PASS | `README.md` keeps repository paths. Relative links resolve in the repository. |
| Bucket README paths | PASS | `release/v0.1/README.md` is copied to the Bucket root. Its relative links resolve inside `release/v0.1/publication/`. |
| Published Markdown links | PASS | Every relative Markdown link in the publication tree resolves. |
| Photographs | PASS | 60 JPEGs, 20 products, 20 straight, 20 tilt, 20 occluded. |
| Predictions | PASS | 180 responses, 60 images, 180 image/prompt pairs, output flag `ok` on all 180. |
| Annotations | PASS | 60 rows, all `verified`. Manifest: 20 products. |
| Scoring files | PASS | `scored_predictions.csv` 60 rows, `metrics_summary.json`, `adjudication_decisions.csv` 60 rows, `v0.1_summary.md`. Set A remains 13/18. Set B remains 15/18, post hoc. |
| Reports and figures | PASS | Technical report, revised analysis, OCR interpretation, prompt provenance, adjudication checklist, two figures, captions. |
| JSONL SHA-256 | PASS | Source and publication copy match `e80a439149189eb57e8e08e55de112667562bf5939488ceebbceb261d2434eff`. `SHA256SUMS` matches every other file in the folder. |
| Five Approve boxes | PASS | All five images have Approve checked and Reject unchecked: `biscuit_local/tilt`, `detergent_local/straight`, `detergent_local/occluded`, `noodles_local/straight`, `noodles_local/tilt`. The file was not regenerated. |
| Reviewer name and date | PASS | Status: approved. Reviewer: Moneeb Hussain. Date: 2026-10-12. The withdrawn-signature and “approval is pending” sentences are gone. |
| EXIF location | PASS | GPS on 0 of 60 photographs. No artist, copyright, or comment fields. |
| Secrets in the publication folder | PASS | No tokens, passwords, or home-directory paths in the folder. |
| License and package artwork | WARNING | No `LICENSE` file. I do not grant an open license. Brand names and pack designs remain their owners' trademarks. |

Overall: **WARNING** for licensing only. Human approval is complete: five Approve boxes, status approved, reviewer Moneeb Hussain, date 2026-10-12. Set A remains 13/18. Set B remains 15/18, post hoc. The Bucket is `https://huggingface.co/buckets/moneebhussain/RegionalPack-VLM`.
