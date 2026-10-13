# Revision changelog

12 October 2026. Prose revision of the README and the technical report. Scores, model outputs, annotations, prompts, and scoring configs were not changed. Nothing was uploaded.

| Item | Correction |
|---|---|
| Central finding | The lead result is now the disagreement between front-of-pack transcription and specific product-line identification, with the CandyLand YOU and Super Crisp responses as the evidence. |
| H2 | Removed the claim that legibility does not account for the brand gap. The report now says transcription error cannot explain every identification failure, and that frequency, familiarity, packaging, and unmatched variants remain unresolved. |
| Brand labels | Added definitions of parent or umbrella brand, specific product line, and accepted answer, and applied them to Bisconni Rite and CandyLand YOU. Product rates stay 13/18 for Set A and 15/18, post hoc, for Set B. |
| Q1 | Rewrote the Fatima answer from the checkout project: YOLOv4-tiny, OpenCV, a moving conveyor, about 3,500 images, 70 products, and fixed-class recognition. |
| Q3 | Replaced the two-item next step with a four-part proposal: matched curation, separated OCR / candidate brand / abstention, a possible external similarity check, and held-out scoring of recognition and abstention. The proposal is marked as not implemented. |
| Citations | Replaced the unnamed related-work sentence with CultureVLM, the Culture Affordance Atlas, and the one-shot probe, each with the claim limited to what the paper states and a working arXiv link. |
| OCR | Stated the excluded small text, the extra-text penalty, the Knorr repetition at character error 221, the 9 Urdu images, and the uncomputed reading-versus-scope split. |
| Origin | Stated why 18 selected products, unmatched packaging, and brand familiarity block a causal reading of the origin counts. |
| Presentation | Added a workflow diagram and a five-row identification-versus-OCR table. Set A and Set B stay separate in the text and in the existing figures. |
| Q2 | The README now points at the JSONL, both score tables, the figures, and the technical report instead of treating the evaluation as a paragraph inside Q1. |
| Reproduction | The documented commands now create a virtualenv and install PyYAML, Pillow, and pytest. The previous `python3 -m pytest` line fails on a system Python that does not provide PyYAML. |
