# Transcription prompt provenance

I left the three source files below unchanged. This note records which wording I sent.

| File | Role | Wording |
|---|---|---|
| `docs/research_spec.md` | Historical specification. Also stored, byte for byte, at `RegionalPack-Cursor-Spec/docs/research_spec.md`. | Prompt 2 still reads: "Transcribe all text visible on the packaging exactly as written, including any text that is not in English. If no text is legible, say so." |
| `prompts/phase0_prompts.yaml` | Prompt sent in Phase 0. | The same all-text sentence. Phase 0 responses stay attached to it. |
| `prompts/prompts.yaml` | Prompt sent in the benchmark. | "Transcribe only the text on the front of the package exactly as written, including any text that is not in English. Include the brand wordmark, the product or flavor name, and any prominent tagline on the main face. Do not transcribe ingredient lists, nutrition tables, barcodes, or regulatory fine print. If no front-of-pack text is legible, say so." |

The specification's scoring rubric already limited the scored reference to the front-of-pack brand, product or flavor, and prominent tagline. The Prompt 2 string in the specification was not rewritten to match that rubric.

The benchmark prompt in `prompts/prompts.yaml` was fixed before the 180-call evaluation. It was not changed after `results/benchmark/benchmark_predictions.jsonl` was saved. Identification and category match the specification text. The saved environment record quotes the front-of-pack transcription prompt that was sent.

In the publication folder that benchmark prompt is [`../configuration/prompts.yaml`](../configuration/prompts.yaml). The historical specification and `prompts/phase0_prompts.yaml` remain in the git repository.
