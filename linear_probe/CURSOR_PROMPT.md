# Prompt for Cursor (paste as one message, with the repo open)

You are helping me run a pre-specified linear-probe study on my RegionalPack-VLM project. Read PROTOCOL.md first
and follow it exactly. Do not change hypotheses, probe settings or primary sites.

Files already written (put them in a new folder `linear_probe/`): PROTOCOL.md, probe_analysis.py (tested on
synthetic data), run_study.py (tested on synthetic features), extract_features.py (UNTESTED against the real model).

Tasks, in order:
1. Find the existing results files from the original evaluation and build `manifest.csv` with the columns
   described at the top of extract_features.py. `model_correct` must come from the FROZEN Set A scorer output,
   per photo. Do not rescore anything. Print a table of the 60 rows and the count of model-wrong photos among
   the 18 primary products so I can check it.
2. Find the exact frozen identification prompt string used in the evaluation and paste it into IDENT_PROMPT
   in extract_features.py. It must be byte-identical.
3. Run extract_features.py with --limit 2 on the Colab T4. Fix any transformers API mismatches (Qwen3-VL class
   names, image_token_id, the vision tower hook). Check that hidden_states has one entry per layer + 1 and that the
   image-token mask selects the expected number of tokens. Then run on all 60 photos and save features.npz.
4. Run: python run_study.py --features features.npz --manifest manifest.csv --out results --n-perm 1000
5. Do NOT tune anything after seeing results. If something looks wrong (a crash, accuracy exactly 1.0 everywhere,
   NaN), find the bug, tell me, and rerun the same protocol.
6. Write results/README.md with: the table from PROTOCOL.md section 6 filled in honestly, the 5 primary sites
   with accuracy, 95% CI, null mean and p-value, the controls, the count of model-wrong photos the probe recovered
   (and which ones), and the limits from section 7. No claim beyond what the table allows.
7. Commit on a branch named `linear-probe`. Do not merge to main.
