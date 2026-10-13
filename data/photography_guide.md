# Photography guide

Use one smartphone for every product. Record the phone model once in the dataset notes. The packs are in hand. Shoot the three frames before evaluation.

## Setup

A category may use its own place. Soap can be shot in one spot and chips in another. Both products in a pair use that same place.

1. For each pair, choose one plain surface and keep it for both packs.
2. Use the same light for both packs in the pair: daytime indoor light from one window, or one lamp in the same place. Do not use the flash. Do not mix window light and lamp light inside a pair.
3. Turn off portrait mode, beauty mode, and filters. Leave HDR off if the phone would otherwise change the color from one shot to the next.
4. Use the main camera for every product. Do not switch to the wide camera inside a pair.
5. Write the shared place in `photo_location` on both manifest rows for that category before you shoot the second pack. The two values must match.

## Three photographs

Stand the package so the front faces you. Choose the camera distance for the straight shot and keep that distance for the other two. Take all three before you put the package away.

| File | What changes | What stays |
|---|---|---|
| `data/raw/<product_id>/straight.jpg` | Front parallel to the phone. The whole front is visible, with a small margin of background. | This distance is the distance for all three frames. |
| `data/raw/<product_id>/tilt.jpg` | Turn the package 30 to 45 degrees so the right edge is farther from the phone. | Same distance, same light, same camera height. |
| `data/raw/<product_id>/occluded.jpg` | Cover the lower-left corner with a hand, a second grocery item, or the frame edge. | Same distance as the straight shot. Do not step closer to make the occlusion. |

The occluded frame changes the obstruction only. Camera distance stays the distance used for `straight.jpg`.

Do not cover the whole brand name on purpose, and do not move the obstruction because the text looks hard to read.

`<product_id>` is the id in [product_manifest.csv](product_manifest.csv), for example `tea_local`.

## Do not

- Do not crop, filter, or retouch after the shot.
- Do not delete a frame because the text is small, tilted, or partly covered, and then shoot an easier one.
- Do not run the model and then replace a photograph.
- Do not copy the six Phase 0 photos into `data/raw/`. Those stay in `RegionalPack-Phase0-Labeled-Photos`.

## Annotation

Annotate into [annotations.csv](annotations.csv) from the photograph and the package in front of you. One row per image. Leave the file header-only until the photographs exist.

| Column | What to write |
|---|---|
| `image_path` | `data/raw/<product_id>/<angle>.jpg` |
| `product_id`, `category`, `brand_true`, `brand_group`, `origin` | The same on all three rows for that product. `brand_true` is the brand on the package. Covering the wordmark does not change it. |
| `script_present` | `latin`, `urdu`, `roman_urdu`, or `mixed` for the package, not only for the visible corner. |
| `text_true_latin`, `text_true_urdu` | Front-of-pack words visible **in that image**. Latin and Urdu are separate. |
| `angle` | `straight`, `tilt`, or `occluded` |
| `occlusion_note` | What the occluder covers. Empty when nothing is covered. |
| `notes` | Faded text, a second spelling, or a script that is neither Latin nor Urdu. |

- Accepted spellings for later brand scoring are already listed in `configs/brand_aliases.yaml`. Do not add a spelling after seeing a model answer.
- A visible script that is neither Latin nor Urdu goes in `notes`. It does not go in `text_true_latin` or `text_true_urdu`, and it is not scored as Latin or Urdu OCR.
- If the occluder hides a word, leave that word out of the occluded row. Do not copy it from the straight photograph. Hidden text is not a failed transcription.
- Ingredient lists, nutrition tables, barcodes, and regulatory fine print stay blank even when the camera can see them.
