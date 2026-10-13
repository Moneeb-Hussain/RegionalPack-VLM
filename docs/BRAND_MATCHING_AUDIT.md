# Brand-matching audit

Primary metric: the response names the product brand. A parent name alone is not a hit. There is no second parent-company score. `parent_names` in `configs/brand_aliases.yaml` is metadata. `brand_match` reads `aliases` only.

Matching ignores case. The words must appear in order, as their own words. `YOU` does not match inside `Youthful`. `Nestlé` does not match `Fruita Vitals`.

Annotations, photographs, and the three prompts were not changed. The model was not loaded.

## Removed from the scored list

| Product | Removed | Still accepted |
|---|---|---|
| `noodles_local` | `Shan` | `Shoop`, `Shoop Chattpata`, `Shoop Chatpatta`, and phrases that also contain `Shoop` |
| `chocolate_local` | `CandyLand`, `Candyland` | `YOU`, `CandyLand YOU` |
| `apple_fruita` | none. `Nestlé` was never an alias by itself | `Fruita Vitals`, `Nestlé Fruita Vitals`, `Nestle Fruita Vitals` |

`Shoop Chattpata` was added because that is the spelling on the pack. `Shoop Chatpatta` stays as the other romanization. `Chattpata` alone does not count.

## All 20 products

| Product | Scored aliases | Parent metadata, not scored |
|---|---|---|
| `cola_intl` | Coca-Cola, Coca Cola, Coke | none recorded |
| `cola_local` | Cola Next, Next Cola | Mezan |
| `biscuit_intl` | Oreo | Mondelēz |
| `biscuit_local` | Rite | Bisconni |
| `tea_intl` | Lipton, Lipton Yellow Label | none recorded |
| `tea_local` | Tapal, Tapal Danedar | none recorded |
| `chocolate_intl` | Cadbury, Cadbury Dairy Milk, Dairy Milk | Mondelēz |
| `chocolate_local` | YOU, CandyLand YOU | CandyLand |
| `chips_intl` | Lay's, Lays | none recorded |
| `chips_local` | Super Crisp, SuperCrisp | Tripple-EM |
| `detergent_intl` | Surf Excel, Surf | none recorded |
| `detergent_local` | Bonus, Bonus Tristar | none recorded |
| `toothpaste_intl` | Colgate | none recorded |
| `toothpaste_local` | Medicam | none recorded |
| `apple_fruita` | Fruita Vitals, with or without Nestlé in the same phrase | Nestlé |
| `apple_fruitien` | Fruitien, Fruitien Joy | Citropak |
| `noodles_intl` | Knorr, Knorr Chicken | Unilever |
| `noodles_local` | Shoop, including Chattpata / Chatpatta and Shan Shoop | Shan |
| `soap_intl` | Lux | none recorded |
| `soap_local` | Capri | ZIL |

Parent names already written in the manifest were copied into `parent_names`. No other parent was guessed. Lipton, Tapal, Lay's, Surf, Bonus, Colgate, Lux, and Coca-Cola have no separate parent string in the manifest, so none was added.

## For human review

These are product-line shortenings, not parent companies. They still count. Say so before inference if any of them should require the longer name.

| Response | Currently | Why it is open |
|---|---|---|
| `Dairy Milk` without Cadbury | counts for `chocolate_intl` | Product line, not Mondelēz |
| `Coke` | counts for `cola_intl` | Product nickname, not a parent company |
| `Surf` without Excel | counts for `detergent_intl` | Printed wordmark. Unilever is not an alias |
| `Bonus` without Tristar | counts for `detergent_local` | Printed brand. The parent firm is not named in the manifest |

`Next`, `Danedar`, `Excel`, `Tristar`, `Joy`, `Chicken`, and `Chattpata` alone do not count.
