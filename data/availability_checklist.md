# Availability checklist

Print this page. Tick it in the store. Do not photograph the benchmark set on this visit.

| | |
|---|---|
| Store | |
| Date | |
| Checker | |

Every row below is copied from `data/product_manifest.csv`.
These packs are photographed. Visible text is not filled in yet.

## Limitations and the exploratory pair

- Biscuits: Oreo chocolate cream and Rite chocolate biscuits with vanilla cream are both sandwich biscuits. The fillings are not identical. Bisconni is the parent mark, not an alias for Rite.
- Chips: Lay's Classic and Super Crisp BBQ are both potato-chip bags. The flavors are not identical.
- Noodles: Knorr Chicken and Shoop Chattpata are both instant noodles. The flavors are not identical. Shoop is the printed brand. Shan alone is not a product-brand alias.
- Toothpaste: Colgate Fresh Gel and Medicam UltraFresh Gel 3-in-1 are both gels. The gel claims are not identical.
- Soap: Lux Vibrant Glow Tangerine and Capri Soothing Glow are both beauty-soap bars. The fragrances are not identical.
- Apple drinks: `apple_fruita` and `apple_fruitien` are photographed and excluded from the primary international-versus-Pakistan comparison. Fruita Vitals is a Nestlé Pakistan sub-brand, and a sale under that name outside Pakistan is not supported.

A category may be photographed in its own place. Both products in a pair use that same place, the same light, and the same background.

## Products

| Shot | Product ID | Group | Role | Exact variant | Package type | Availability | Notes |
|---|---|---|---|---|---|---|---|
| ☐ | cola_intl | international | primary | Coca-Cola: original cola | 1.5 L PET bottle | in_hand | Do not substitute Coke Zero or Sprite. |
| ☐ | cola_local | local | primary | Cola Next: original cola | 1.5 L PET bottle | in_hand | https://colanext.com/about and Soch Fact Check 2024. |
| ☐ | biscuit_intl | international | primary | Oreo: chocolate cream sandwich biscuits | 4-pack in hand | in_hand | Filling mismatch: Oreo chocolate cream versus Rite chocolate biscuits with vanilla cream. Both are sandwich biscuits. The fillings are not identical. |
| ☐ | biscuit_local | local | primary | Rite: rich chocolate biscuits with vanilla cream | single pack in hand | in_hand | Filling mismatch: Oreo chocolate cream versus Rite chocolate biscuits with vanilla cream. Both are sandwich biscuits. The fillings are not identical. |
| ☐ | tea_intl | international | primary | Lipton: Yellow Label, loose black tea | leaf carton | in_hand | Overlap is latin_01.jpg. Not a tea bag. |
| ☐ | tea_local | local | primary | Tapal: Danedar, loose black tea | leaf carton | in_hand | Same subtype as Lipton Yellow Label: loose black tea in a carton. |
| ☐ | chocolate_intl | international | primary | Cadbury: Dairy Milk, plain milk chocolate | bar in hand | in_hand | Photographed Cadbury Dairy Milk bar. Wrapper prints 1 serving = 27 g. The 16.5 g target was not this bar. |
| ☐ | chocolate_local | local | primary | YOU: milk chocolate bar | bar in hand | in_hand | Printed wordmark is YOU. CandyLand is the parent range and is not a product-brand alias by itself. YOU alone is a correct brand identification. CandyLand YOU is accepted. |
| ☐ | chips_intl | international | primary | Lay's: Classic potato crisps | bag | in_hand | Flavor mismatch: Lay's Classic versus Super Crisp BBQ. Both are potato-chip bags. The flavors are not identical. |
| ☐ | chips_local | local | primary | Super Crisp: BAR.B.Q flavoured potato chips | bag | in_hand | Flavor mismatch: Lay's Classic versus Super Crisp BBQ. Both are potato-chip bags. The flavors are not identical. |
| ☐ | detergent_intl | international | primary | Surf Excel: washing powder, standard powder | pouch | in_hand | Washing powder. Not a laundry bar. |
| ☐ | detergent_local | local | primary | Bonus: Tristar, detergent powder | pouch | in_hand | https://colgate.com.pk/product/bonus-tristar/ |
| ☐ | toothpaste_intl | international | primary | Colgate: Fresh Gel toothpaste | gel carton in hand | in_hand | Variant mismatch: Colgate Fresh Gel versus Medicam UltraFresh Gel 3-in-1. Both are gel toothpaste. The gel claims are not identical. |
| ☐ | toothpaste_local | local | primary | Medicam: UltraFresh Gel 3-in-1 red gel | 75 g carton in hand | in_hand | Variant mismatch: Colgate Fresh Gel versus Medicam UltraFresh Gel 3-in-1. Both are gel toothpaste. The gel claims are not identical. |
| ☐ | apple_fruita | exploratory | exploratory | Fruita Vitals: apple nectar | carton in hand | in_hand | Exploratory pair. Fruita Vitals is a Nestlé Pakistan juice sub-brand. A sale under that name outside Pakistan is not supported, so the pair is outside the primary comparison. Fruitien is a Citropak brand launched in Pakistan. The photographed packs are an apple nectar and an apple fruit drink. Those drink types are not identical. |
| ☐ | apple_fruitien | exploratory | exploratory | Fruitien: Joy apple fruit drink | carton in hand | in_hand | Exploratory pair. Fruita Vitals is a Nestlé Pakistan juice sub-brand. A sale under that name outside Pakistan is not supported, so the pair is outside the primary comparison. Fruitien is a Citropak brand launched in Pakistan. The photographed packs are an apple nectar and an apple fruit drink. Those drink types are not identical. |
| ☐ | noodles_intl | international | primary | Knorr: chicken instant noodles | single pack | in_hand | Flavor mismatch: Knorr Chicken versus Shan Shoop Chatpatta. Both are instant noodles. The flavors are not identical. |
| ☐ | noodles_local | local | primary | Shoop: Chattpata instant noodles | single pack | in_hand | Flavor mismatch: Knorr Chicken versus Shan Shoop Chatpatta. Both are instant noodles. The flavors are not identical. |
| ☐ | soap_intl | international | primary | Lux: Vibrant Glow tangerine beauty-soap bar | wrapped bar | in_hand | Fragrance mismatch: Lux Vibrant Glow Tangerine versus Capri Soothing Glow. Both are beauty-soap bars. The fragrances are not identical. |
| ☐ | soap_local | local | primary | Capri: Soothing Glow beauty-soap bar | wrapped bar | in_hand | Fragrance mismatch: Lux Vibrant Glow Tangerine versus Capri Soothing Glow. Both are beauty-soap bars. The fragrances are not identical. |

Photograph only the pack named in the variant column.
A flavor or gel-claim difference stays in the limitation. It is not edited into a match.

## Accepted brand aliases

Fixed in `configs/brand_aliases.yaml` before inference.
Prompt 1 may use these strings for the brand. A parent company is not an alias.

| Product ID | Accepted names |
|---|---|
| cola_intl | Coca-Cola; Coca Cola; Coke |
| cola_local | Cola Next; Next Cola |
| biscuit_intl | Oreo |
| biscuit_local | Rite |
| tea_intl | Lipton; Lipton Yellow Label |
| tea_local | Tapal; Tapal Danedar |
| chocolate_intl | Cadbury; Cadbury Dairy Milk; Dairy Milk |
| chocolate_local | YOU; CandyLand YOU |
| chips_intl | Lay's; Lays |
| chips_local | Super Crisp; SuperCrisp |
| detergent_intl | Surf Excel; Surf |
| detergent_local | Bonus; Bonus Tristar |
| toothpaste_intl | Colgate |
| toothpaste_local | Medicam |
| apple_fruita | Fruita Vitals; Nestlé Fruita Vitals; Nestle Fruita Vitals |
| apple_fruitien | Fruitien; Fruitien Joy |
| noodles_intl | Knorr; Knorr Chicken |
| noodles_local | Shoop; Shoop Chattpata; Shoop Chatpatta; Shan Shoop; Shan Shoop Chattpata; Shan Shoop Chatpatta |
| soap_intl | Lux |
| soap_local | Capri |

Photography and annotation, after this list is locked: `data/photography_guide.md`.
