# Brand adjudication

This file is post hoc. It was written after the frozen score existed. It does not replace that score.

Frozen automatic brand majority, recomputed from image rows: 13/18.
Adjudicated brand majority, recomputed from the image decisions below: 15/18.

The adjudicated total is the number of primary products with at least 2 of 3 photographs marked successful in `results/adjudication_decisions.csv`. It is not typed in by hand.

## Rules applied

| Rule | Decision |
|---|---|
| The response says Bisconni or Biscotti is the brand | Not a Rite product-brand success, including when Rite is called the flavor or the product name |
| The response names Bonus Tristar | Product-brand success in the adjudicated set. A gum description in the same answer is noted and does not cancel the brand decision |
| The response says the brand name is Shoop | Product-brand success in the adjudicated set |
| The response says the specific brand cannot be identified | Failure, even if the name is quoted |

## Reviewed responses

### `biscuit_local/straight`

Frozen success: False. Adjudicated success: False. Status: adjudicated_not_product_brand. Difference: unchanged.

Response:

```text
The product shown in the image is Bisconni Rite, which is a type of chocolate biscuit with vanilla cream. The specific brand name is Bisconni.
```

Rationale: The response assigns the brand role to Bisconni or Biscotti. Rite is not named as the brand. Parent-as-brand is not product-brand identification.

### `biscuit_local/tilt`

Frozen success: True. Adjudicated success: False. Status: adjudicated_not_product_brand. Difference: frozen_hit_adjudicated_miss.

Response:

```text
The product shown in the image is a biscuit bar from the brand Biscotti, specifically the "Rite" flavor. The packaging indicates it is a "Rich Chocolate Biscuits with Vanilla Cream" and features an image of a chocolate biscuit with cream. The brand name is Biscotti, and the product name is Rite.
```

Rationale: The response assigns the brand role to Bisconni or Biscotti. Rite is not named as the brand. It also calls Rite a flavor. It also calls Rite the product name. Parent-as-brand is not product-brand identification.

### `biscuit_local/occluded`

Frozen success: False. Adjudicated success: False. Status: adjudicated_not_product_brand. Difference: unchanged.

Response:

```text
The product shown in the image is a biscuit bar from the brand Bisconni, specifically the "Rite" flavor. The packaging indicates it is "Rich Chocolate Biscuits with Vanilla Cream."
```

Rationale: The response assigns the brand role to Bisconni or Biscotti. Rite is not named as the brand. It also calls Rite a flavor. Parent-as-brand is not product-brand identification.

### `chocolate_local/straight`

Frozen success: False. Adjudicated success: False. Status: confirmed_abstention. Difference: unchanged.

Response:

```text
The product shown in the image is a chocolate bar. The packaging features the text "you MILK CHOCOLATE" in a stylized font. However, the specific brand name cannot be identified with confidence from the image.
```

Rationale: The identification says the specific brand cannot be identified. A name inside quotation marks is packaging text, not a brand identification.

### `chocolate_local/tilt`

Frozen success: False. Adjudicated success: False. Status: confirmed_abstention. Difference: unchanged.

Response:

```text
The product shown in the image is a chocolate bar. The specific brand name is not identifiable from the image.
```

Rationale: The identification says the specific brand cannot be identified. A name inside quotation marks is packaging text, not a brand identification.

### `chocolate_local/occluded`

Frozen success: False. Adjudicated success: False. Status: confirmed_abstention. Difference: unchanged.

Response:

```text
The product shown in the image is a chocolate bar. The packaging features the text "you MILK CHOCOLATE" in gold lettering on a dark green background, with an image of two chocolate pieces on the right side. The brand name is not clearly visible, so I cannot identify the specific brand with confidence.
```

Rationale: The identification says the specific brand cannot be identified. A name inside quotation marks is packaging text, not a brand identification.

### `chips_local/straight`

Frozen success: False. Adjudicated success: False. Status: confirmed_abstention. Difference: unchanged.

Response:

```text
The product shown in the image is a bag of "Super Crisp" flavored potato chips with a "Bar.B.Q" flavor. The specific brand name is not clearly identifiable from the image.
```

Rationale: The identification says the specific brand cannot be identified. A name inside quotation marks is packaging text, not a brand identification.

### `chips_local/tilt`

Frozen success: False. Adjudicated success: False. Status: confirmed_abstention. Difference: unchanged.

Response:

```text
The product shown in the image is a bag of potato chips. The specific brand name is not clearly visible in the image.
```

Rationale: The identification says the specific brand cannot be identified. A name inside quotation marks is packaging text, not a brand identification.

### `chips_local/occluded`

Frozen success: False. Adjudicated success: False. Status: confirmed_abstention. Difference: unchanged.

Response:

```text
The product shown in the image is a bag of potato chips. The specific brand name is not clearly visible in the image.
```

Rationale: The identification says the specific brand cannot be identified. A name inside quotation marks is packaging text, not a brand identification.

### `detergent_local/straight`

Frozen success: False. Adjudicated success: True. Status: adjudicated_product_brand. Difference: frozen_miss_adjudicated_hit.

Response:

```text
The product shown in the image is a laundry detergent. The specific brand name is "Bonus Tristar".
```

Rationale: The response explicitly identifies Bonus Tristar.

### `detergent_local/tilt`

Frozen success: True. Adjudicated success: True. Status: adjudicated_product_brand. Difference: unchanged.

Response:

```text
The product shown in the image is a packet of Bonus Tristar, a brand of sugar-free gum. The packaging indicates it is a "NEW" product with a "50" on the top left, which likely refers to the number of pieces in the pack. The text on the package is in Arabic, which suggests the product is likely from a country where Arabic is spoken. The brand name "Bonus" is prominently displayed in large white letters with a blue outline, and "Tristar" is written below it in red. The packaging also features a rainbow design and small illustrations of flowers. The product is likely a sugar-free gum
```

Rationale: The response explicitly identifies Bonus Tristar. The same answer calls the product sugar-free gum. That is a product-type error. It does not cancel the explicit Bonus Tristar brand identification.

### `detergent_local/occluded`

Frozen success: False. Adjudicated success: True. Status: adjudicated_product_brand. Difference: frozen_miss_adjudicated_hit.

Response:

```text
The product shown in the image is a cleaning powder, specifically a brand called "Bonus Tristar". The packaging is in a yellow color with a red top and features the brand name "Bonus Tristar" in large white letters. The price is listed as Rs. 50, and the packaging also includes Arabic text.
```

Rationale: The response explicitly identifies Bonus Tristar.

### `noodles_local/straight`

Frozen success: False. Adjudicated success: True. Status: adjudicated_product_brand. Difference: frozen_miss_adjudicated_hit.

Response:

```text
The product shown in the image is a packet of instant noodles. The specific brand name is "Shoop".
```

Rationale: The response says the specific brand name is Shoop.

### `noodles_local/tilt`

Frozen success: False. Adjudicated success: True. Status: adjudicated_product_brand. Difference: frozen_miss_adjudicated_hit.

Response:

```text
The product shown in the image is a packet of instant noodles. The specific brand name is "Shoop".
```

Rationale: The response says the specific brand name is Shoop.

### `noodles_local/occluded`

Frozen success: True. Adjudicated success: True. Status: adjudicated_product_brand. Difference: unchanged.

Response:

```text
The product shown in the image is a packet of instant noodles. The specific brand name is Shoop.
```

Rationale: The response says the specific brand name is Shoop.
