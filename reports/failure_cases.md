# Failure cases and contrasting responses

Excerpts are copied from `benchmark_predictions.jsonl`. The selection rule is: every abstention, every punctuation-blocked assertion, the gum/cleaning contradiction, the Biscotti spelling, the repeated Knorr line, one soap and one toothpaste category miss, and one clear international hit.

## `chocolate_local/straight`

Frozen brand match: False. Outcome: abstention. Adjudication: abstention_not_a_hit. Category match: True.
Flags: alias_quoted_inside_abstention|category_also_says_dairy.

Identification:

```text
The product shown in the image is a chocolate bar. The packaging features the text "you MILK CHOCOLATE" in a stylized font. However, the specific brand name cannot be identified with confidence from the image.
```

Category:

```text
This is a chocolate bar, which falls under the category of dairy products.
```

Transcription:

```text
you
MILK
CHOCOLATE
```

Latin CER 0.00. Urdu CER not scored. Repeat count 1.

## `chips_local/straight`

Frozen brand match: False. Outcome: abstention. Adjudication: abstention_not_a_hit. Category match: True.
Flags: alias_quoted_inside_abstention.

Identification:

```text
The product shown in the image is a bag of "Super Crisp" flavored potato chips with a "Bar.B.Q" flavor. The specific brand name is not clearly identifiable from the image.
```

Category:

```text
This is a snack.
```

Transcription:

```text
SUPER CRISP
FLAVOURED POTATO CHIPS
BAR.B.Q
FLAVOUR
PER 30 gm
ENERGY
153.2 kcal
60
S.60
```

Latin CER 0.72. Urdu CER not scored. Repeat count 1.

## `biscuit_local/tilt`

Frozen brand match: True. Outcome: asserted_alias. Adjudication: none. Category match: True.
Flags: wrong_parent_spelling_biscotti.

Identification:

```text
The product shown in the image is a biscuit bar from the brand Biscotti, specifically the "Rite" flavor. The packaging indicates it is a "Rich Chocolate Biscuits with Vanilla Cream" and features an image of a chocolate biscuit with cream. The brand name is Biscotti, and the product name is Rite.
```

Category:

```text
This is a snack.
```

Transcription:

```text
BISCONNI
Rite
RICH CHOCOLATE BISCUITS WITH VANILLA CREAM
```

Latin CER 0.00. Urdu CER not scored. Repeat count 1.

## `biscuit_local/straight`

Frozen brand match: False. Outcome: punctuation_blocked_assertion. Adjudication: pending_human. Category match: True.
Flags: alias_present_but_comma_or_quotes_block_frozen_match|parent_named_as_the_brand.

Identification:

```text
The product shown in the image is Bisconni Rite, which is a type of chocolate biscuit with vanilla cream. The specific brand name is Bisconni.
```

Category:

```text
This is a snack.
```

Transcription:

```text
BISCONNI
Rite
RICH CHOCOLATE BISCUITS WITH
VANILLA CREAM
```

Latin CER 0.00. Urdu CER not scored. Repeat count 1.

## `detergent_local/tilt`

Frozen brand match: True. Outcome: asserted_alias. Adjudication: none. Category match: True.
Flags: identification_calls_it_gum.

Identification:

```text
The product shown in the image is a packet of Bonus Tristar, a brand of sugar-free gum. The packaging indicates it is a "NEW" product with a "50" on the top left, which likely refers to the number of pieces in the pack. The text on the package is in Arabic, which suggests the product is likely from a country where Arabic is spoken. The brand name "Bonus" is prominently displayed in large white letters with a blue outline, and "Tristar" is written below it in red. The packaging also features a ra…
```

Category:

```text
This is a cleaning product.
```

Transcription:

```text
50
زيادة مهلك
زيادة صافاتي!
BONUS
TRISTAR
NEW
```

Latin CER 2.46. Urdu CER 1.5. Repeat count 1.

## `detergent_local/straight`

Frozen brand match: False. Outcome: punctuation_blocked_assertion. Adjudication: pending_human. Category match: True.
Flags: alias_present_but_comma_or_quotes_block_frozen_match.

Identification:

```text
The product shown in the image is a laundry detergent. The specific brand name is "Bonus Tristar".
```

Category:

```text
cleaning product
```

Transcription:

```text
زيادة جهاك، زيادة صفاتي
R5
50
NEW
BONUS
TRISTAR
إلى شتاد خلائق
```

Latin CER 3.77. Urdu CER 2.1363636363636362. Repeat count 1.

## `noodles_local/straight`

Frozen brand match: False. Outcome: punctuation_blocked_assertion. Adjudication: pending_human. Category match: True.
Flags: alias_present_but_comma_or_quotes_block_frozen_match.

Identification:

```text
The product shown in the image is a packet of instant noodles. The specific brand name is "Shoop".
```

Category:

```text
This is a snack, specifically instant noodles.
```

Transcription:

```text
shoop
RS 45
CHATPATA
Chattpata
FLAVOUR
INSTANT
Noodles
Best
Suggestion
awesome
```

Latin CER 4.20. Urdu CER 12.833333333333334. Repeat count 1.

## `noodles_intl/tilt`

Frozen brand match: True. Outcome: asserted_alias. Adjudication: none. Category match: True.
Flags: repeated_transcription_line.

Identification:

```text
The product shown in the image is Knorr Chicken instant noodles.
```

Category:

```text
This is a snack, specifically instant noodles.
```

Transcription:

```text
Knorr
SINCE 1838
chicken
RS-50
INSTANT
NO added preservatives
NO added preservatives
NO added preservatives
NO added preservatives
NO added preservatives
NO added preservatives
NO added preservatives
NO added preservatives
NO added preservatives
NO added preservatives
NO added preservatives
NO added preservatives
NO added preservatives
NO added preservatives
NO added preservatives
NO added preserv…
```

Latin CER 221.00. Urdu CER not scored. Repeat count 46.

## `soap_intl/straight`

Frozen brand match: True. Outcome: asserted_alias. Adjudication: none. Category match: False.
Flags: category_uses_prompt_example_cleaning_product.

Identification:

```text
The product shown in the image is a bar of soap. The specific brand is LUX.
```

Category:

```text
This is a cleaning product.
```

Transcription:

```text
LUX
Vibrant Glow
TANGERINE
VITAMIN C+E
```

Latin CER 0.00. Urdu CER not scored. Repeat count 1.

## `toothpaste_intl/straight`

Frozen brand match: True. Outcome: asserted_alias. Adjudication: none. Category match: False.
Flags: category_uses_prompt_example_cleaning_product.

Identification:

```text
The product shown in the image is Colgate Fresh Gel.
```

Category:

```text
This is a cleaning product.
```

Transcription:

```text
Colgate® FRESH GEL
✓ Strong Teeth ✓ Fresh Breath ✓ Cavity Protection
Vegan
Recyclable Tube®
```

Latin CER 0.49. Urdu CER not scored. Repeat count 1.

## `cola_intl/straight`

Frozen brand match: True. Outcome: asserted_alias. Adjudication: none. Category match: True.
Flags: none.

Identification:

```text
The product shown in the image is Coca-Cola.
```

Category:

```text
beverage
```

Transcription:

```text
Coca-Cola
ORIGINAL TASTE
```

Latin CER 0.00. Urdu CER not scored. Repeat count 1.
