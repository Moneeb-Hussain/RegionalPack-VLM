# Product list for the shelf check

Status: **collected and ready for photography.** The 20 packs are in hand. Formal photographs have not been taken. The hypotheses are unchanged.

10 categories, 20 packaged products, 3 photographs each. Nine pairs are the primary international-versus-Pakistan comparison. The apple-drink pair is exploratory and is left out of that comparison.

This evaluation is descriptive. A difference between the two groups would be an association at n = 20 products. It would not show that geography caused the difference. Brand identification (prompt 1) and front-of-pack transcription (prompt 2) stay separate scores. A missed brand is not an OCR error, and text hidden by the occluding object is not a failed transcription.

## Membership rule

The unit is the **packaged product**, not the parent company.

| Level | What it is | How it is used |
|---|---|---|
| Parent | The firm that owns the brand now | Recorded so ownership is not mistaken for origin |
| Product line | The brand name a shopper uses | This is `brand` |
| Packaged product | One variant and one package type | This is what gets photographed |

A product line is in exactly one group. Test them in this order and stop:

| Order | Rule id | Group | Test |
|---|---|---|---|
| 1 | `pakistan_market_line` | `pakistan_origin` | The product line was introduced as a brand for the Pakistan market. |
| 2 | `sold_under_same_name_outside_pakistan` | `international` | Rule 1 does not apply, and the same brand name is sold outside Pakistan. |

Export sales do not also put a Pakistan-market line in the international group. A later owner, a local bottler, or a factory in Pakistan does not create a second membership. A line that matches neither rule is not locked.

`group_rule` in [product_manifest.csv](product_manifest.csv) stores the one rule that applied.

## Corrections

**Chocolate, replacing milk.** The Almarai and Olper's milk rows are withdrawn before any benchmark photograph. The reason is cost and availability. Phase 0 responses and model performance did not decide the change. The replacement pair is plain milk chocolate: Cadbury Dairy Milk and CandyLand YOU. A wafer, a filled chocolate, and a chocolate biscuit are different products. Both chocolate rows stay pending a physical shelf check. Dairy Omung was in Phase 0 and is not added.

**Biscuits.** Oreo is one global sandwich biscuit. Bisconni is the parent range. The paired pack is **Rite dark sandwich biscuits with vanilla cream**. The Oreo pack is the original dark sandwich biscuit with vanilla cream, so the biscuit and the filling match. Golden Oreo and a chocolate-cream filling do not. Peek Freans Marie was in Phase 0 and is a different biscuit type.

Sources: [Bisconni](https://ismailindustries.com.pk/companies/bisconni), [Aurora, Eclipse launch](https://aurora.dawn.com/news/1145335), 2019 CLD 562 (Ismail Industries v Mondelēz) on Rite and Oreo.

**Soap.** Lux is a Unilever beauty soap. Sufi laundry soap is a laundry bar. Those are not the same subtype. The local bar is **Capri**, launched by ZIL in 1969 as a Pakistan beauty soap. A holding company buying ZIL later does not move Capri into the international group.

Sources: [Lux Pakistan](https://www.lux.com/pk/home.html), [Sufi laundry](https://sufigroup.biz/laundry/), [ZIL annual report 2019](https://www.zil.com.pk/wp-content/uploads/2021/04/ZIL-Annual-Report-2019.pdf), [Capri](https://www.capri.com.pk/).

**Cola Next** is Mezan Beverages' Pakistan cola, launched in 2016. It is not affiliated with Coca-Cola. Sources: [Cola Next about](https://colanext.com/about), [Soch Fact Check](https://www.sochfactcheck.com/cola-next-is-not-affiliated-with-coca-cola/).

## Proposed pairs

| Category | International pack | Pakistan-origin pack | Ids |
|---|---|---|---|
| cola | Coca-Cola original, 1.5 L PET | Cola Next original, 1.5 L PET | `cola_intl`, `cola_local` |
| biscuit | Oreo original dark sandwich biscuits with vanilla cream, family pack | Rite dark sandwich biscuits with vanilla cream, family pack | `biscuit_intl`, `biscuit_local` |
| tea | Lipton Yellow Label loose leaf, carton | Tapal Danedar loose leaf, carton | `tea_intl`, `tea_local` |
| chocolate | Cadbury Dairy Milk plain bar, preferably 16.5 g | CandyLand YOU Creamiest plain bar, preferably 21 g | `chocolate_intl`, `chocolate_local` |
| chips | Lay's Classic potato crisps, bag | Super Crisp BBQ potato chips, bag | `chips_intl`, `chips_local` |
| detergent | Surf Excel washing powder, pouch | Bonus Tristar detergent powder, pouch | `detergent_intl`, `detergent_local` |
| toothpaste | Colgate Fresh Gel, gel tube | Medicam UltraFresh Gel 3-in-1, gel tube | `toothpaste_intl`, `toothpaste_local` |
| apple, exploratory | Fruita Vitals Apple, pack in hand | Fruitien Joy apple fruit drink, pack in hand | `apple_fruita`, `apple_fruitien` |
| noodles | Knorr Chicken instant noodles, one pack | Shan Shoop Chatpatta instant noodles, one pack | `noodles_intl`, `noodles_local` |
| soap | Lux beauty-soap bar | Capri beauty-soap bar | `soap_intl`, `soap_local` |

## Limitations on the collected packs

These differences are recorded. They are not edited into a match.

| Pair | What is the same | What is not the same |
|---|---|---|
| Chips | Potato-chip bag | Lay's Classic versus Super Crisp BBQ |
| Noodles | Instant noodles | Knorr Chicken versus Shan Shoop Chatpatta |
| Toothpaste | Gel toothpaste | Colgate Fresh Gel versus Medicam UltraFresh Gel 3-in-1 |

## Apple drinks, exploratory

Nestlé created Fruita Vitals as a juice sub-brand for the Pakistan market. A 2011 WARC case describes that Pakistan repositioning, and the Express Tribune reported the 2011 Pakistan relaunch. A sale under the Fruita Vitals name outside Pakistan is not supported here, so the product is not labeled international. Fruitien is Citropak's Pakistan brand. The pair is photographed with the other 18 products and excluded from the primary comparison.

Sources: [WARC, 22 May 2011](https://www.warc.com/en/article/nestl%C3%A9-fruita-vitals%3A-fruit-fuel-842c611264f845d99ed79c222975bcb5), [Express Tribune, 30 Apr 2011](https://tribune.com.pk/story/159468/juices-and-nectars-nestle-re-launches-fruita-vitals), [Fruitien products](https://fruitienofficial.com/our-products/), [Citropak](https://citropakofficial.com/).

The structured copy is [product_manifest.csv](product_manifest.csv). The store sheet is [availability_checklist.md](availability_checklist.md).

## Accepted brand aliases

[configs/brand_aliases.yaml](../configs/brand_aliases.yaml) is the full list of strings that count as the brand on Prompt 1. It is fixed before inference. A parent company is not an alias: Bisconni is not Rite, Mondelēz is not Cadbury, and YOU alone is not CandyLand. Brand identification stays separate from the OCR scores.

## Phase 0 products

The six tested products are listed in [phase0_seen_products.csv](phase0_seen_products.csv). The filenames are the ones in `RegionalPack-Phase0-Labeled-Photos`. Those files have no camera filename in their metadata. The package face in each file supplied the match. Phase 0 model output was not used.

| Tested product | Filename | On this list? |
|---|---|---|
| Lipton Yellow Label | `latin_01.jpg` | Yes, `tea_intl`. Kept as the specification's tea example. |
| National Garam Masala | `urdu_01.jpg` | No |
| Dairy Omung Milk | `urdu_02.jpg` | No. The pack in that file is Low Fat Milk, 220 ml. |
| Shan Chicken Tikka | `urdu_03.jpg` | No |
| Peek Freans Marie | `latin_02.jpg` | No |
| Shangrila Garlic Chilli Sauce | `latin_03.jpg` | No |

`used_to_choose_manifest` is `no` on every row. Phase 0 photographs stay outside `data/raw/`. A new three-angle set is required even for Lipton.

## Ground truth

[annotations.csv](annotations.csv) is header only. After photography, each row records the front-of-pack text **visible in that photograph**. Words covered in `occluded.jpg` are written in `occlusion_note` and left out of `text_true_latin` and `text_true_urdu` for that row. They are not copied from the straight photograph, and they are not a transcription failure. Ingredient lists, nutrition tables, barcodes, and regulatory fine print stay out.

`brand_true` is still the brand on the package. It does not change when the occluder covers the wordmark. Brand identity and visible text are different fields.
