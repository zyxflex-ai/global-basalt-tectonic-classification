# English manuscript v4 revision audit

## One-sentence argument

In global eight-class basalt classification, publication-grouped validation exposes source-overlap bias and provides a more defensible transfer estimate, while frozen-model SHAP attribution remains descriptive rather than causal.

## Terminology ledger

| Canonical term | Decision |
|---|---|
| publication-grouped cross-validation | Primary validation design |
| sample-random cross-validation | Controlled biased comparator |
| frozen publication-group holdout set | Final evaluation set |
| macro-F1 | Canonical class-balanced F1 term in prose |
| balanced accuracy | Lowercase in prose; display capitalization retained in tables |
| publication-cluster bootstrap | Resampling unit and uncertainty method |
| XGBoost, RF, RBF-SVM | Canonical model names |
| SHAP attribution | Model-response interpretation; not causal explanation |
| total Fe | Manuscript display term for the total-iron predictor |
| raw class margin | SHAP output scale |

## Evidence allocation

| Material | Classification | Destination |
|---|---|---|
| Split-unit comparison and measured publication overlap | Core discovery | Results and Discussion opening |
| Frozen-holdout class-balanced performance | Necessary support | Results and Conclusions |
| Calibration diagnostics | Qualification | Brief Discussion boundary |
| Class-specific confusion | Core failure boundary | Results and Discussion |
| SHAP ranking, direction, and dependence | Necessary interpretation | Results and bounded Discussion |
| Missingness-only diagnostic | Qualification | Discussion limitation |
| Sensitivity analyses | Robustness | Results; not repeated numerically in Discussion |
| Ti-Zr-Y projection | Qualitative comparator | Results and interpretive boundary |

## Changes

- Added a four-part Discussion organized by validation meaning, class boundaries, SHAP interpretation, and external-validity limits.
- Added a bounded two-paragraph Conclusions section led by the source-overlap result.
- Corrected the Figure 8 Results paragraph so it no longer claims relationships absent from the final eight-panel figure.
- Replaced all nine caption paragraphs with the synchronized English legends from the final figure package.
- Added ten geological references cited by the Discussion and retained the existing cited references.
- Canonicalized macro-F1 and total Fe in prose (32 edited paragraphs).

## Word-count record

- Original v3 paragraph text: 6,695 words
- Added Discussion: 901 words
- Added Conclusions: 167 words
- Complete v4 paragraph text: 8,323 words
