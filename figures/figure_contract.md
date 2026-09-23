# Journal of Earth Science figure contract

## Manuscript-level conclusion

Publication-disjoint and time-closed validation reveal substantial limits to the apparent generalization of global basalt tectonic classifiers, while source coverage and class boundaries explain important residual variation.

## Evidence architecture

| Figure | Unique claim | Archetype | Principal source |
|---|---|---|---|
| 1 | Publication identity verification precedes leakage-controlled development/test separation. | Workflow schematic | Frozen split and publication audit counts |
| 2 | Sample-random validation inflates performance because publications overlap across folds. | Four-panel validation comparison | `05_results/validation_bias/random_vs_grouped_cv_*_v4.csv` |
| 3 | Frozen-holdout errors concentrate at specific tectonic boundaries. | Row-normalized heat map | `05_results/final_test/final_independent_test_predictions_v4.csv` |
| 4 | Feature importance is strongly class-specific. | Eight-panel ranked bars | `05_results/shap/shap_importance_independent_test_v4.csv` |
| 5 | Transfer performance varies substantially among unseen publications and is not explained by publication size. | Box-and-scatter composite | `05_results/publication_heterogeneity/*.csv` |
| 6 | Time-closed performance remains useful but class- and source-dependent. | Temporal and source-coverage composite | `05_results/figure_source_data/figure6_*_v9.csv` |

## Output contract

- Target: Journal of Earth Science research article.
- Backend: Python/Matplotlib for drawing and export.
- Repository formats: PNG preview, editable SVG, and PDF.
- Submission format: 600-dpi LZW-compressed TIFF generated from the same figure objects.
- Main displays: six figures and one table.
- Integrity: all quantitative panels trace to repository tables, predictions, or fixed split artefacts.
- Interpretation boundary: SHAP values describe model attribution, not geochemical causation; the PetDB ablation is a source-coverage stress test, not a causal database comparison.
