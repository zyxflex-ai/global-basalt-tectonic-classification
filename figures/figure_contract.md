# G3 English figure contract

## Manuscript-level conclusion

Publication-level separation is necessary for a credible estimate of generalization in global basalt tectonic-setting classification, and the frozen XGBoost model retains interpretable, class-specific geochemical structure under that evaluation design.

## Evidence architecture

| Figure | Unique claim | Archetype | Source |
|---|---|---|---|
| 1 | Publication identity verification precedes leakage-controlled development/test separation. | Schematic-led workflow | Frozen v4 audit counts |
| 2 | Sample-random validation inflates out-of-fold performance because sources overlap across folds. | Validation envelope; hero evidence | `random_vs_grouped_cv_*_v4.csv` |
| 3 | Frozen-holdout errors are concentrated in specific class boundaries rather than uniformly distributed. | Row-normalized heat map | `final_independent_test_predictions_v4.csv` |
| 4 | A compact set of major- and trace-element variables dominates global attribution magnitude. | Ranked quantitative bars | `shap_importance_independent_test_v4.csv` |
| 5 | Attribution magnitudes differ among the eight tectonic classes. | Repeated class-specific bars | Same SHAP importance table |
| 6 | Arc-related classes show distinct signed attribution distributions. | Attribution distribution grid | Frozen per-sample SHAP cache |
| 7 | Non-arc and intraplate classes show distinct signed attribution distributions. | Attribution distribution grid | Frozen per-sample SHAP cache |
| 8 | Selected high-ranking features have nonlinear, class-specific response profiles. | Quantile-response grid | `shap_dependence_selected_bins_stable50_v4.csv` |
| 9 | The classical Ti-Zr-Y diagram provides a qualitative low-dimensional comparison but not an accuracy benchmark. | Comparative ternary grid | Frozen Stable-50 holdout data |

## Output contract

- Target: G3 submission, approximately 183 mm double-column width.
- Backend: Python/Matplotlib only for drawing, preview, export, and QA.
- Typography: sans serif, minimum source text size 7 pt; lowercase bold panel labels 9 pt.
- Formats: editable SVG, editable-text PDF, 600-dpi TIFF, and 300-dpi PNG preview.
- Integrity: all plotted observations retained; dense point clouds rasterized only inside vector PDF/SVG containers.
- Color: color-vision-friendly blue/orange/red categorical palette and viridis continuous scale; no rainbow map.
- Boundaries: Pearce-Cann A-D polygons are digitized approximations used only for visual comparison.

