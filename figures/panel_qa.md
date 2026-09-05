# Rendered figure QA

All panels were inspected from the final Python-rendered PNG previews at the declared double-column width. The same Python backend generated every SVG, PDF, TIFF and PNG. Static source preflight returned 20 PASS, 0 WARN and 0 FAIL. PDF text audits passed for all nine figures; the minimum detected text size ranged from 6.2 to 7.8 pt, above the 5 pt floor.

| Figure/panel | Unique claim or role | Center/summary | Spread/interval | Unit | Labels/collision | Pass |
|---|---|---|---|---|---|---|
| 1 | Publication-aware, leakage-controlled workflow | Counts in each node | Not applicable | Samples and publication groups | No clipping; arrows clear of text | Yes |
| 2a | Split-dependent OOF performance | Median | Interquartile range | Ten repeated partitions per scheme | Legend and axes clear | Yes |
| 2b | Bootstrap score inflation | Median | 95% percentile interval | 2,000 publication-cluster resamples per metric | Value labels clear of intervals | Yes |
| 2c | Class-specific F1 inflation | Median | Interquartile range | Ten repeated partitions | No label collisions | Yes |
| 2d | Direct source-overlap audit | Median | Raw fold distribution | 50 folds per scheme | Percent labels clear | Yes |
| 3 | Frozen-holdout confusion structure | Row percentage and count | Not applicable | 10,086 samples | Cell text remains legible | Yes |
| 4 | Global attribution ranking | Mean absolute SHAP | Not applicable | Holdout samples × class outputs | Feature labels and bars clear | Yes |
| 5a–h | Class-specific attribution ranking | Mean absolute SHAP | Not applicable | 10,086 samples per class output | Independent axes declared; no clipping | Yes |
| 6a–d | Signed attribution distributions, arc classes | All per-sample SHAP values | Raw distribution | 10,086 samples per feature row | Missing values grey; colour bar clear | Yes |
| 7a–d | Signed attribution distributions, non-arc classes | All per-sample SHAP values | Raw distribution | 10,086 samples per feature row | Missing values grey; colour bar clear | Yes |
| 8a–h | Nonlinear selected-feature responses | Median in 15 quantile bins | 25th–75th percentile | Measured holdout samples | Zero baselines, labels and bands clear | Yes |
| 9a–b | Classical low-dimensional comparison | Raw sample positions | Not applicable | 7,838 complete positive measurements | OPB contrast strengthened; field labels readable | Yes |

## Export checks

- SVG text remains editable (`svg.fonttype = none`).
- PDF text is embedded as TrueType text (`pdf.fonttype = 42`).
- Dense scatter layers are rasterized within vector containers; axes, boundaries and text remain vector.
- TIFF files are 600 dpi; PNG files are 300 dpi previews.
- No samples were downsampled for display.
- Figure 8 uses binned trend curves and interquartile shading only; no raw scatter layer is claimed in its legend.
- Figure 9 boundaries are explicitly limited to qualitative visual comparison.

