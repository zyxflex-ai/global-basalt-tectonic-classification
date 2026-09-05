# Data dictionary

The machine-readable column dictionary is stored in `data_dictionary.csv` in
this directory. The principal fields are summarized below.

| Field or field family | Meaning | Unit / coding |
|---|---|---|
| `UNIQUE_ID`, `GROUP_ID` | Stable record and sample-group identifiers | text |
| `DATABASE` | Source synthesis database | `GEOROC` or `PetDB` |
| `CITATION`, `PAPER_DOI` | Original publication attribution | text / DOI |
| `PAPER_ID` | Verified source-publication identity | text |
| `CV_GROUP_V4` | Leakage-control grouping unit | text |
| `INNER_FOLD` | Frozen grouped development fold | integer 0–4 |
| `LABEL` | Harmonized tectonic class | CAB, IAB, IOAB, BABB, MORB, OIB, OPB, CFB |
| `SIO2(WT%)` ... `FE_TOTAL(WT%)` | Major-element concentrations | weight percent |
| `SC(PPM)` ... `U(PPM)` | Trace-element concentrations | parts per million |
| `N_STABLE50` | Number of measured values among the 19 Stable-50 predictors | integer 0–19 |
| `PAPER_ID_STATUS`, `CV_GROUP_V4_BASIS` | Publication-identity audit fields | categorical text |

Blank numeric cells are missing measurements. No zero or sentinel value is used
to represent missingness in the processed analysis files.
