# PetDB provenance and licence audit

Audit date: 2026-09-22

## Decision

The PetDB-derived subset is traceable. EarthChem's published Terms of Use permit
circulation or publication of materials obtained through its systems when the
applicable citation requirements are met. The repository must not assign a new
uniform licence to the PetDB-derived measurements because the portal export does
not contain record-level licence fields and EarthChem states that its digital
materials may use varying Creative Commons licences. This audit relies on the
published terms and does not claim a separate individual authorization.

## Scope and provenance checks

- Original portal export: `PetDB_MORB_ALL.csv`, downloaded 2026-08-11.
- Original export: 23,045 analytical rows, 7,785 unique sample URLs, and 451
  citation URLs.
- Retained release subset: 4,734 rows belonging to 303 PetDB publication records.
- All 4,734 retained rows are labelled `MORB` in the harmonized compilation.
- Every retained row has a unique PetDB sample URL and an original citation.
- Every retained sample URL and citation pair matches the original portal export
  exactly.
- The 303 publication records each map to exactly one PetDB citation URL.
- Eleven final publication groups contain records matched across GEOROC and
  PetDB; the original database identity remains present at row level.

The machine-readable result is `references/petdb_source_records.csv`. It records
the database-specific publication identifier, final leakage-control group,
original citation, PetDB citation URL, retained sample count, source export, and
download date.

## Applicable EarthChem requirements

EarthChem's Terms of Use, last updated in October 2025, state that materials
obtained through its systems may use varying Creative Commons licences and that
circulation or publication requires adequate citation. For synthesis-database or
portal downloads, EarthChem asks users to cite the portal URL, download date,
and, where possible, query parameters. It also strongly encourages citation of
the original data contributors and recommends the PetDB database paper by
Lehnert et al. (2000).

Official sources:

- EarthChem Terms of Use: https://earthchem.org/legal/terms-of-use
- PetDB portal: https://earthchem.org/petdb
- Lehnert et al. (2000): https://doi.org/10.1029/1999GC000026

## Release conditions

1. Keep the MIT licence limited to original software.
2. Do not describe the PetDB-derived measurements as MIT, CC0, CC BY, or CC
   BY-SA unless EarthChem supplies an applicable record-level licence.
3. Keep `DATA_LICENSE.md`, this audit, and both attribution tables in every
   public release.
4. Cite PetDB, the 2026-08-11 access date, and the Lehnert et al. database paper
   in the manuscript.
5. Supply `references/source_publications.csv` as the secondary bibliography for
   original contributors.
6. State that the retained source export contains whole-rock mafic volcanic
   samples classified as basalt or tholeiite; do not claim these were the exact
   portal query parameters unless the author can confirm them.

## Due-diligence correspondence and residual risk

The author asked EarthChem to clarify redistribution of the processed,
attributed subset. EarthChem acknowledged the request under ticket 20691, but no
substantive response had been received as of 22 September 2026. The published
Terms of Use therefore remain the documented basis for circulation and
publication. A later individualized reply should be retained with the project
records, but it is not represented here as a prerequisite or as permission
already granted.

The remaining risk is that the portal CSV does not expose a uniform
record-level licence for every synthesized record. The release addresses this
by preserving database identity and original citations, applying no new licence
to PetDB-derived measurements, and limiting the MIT licence to original code.

## Manuscript-ready availability wording

### Data Availability

The processed data supporting this study, frozen development and holdout
partitions, sample-level holdout predictions, and source data underlying the
figures are available at
https://doi.org/10.5281/zenodo.22478958, with the associated development
repository at https://github.com/zyxflex-ai/global-basalt-tectonic-classification.
The source records were compiled from GEOROC (https://georoc.eu/) and PetDB
(https://earthchem.org/petdb) between 7 and 11 August 2026. PetDB data were
downloaded on 11 August 2026. Database identifiers, source-publication
citations, and PetDB citation URLs are retained in the repository attribution
tables. GEOROC-derived records are provided under CC BY-SA 4.0. PetDB-derived
records are provided in accordance with EarthChem's published Terms of Use
(https://earthchem.org/legal/terms-of-use), with database attribution, the
download date, and retained source-publication citations. The author asserts no
additional licence over those measurements and does not claim individual
authorization from EarthChem.

### Code Availability

All analysis and figure-generation scripts, pinned software dependencies,
integrity tests, a trained XGBoost model, model metadata, and tested usage
examples are available at
https://doi.org/10.5281/zenodo.22478958, with ongoing development at
https://github.com/zyxflex-ai/global-basalt-tectonic-classification. The
author's original code is released under the MIT License. The repository README,
quick-start tutorial, user guide, and reproducibility guide document the inputs,
outputs, execution order, expected results, and computational requirements.
