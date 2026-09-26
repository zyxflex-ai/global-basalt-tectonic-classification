# Chemical Geology repository readiness

Audit date: 2026-09-26

## Completed

- English README with installation, basic use, contact, and repository scope.
- MIT licence for original code and a separate third-party data notice.
- Pinned `requirements.txt` and `environment.yml`.
- Data dictionary, provenance table, reproducibility guide, quick-start tutorial,
  user guide, and figure-to-source mapping.
- Frozen train/holdout files, grouped folds, source tables, sample-level holdout
  predictions, signed SHAP cache, final figures, and loadable XGBoost model.
- English code comments and no ZIP, RAR, or 7z archive used as the repository.
- Automated checks for counts, class balance, publication-group isolation,
  attribution cardinality, headline metrics, and model metadata.
- Exact provenance audit for all 4,734 retained PetDB rows and 303 PetDB
  publication records.
- Public GitHub release `v1.0.0`, archived in Zenodo under the version DOI
  https://doi.org/10.5281/zenodo.22478958.
- Public GitHub release `v1.1.0`, archived in Zenodo under the version DOI
  https://doi.org/10.5281/zenodo.22898634.
- Public GitHub release `v1.1.1`, archived in Zenodo under the version DOI
  https://doi.org/10.5281/zenodo.22920294.
- Public GitHub release `v1.1.2`, archived in Zenodo under the version DOI
  https://doi.org/10.5281/zenodo.22967141.
- Fully time-closed temporal validation, direct Stable-50 missingness comparison,
  label-boundary sensitivity, recoverable label mapping, and
  database-by-class-by-time support tables added for the JES revision.
- JES v9 analysis revision and subsequent integrity updates are on `main`.
- Chemical Geology reviewer-directed sensitivity scripts 93-94 and their six
  machine-readable result tables are included in release `v1.1.2`.

## Required before submission

- Preserve the PetDB licence boundary and attribution files when the repository
  becomes public; do not apply a new uniform licence to PetDB-derived values.
- Ensure the manuscript Data Availability and Code Availability text matches the
  released files exactly.

## Recommended before or at submission

- Retain EarthChem ticket 20691 and any later substantive reply with the project
  records. Under the published Terms of Use, individualized confirmation is not
  treated as a release prerequisite; do not state that EarthChem granted
  bespoke permission.
- Test the public GitHub URL and Zenodo version DOI while signed out of the
  author's account.

## Release rule

Release `v1.0.0` was archived on 2026-09-06 and remains immutable. Release
`v1.1.0` was archived on 2026-09-22, and `v1.1.1` was archived on 2026-09-23.
Release `v1.1.2` adds the Chemical Geology sensitivity analyses. The manuscript
should cite the immutable version DOI https://doi.org/10.5281/zenodo.22967141;
the concept DOI https://doi.org/10.5281/zenodo.22478957 links to the newest
available version.
