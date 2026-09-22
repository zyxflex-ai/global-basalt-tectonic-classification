# Journal of Earth Science repository readiness

Audit date: 2026-09-22

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
- Fully time-closed temporal validation, direct Stable-50 missingness comparison,
  label-boundary sensitivity, recoverable label mapping, and
  database-by-class-by-time support tables added for the JES revision.
- JES v9 analysis revision and subsequent integrity updates are on `main`.

## Required before submission

- Create a versioned GitHub release from the v9 revision.
- Archive that release in Zenodo and cite its new version DOI in the final
  manuscript if the archive receives a new DOI; do not overwrite the existing
  immutable v1.0.0 DOI.
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

Release `v1.0.0` was archived on 2026-09-06. Manuscript citations should use the
version DOI https://doi.org/10.5281/zenodo.22478958 so that the cited files remain
fixed; the concept DOI https://doi.org/10.5281/zenodo.22478957 may be used when a
link to the newest available release is intended.
