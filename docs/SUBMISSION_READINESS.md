# Computers & Geosciences repository readiness

Audit date: 2026-09-05

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

## Required before submission

- Make the GitHub repository public; Computers & Geosciences requires public
  repository access at submission.
- Preserve the PetDB licence boundary and attribution files when the repository
  becomes public; do not apply a new uniform licence to PetDB-derived values.
- Ensure the manuscript Data Availability and Code Availability text matches the
  released files exactly.

## Recommended before or at submission

- Create a tagged GitHub release after the manuscript analysis is frozen.
- Archive that release in a durable repository such as Zenodo and replace any
  pending DOI fields in the manuscript and repository metadata with the assigned
  identifier.
- Obtain written confirmation from EarthChem that the processed, attributed
  PetDB subset may be redistributed without applying a new licence to those
  values; retain the reply with the project records.
- Test the public URL and, if created, the DOI while signed out of the author's
  account.

## Release rule

Do not create a `v1.0.0` release or claim a release date in `CITATION.cff` until
the final public artifact exists. Add the actual version, release date, and DOI
only after they have been assigned.
