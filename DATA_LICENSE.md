# Data and figure licensing

The MIT licence in `LICENSE` applies only to original software in this
repository. It does not replace the terms attached to third-party geochemical
records.

## GEOROC-derived records

GEOROC data are provided under the Creative Commons Attribution-ShareAlike 4.0
International licence (CC BY-SA 4.0). GEOROC requires attribution to both
GEOROC and the original data sources, and redistributed adaptations must use
the same licence. The GEOROC-derived portions of the processed CSV files and
derived tabular source data are therefore distributed under CC BY-SA 4.0.

- Database: https://georoc.eu/
- Licence: https://creativecommons.org/licenses/by-sa/4.0/
- Citation guidance: https://georoc.eu/georoc/cite.asp

## PetDB/EarthChem-derived records

EarthChem states that materials obtained through its systems may use varying
Creative Commons licences and that circulation or publication requires adequate
citation. For synthesis-database or portal downloads, EarthChem requests the
portal URL, download date, and, where possible, query parameters; it also
strongly encourages citation of the original contributors.

The PetDB portal export used here was downloaded on 11 August 2026. The retained
subset contains 4,734 records from 303 PetDB publication records. The source CSV
does not provide a record-level licence field, so this repository does not apply
a new uniform licence to the PetDB-derived measurements. Those values remain
subject to the applicable EarthChem and source-record terms. The original
software licence in `LICENSE` does not extend to them.

- PetDB: https://earthchem.org/petdb
- EarthChem terms: https://earthchem.org/legal/terms-of-use
- Provenance and licence audit: `docs/PETDB_LICENSE_AUDIT.md`
- PetDB publication links: `references/petdb_source_records.csv`

## Attribution file

`references/source_publications.csv` contains the database, source citation,
DOI when available, verified publication-group identifier, and number of
retained samples. It is part of the required attribution trail and should be
kept with redistributed data. `references/petdb_source_records.csv` additionally
maps every retained PetDB publication record to its PetDB citation URL and
download provenance.

## Figures

Figures and figure-source tables derived from the combined compilation are
shared under CC BY-SA 4.0 to preserve compatibility with the GEOROC
share-alike requirement. Third-party names and citations remain the property of
their respective creators.
