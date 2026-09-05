# Cross-database publication identity verification (v4)

## Summary

- Candidate pairs checked: 14
- Verified as the same publication: 7
- Verified as different publications: 7
- Unresolved pairs: 0
- Decision rule: DOI and full bibliographic metadata take precedence over first-author surname and year; sample-name agreement is supporting evidence only.

## Field-level decisions

| # | GEOROC record | Decision | Evidence |
|---:|---|---|---|
| 1 | 13694, Geist D. J. (2006) | SAME | Both records are *Submarine Fernandina: Magmatism at the leading edge of the Galápagos hot spot*, with the same seven authors and DOI [10.1029/2006GC001290](https://doi.org/10.1029/2006GC001290). |
| 2 | 3166, Moore J. G. (1970) | DIFFERENT | GEOROC is *Submarine basalts from the Revillagigedo Islands region, Mexico*, DOI [10.1016/0025-3227(70)90022-8](https://doi.org/10.1016/0025-3227(70)90022-8); PetDB is *Water content of basalt erupted on the ocean floor*, DOI [10.1007/BF00388949](https://doi.org/10.1007/BF00388949). |
| 3 | 14609, Zhang Yu-Tao (2010) | DIFFERENT | GEOROC is a Tarim Basin paper in *Gondwana Research*, DOI [10.1016/j.gr.2010.03.006](https://doi.org/10.1016/j.gr.2010.03.006); PetDB is an East Pacific Rise paper by Zhang G. et al., DOI [10.1016/j.jvolgeores.2010.03.002](https://doi.org/10.1016/j.jvolgeores.2010.03.002). |
| 4 | 14737, Zhang Chuan-Lin (2010) | DIFFERENT | GEOROC is *Diverse Permian magmatism in the Tarim Block*, DOI [10.1016/j.lithos.2010.08.007](https://doi.org/10.1016/j.lithos.2010.08.007); PetDB is the East Pacific Rise paper, DOI [10.1016/j.jvolgeores.2010.03.002](https://doi.org/10.1016/j.jvolgeores.2010.03.002). |
| 5 | 2282, Byerly G. R. (1976) | SAME | Title, three authors, journal, volume, pages and DOI [10.1016/0012-821X(76)90248-X](https://doi.org/10.1016/0012-821X(76)90248-X) agree. |
| 6 | 6185, Haase K. M. (2002) | DIFFERENT | GEOROC concerns the Kermadec Arc–Havre Trough, DOI [10.1029/2002GC000335](https://doi.org/10.1029/2002GC000335); PetDB concerns Easter Microplate MORB, DOI [10.1016/S0009-2541(01)00327-8](https://doi.org/10.1016/S0009-2541(01)00327-8). |
| 7 | 8351, Harpp K. S. (2003) | SAME | Title, four authors, journal and DOI [10.1029/2003GC000531](https://doi.org/10.1029/2003GC000531) agree. |
| 8 | 2038, Humphris S. E. (1982) | SAME | Title, authors, journal, volume, pages and DOI [10.1016/0009-2541(82)90051-1](https://doi.org/10.1016/0009-2541(82)90051-1) agree; PetDB had recorded only the last page rather than the full range. |
| 9 | 2260, Kempton P. D. (2000) | SAME | Title, author list, journal, volume, pages and DOI [10.1016/S0012-821X(00)00047-9](https://doi.org/10.1016/S0012-821X(00)00047-9) agree. |
| 10 | 161, Martin C. E. (1991) | SAME | Title, author, journal, volume, pages and DOI [10.1016/0016-7037(91)90318-Y](https://doi.org/10.1016/0016-7037(91)90318-Y) agree. |
| 11 | 18767, Sinton J. M. (2014) | DIFFERENT | GEOROC is *Kaʻena Volcano—A precursor volcano of the island of Oʻahu*, DOI [10.1130/B30936.1](https://doi.org/10.1130/B30936.1); PetDB is a book chapter by Christopher W. Sinton et al., DOI [10.1002/9781118852538.ch16](https://doi.org/10.1002/9781118852538.ch16). |
| 12 | 656, White W. M. (1979) | DIFFERENT | GEOROC is *The petrology and geochemistry of the Azores Islands*, DOI [10.1007/BF00372322](https://doi.org/10.1007/BF00372322); PetDB is *Geochemistry of basalts from the FAMOUS area: A re-examination* in the Carnegie Institution yearbook. |
| 13 | 11979, Zellmer G. F. (2008) | SAME | Title, four authors, journal, volume and DOI [10.1016/j.epsl.2008.02.026](https://doi.org/10.1016/j.epsl.2008.02.026) agree; the one-page endpoint difference is a metadata variation. |
| 14 | 29214, Zhang Wei (2025) | DIFFERENT | GEOROC is an arc Mg-isotope paper, DOI [10.1016/j.gca.2024.12.024](https://doi.org/10.1016/j.gca.2024.12.024); PetDB is a MORB carbon paper by Zhang H. et al., DOI [10.1029/2024GL111125](https://doi.org/10.1029/2024GL111125). |

## Implementation consequence

The seven SAME pairs share one `PAPER_ID`. The seven DIFFERENT pairs retain independent `PAPER_ID` values. The pre-existing v4 train/test and inner-fold assignments remain conservative and were retained; identity fields were refreshed and checked for publication-level leakage.
