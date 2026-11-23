# Code-component requirements and evidence

This checklist concerns the code component only. It does not claim completion of report or video deliverables.

| Assignment task | Implementation | Exported evidence |
|---|---|---|
| Official WDI, specified 11 IDs, common year | `load_data`, `INDICATORS`, frozen source JSON | `data/provenance.json`, `data/raw/` |
| Pivot to one economy per row | `load_data`: pivot and validated country-code merge | `data/processed/country_table_observed_2022.csv` |
| Remove aggregates | `load_data`: official Region filter and eligible-code matching | Counts in `outputs/results.json`; country table |
| Keep Region/LendingType/IncomeGroup outside inputs | Explicit `FEATURES`; `representation` selects numerical columns only | `tests/check_results.py` changes labels and verifies unchanged fits |
| Inspect dimensions and missingness before treatment | `load_data` | Tables `01` through `05`; figure `01` |
| Numerical summary statistics | `eda` | Table `06_observed_summary_statistics.csv` |
| Extreme observations | `eda` and `interpretation` | Tables `07`, `21`, `23`, `27`; `DATA_QUALITY.md` |
| Six required distributions | `eda` | Figure `02_required_distributions` |
| Assess log GDP per capita | `eda`, `representation`, `robustness` | Figure `03_gdp_transform`; raw-GDP diagnostic in table `24` |
| Compare GDP, Internet, life expectancy, electricity by income group | `eda` | Figure `04_income_group_comparisons` |
| Correlation visualization and redundant variables | `eda`, `preprocess`; notebook interpretation | Figures `05`, `09`; tables `08`, `09`, `12` |
| Four specified pairwise relationships with categorical colours | `eda` | Figure `06_required_pairwise_relationships` |
| Document deletion/imputation | `load_data`, `representation` | Tables `01`, `02`, `13`; individual `imputation_flags.csv` |
| Standardize before PCA | `representation` | `standardized_model_features.csv`, fitted parameters table `13` |
| Explain PCA variance and retained count | `preprocess`; notebook section 3 | Figure `07`; table `10`; `outputs/results.json` |
| Interpret component coefficients | `preprocess`; notebook section 3 | Figure `08`; table `11`; notebook top-coefficient output and discussion |
| Apply K-means; select/justify k | `fit_clusters`, `clustering` | Figure `10`; table `14`; notebook discussion |
| Explain socioeconomic profiles | `interpretation`; notebook section 5 | Figure `14`; observed profile means, medians and counts in tables `15_*` |
| Compare clusters with income groups | `interpretation` | Figure `15`; count, row-percent and column-percent tables `16_*_IncomeGroup_*` |
| Compare clusters with geography | `interpretation` | Figure `16`; tables `16_*_Region_*` |
| Examine unusual economies | `interpretation` | Figure `18`; tables `21` through `23`; country assignments with distance, percentile, silhouette, imputation count |
| Hierarchical method, same representation | `clustering`: Ward on same matrix and k | Figures `11`, `12`, `13`; country assignments |
| Compare K-means and Ward | `interpretation` | Figure `17`; tables `18`, `19`, `20`; ARI/NMI in results JSON |
| Document hyperparameters and choices | `CONFIG`, README, notebook methods text | Fitted parameters table `13`; results JSON |
| Sensitivity to modeling decisions | `robustness` | Tables `24`, `25`, `26`; notebook section 6 |
| Source anomaly scrutiny | Flag in `load_data`; quality-exclusion diagnostic | Table `27`; preserved point-series response; `DATA_QUALITY.md` |
| Reproducible notebook/scripts with README | Executed notebook, `analysis.py`, dependencies, small source snapshots | README, validation log, integrity manifest |

Figures are provided as both PNG and PDF. Table prefixes are identifiers, not counts of files: several profile/comparison families export multiple CSVs.
