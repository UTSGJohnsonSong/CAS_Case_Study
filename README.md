# Student Renters Insurance Pricing — CAS Predictive Modeling Case Study

Risk-based pricing analysis of a **40,071-exposure** student renters insurance
portfolio (CAS Predictive Modeling Case Competition dataset). Builds an
end-to-end pipeline from raw policy data to a defensible three-tier pricing
structure.

## Approach

**Two-part frequency × severity framework** — actuarially standard and
regulator-friendly:

- **Frequency**: logistic regression on claim occurrence (4.54% portfolio rate)
- **Severity**: Gamma GLM with log link, conditional on claim
- **Expected loss**: EL = P(claim) × E(amount | claim)

ML alternatives (GBM, neural nets) were evaluated and deliberately rejected:
marginal individual-level gains did not justify the loss of interpretability
required for regulatory rate filing.

## Key results

| Metric | Value |
|---|---|
| Portfolio calibration (predicted vs. actual EL) | $210.84 vs. $212.49 (0.8% diff) |
| Decile-level R² | 0.98 |
| Strongest risk signal | Greek affiliation: 4–5× expected-loss differential |
| Secondary signal | Off-campus housing: 2–2.5× |
| Three-tier separation (High/Low actual loss) | ~7.7× |

High risk is driven primarily by claim **frequency**, not severity — supporting
prevention-focused pricing rather than loss-cap optimization.

## Repository contents

| File | Description |
|---|---|
| `analysis.py` | Full pipeline: cleaning, EDA, GLM fitting, tier construction |
| `grid_search_optimal_par.py` | Hyperparameter/threshold search |
| `KEY_FINDINGS.txt` | Complete internal analysis summary |
| `EXEC_MEMO.txt` | Executive memo (business recommendation) |
| `01–04_*.png` | Target variable, risk signals, expected loss, tier charts |

## Run it

```bash
pip install pandas numpy scikit-learn scipy matplotlib
python analysis.py
```

*Dataset provided by the CAS case competition (included as `.xlsx`).*
