# Cosmological Model Comparison — ΛCDM vs PBUF

Generated automatically by PBUF Reporting Module.


### Overview

Comparison of statistical fit metrics for all datasets (CMB, SN, BAO_ISO, BAO_ANISO, CC, RSD).



## Dataset-Level χ² Summary

| Dataset | χ²_LCDM | χ²_PBUF | Δχ² (PBUF-LCDM) | AIC_LCDM | AIC_PBUF | ΔAIC |
|---|---|---|---|---|---|---|
| CMB | 0.0008 | 0.0011 | 0.0003 | 4.0008 | 6.0011 | 2.0003 |
| SN | 1.4338 | 24.9069 | 23.4731 | 5.4338 | 28.9069 | 23.4731 |
| BAO_ISO | 7.15e+05 | 1.56e+06 | 8.45e+05 | 7.15e+05 | 1.56e+06 | 8.45e+05 |
| BAO_ANISO | 4125.0992 | 7793.2360 | 3668.1368 | 4129.0992 | 7799.2360 | 3670.1368 |
| CC | 14.4223 | 134.4425 | 120.0202 | 18.4223 | 138.4425 | 120.0202 |
| RSD | 242.8277 | 639.6805 | 396.8528 | 248.8277 | 645.6805 | 396.8528 |

## Global Fit Summary

**LCDM** — χ²_total = 7.19e+05, AIC_total = 7.19e+05, BIC_total = 7.19e+05, reduced χ² = 1.002345
**PBUF** — χ²_total = 1.57e+06, AIC_total = 1.57e+06, BIC_total = 1.57e+06, reduced χ² = 2.004123

## Model Comparison Metrics

| Metric | Value | Preferred Model |
|---|---|---|
| ΔAIC (PBUF-LCDM) | 8.50e+05 | LCDM |
| ΔBIC (PBUF-LCDM) | 8.50e+05 | LCDM |

Interpretation: lower AIC/BIC values indicate stronger statistical evidence.


## Notes

- All χ² values are absolute, not reduced, unless noted.
- AIC and BIC computed as χ² + penalty(k), where k = number of parameters.
- ΔAIC > 10 generally indicates strong model preference.
- Data loaded from standardized `.npz`  datasets (CMB, SN, BAO, CC, RSD).
- Results generated using PBUF4 unified test suite.
