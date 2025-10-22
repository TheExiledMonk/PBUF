---
title: "Planck-Bound Unified Framework (PBUF) — Methodology and Verification Protocol"
author: "Fabian Olesen"
affiliation: "Independent Researcher, Cebu / Toledo, Philippines"
version: "v3.0-draft"
date: "2025-10-22"
doi: "10.5281/zenodo.17394412"
repository: "https://github.com/TheExiledMonk/PBUF"
license: "MIT (code), CC-BY-4.0 (documentation)"
---

# Methodology

## 1. Experimental Philosophy

The **Planck-Bound Unified Framework (PBUF)** is developed under the principle of **empirical falsifiability before interpretation**.  
Rather than constructing a theoretical narrative and tuning data to fit it, PBUF enforces a bottom-up validation pipeline where each dataset independently tests the model’s internal consistency.

Every equation, dataset, and numerical result can be traced, reproduced, and verified from open-source code and public archives.  
This approach ensures that PBUF evolves as a *testable physical framework* rather than a descriptive hypothesis.

---

## 2. Reproducible Computational Framework

All numerical results are generated from a unified, fully reproducible codebase.  
Each fit (CMB, SN, BAO, CC, RSD, and joint) is automatically logged with:

- **Commit hash** and **environment fingerprint**  
- **Dataset checksums** and metadata  
- **Numerical tolerances** and convergence diagnostics  
- **Full provenance logs** (optimizer type, iteration count, χ² trajectory)

Any independent researcher can exactly reproduce the published results with no hidden parameters, random seeds, or untracked configurations.

---

## 3. Multi-Layer Verification

To prevent numerical or conceptual drift, PBUF implements several independent verification layers:

- **Cross-implementation verification:** results are reproduced across multiple independently written codebases (v1, v2, v3).  
- **Unit and integrity testing:** over 40 automated tests validate recombination physics, covariance conditioning, H(z) ratios, and sound horizon predictions.  
- **Mock-to-real transition validation:** all pipelines are tested using synthetic datasets before applying real observational data.  
- **Parameter propagation control:** global parameters are centralized and automatically verified between fit stages.

These safeguards ensure that every reported χ² value and AIC/BIC metric is both physically and numerically trustworthy.

---

## 4. Data Provenance and Integrity

All observational datasets — Pantheon+, BAO DR16, Planck 2018, and others — are acquired through automated or semi-manual data importers with full checksum validation.  
Each dataset includes a structured metadata file describing:

| Field | Description |
|--------|-------------|
| **Source DOI / Reference** | Original publication or repository |
| **License** | Legal terms for scientific use |
| **Checksum** | SHA-256 integrity hash |
| **Extraction Method** | Automated or manual transcription |
| **Fields & Units** | Explicit column and unit definitions |

The joint fitter will **refuse to execute** if any required dataset is missing, corrupted, or fails validation — ensuring no silent partial runs.

---

## 5. Optimization and Consistency

Parameter optimization is first performed using **CMB distance priors**, providing physically calibrated baselines for both ΛCDM and PBUF models.  
These optimized parameters are then propagated across BAO, SN, CC, and RSD fits to maintain cosmological coherence.

Each optimization run records:

- χ² improvement relative to prior defaults  
- Optimization method, bounds, and convergence diagnostics  
- Provenance metadata (timestamp, checksum, environment)

This guarantees consistent use of best-fit parameters across all subsequent analyses.

---

## 6. Transparency and Falsifiability

All plots, tables, and reported fits are generated directly from open-source scripts — no manual adjustments or cosmetic edits are performed.  
The framework is designed so that **any researcher can disprove PBUF** by rerunning the same pipeline on the same data and observing a contradiction.

As of the current release, all independent implementations yield consistent results within numerical precision.

---

## 7. Verification Artifacts

| Artifact | Description | Location |
|-----------|-------------|-----------|
| `proof_dossier.pdf` | Summary of empirical fits and χ² metrics | `/proofs/results/` |
| `unified_report.html` | Interactive dashboard for parameters and residuals | `/reports/output/` |
| `manifest.json` | Dataset provenance and checksum registry | `/data/manifest.json` |
| `fit_results.json` | Detailed numerical results for each experiment | `/proofs/results/<experiment>/` |

These files provide a complete audit trail for every published number.

---

## 8. Open Science Commitment

All code, data, and documentation are released under open licenses  
(**MIT** for source code, **CC-BY-4.0** for documentation and derived content).  

Every public release is fingerprinted via **Zenodo DOI** and **Git commit hash**, ensuring verifiable reproducibility and authorship traceability.  
Reproducibility is not optional — it is the foundation of the framework.

---

> **Summary:**  
> PBUF’s methodology prioritizes transparency, reproducibility, and falsifiability at every stage.  
> Each fit is not merely a result but a verifiable experiment that can be rerun by anyone.  
> The framework embodies a return to the core scientific ideal: **to make reality, not rhetoric, the final judge.**
