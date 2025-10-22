🪶 PBUF Phase A.3 — Visualization, Reporting & Extended Fit Integration
🎯 Goal

Implement a local HTML report generator and PDF proof dossier that automatically compile:

all per-dataset fit results,

comparative tables (PBUF vs ΛCDM),

statistical summaries (Δχ², ΔAIC, ΔBIC),

standardized plots (residuals, pulls, contours, correlations),

and embed everything into a single exportable PDF.

Simultaneously, add Cosmic Chronometers (CC) and RSD fitters and integrate them into the unified joint runner.

⚙️ 1. New Fitters: CC & RSD
1.1 Files
pipelines/
 ├── fit_cc.py
 ├── fit_rsd.py
 ├── fit_joint.py          ← updated to include CC & RSD

1.2 Cosmic Chronometers (CC)

Dataset: data/manual/cc_moresco2022_extracted.csv

Columns: z, H(z), σ_H

Formula: χ² = Σ [(Hobs − Hmodel)/σ_H]²

Model: H(z) from either GR (ΛCDM) or PBUF elastic expansion

Integrates into engine.run_fit()

Output: proofs/results/CC_<model>_.../fit_results.json

1.3 RSD (fσ₈)

Dataset: data/manual/rsd_nesseris2017_extracted.csv

Columns: z, fσ8, σ_fσ8

Model: fσ8(z) = f(z) × σ8(z) computed using growth rate from model background

Formula: χ² = Σ [(fσ8_obs − fσ8_model)/σ_fσ8]²

Output: proofs/results/RSD_<model>_.../fit_results.json

1.4 Integration into Joint

Update fit_joint.py to include:

DATASETS_REQUIRED = ["pantheon_plus", "bao_dr16", "bao_dr16_aniso",
                     "cmb_planck2018", "cc_moresco2022", "rsd_nesseris2017"]


and automatically call

from pipelines.fit_cc import likelihood_cc
from pipelines.fit_rsd import likelihood_rsd


The engine.run_fit() dispatcher must now include cc and rsd likelihood branches.

If any dataset fails validation → raise:

RuntimeError("Joint fit aborted: missing or invalid datasets — see verify-datasets output.")

🌐 2. Local HTML Report Generator
2.1 Files
reports/
 ├── templates/
 │   ├── base.html
 │   ├── dashboard.html
 │   └── proof_dossier.html
 ├── static/
 │   ├── css/report.css
 │   └── js/report.js
 ├── build_report.py
 ├── pdf_export.py
 └── output/
     ├── unified_report.html
     └── Proof-Dossier-vX.pdf

2.2 Workflow

build_report.py scans proofs/results/**/fit_results.json.

Uses Jinja2 templates to generate:

summary tables (ΔAIC / ΔBIC / Verdict badges)

dataset provenance from data/manifest.json

embedded figures (PNG or SVG)

clickable “Generate PDF Dossier” button

Saves to reports/output/unified_report.html

Opens locally via browser (python -m http.server or file://)

🖼️ 3. Plot Generation Specification

Each fit generates the following standard plots in proofs/plots/<dataset>_<model>/:

Plot ID	Description	Source Function	Output File
P1	Residuals vs Redshift	residuals_vs_z()	<dataset>_residuals_vs_z.png
P2	Pull Distribution	pull_distribution()	<dataset>_pull_distribution.png
P3	Correlation Matrix	correlation_matrix()	<dataset>_correlations.png
P4	χ² Surface (for PBUF k_sat)	chi2_surface()	<dataset>_chi2_vs_ksat.png
P5	Parameter Contours (H₀ vs Ωₘ₀ or α)	contour_plot()	<dataset>_param_contour.png
P6	Comparative ΔAIC Bar Chart	global_summary()	summary_aic_comparison.png

All plots auto-save as PNG (for embedding) and SVG (for PDF vector fidelity).

📘 4. PDF Proof Dossier Generator
4.1 Backend

Use WeasyPrint (preferred for CSS fidelity).
Fallback: reportlab.platypus if headless render fails.

4.2 Content Structure

Cover Page

Title, version, DOI, author, date, commit hash

Executive Summary

Table of Δχ², ΔAIC, ΔBIC per dataset

Dataset Sections

Summary tables

Embedded plots (P1–P3)

Best-fit parameters

ΔAIC verdict badge

Joint Fit Results

χ² breakdown

ΔAIC/AICc comparison

Combined contours

Provenance Appendix

Dataset metadata (license, DOI, checksums)

Footer

Generation timestamp + SHA-256 digest

4.3 Generation

From command line or button in dashboard:

python reports/pdf_export.py --from reports/output/unified_report.html


Outputs:
reports/output/Proof-Dossier-v3-PhaseA.pdf

🔍 5. Plotting Library & Theme

Matplotlib (default style: clean whitegrid)

Seaborn disabled by default for reproducibility

Global fonts: DejaVu Sans / CMU Serif

Resolution: 300 DPI for PDF inclusion

Color palette:

ΛCDM → #007bff (blue)

PBUF → #dc3545 (red)

Joint/Combined → #28a745 (green)

🧮 6. New Makefile Targets
make fit-cc             # run only CC fitter
make fit-rsd            # run only RSD fitter
make fit-joint          # run full joint incl. CC & RSD
make report             # build HTML report
make proof              # export PDF dossier
make serve-report       # open local dashboard


Joint run will internally call all five fitters, then aggregate.

🧪 7. Validation & Testing

pytest should verify HTML and PDF generation succeed (no missing plots).

Confirm PDF embeds images and text correctly.

Ensure all plots are regenerated when results change.

Cross-check that ΔAIC and χ² tables in PDF match JSON outputs bit-for-bit.

✅ Acceptance Criteria

fit_cc.py and fit_rsd.py run independently and via joint fit.

Missing-data check still prevents joint run.

build_report.py creates a complete interactive dashboard.

“Generate PDF” button or CLI produces a single, well-formatted dossier.

All standard plots (P1–P6) exist for every dataset.

Phase A.3 locked release includes HTML + PDF proofs with visual and numeric parity.