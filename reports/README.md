# PBUF4 Reports Module

A comprehensive scientific reporting system for cosmological model comparison between ΛCDM and PBUF frameworks.

## 🚀 Overview

The `reports` module provides publication-ready reporting tools for cosmological model fitting results. It generates interactive HTML dashboards, LaTeX-compatible Markdown summaries, structured JSON data, and scientific visualizations that can be directly embedded in scientific papers, README files, or converted to PDF.

## ✨ Features

- **🖥️ Interactive HTML Dashboard**: Self-contained web interface with embedded plots and statistics
- **📄 Publication-ready Markdown**: LaTeX-compatible tables for scientific papers
- **📊 Machine-readable JSON**: Structured data for further analysis and reproducibility
- **🖼️ Scientific Visualizations**: Publication-quality plots (H(z), μ(z), BAO, CMB, RSD)
- **📈 Statistical Analysis**: χ², AIC, BIC, ΔAIC, ΔBIC with proper model comparison
- **🔬 Professional Output**: Follows cosmological literature standards
- **⚡ Complete Pipeline**: Automated generation of all report formats

## 🚀 Quick Start

### Using the Virtual Environment

The project uses a virtual environment with all required dependencies:

```bash
# Activate the virtual environment
source venv/bin/activate

# Run the example reports
cd /home/fabian/PBUF4
PYTHONPATH=/home/fabian/PBUF4 python reports/example_usage.py
```

### Generating Reports

#### Option 1: Full Pipeline (Recommended)

```python
from reports.report_pipeline import build_full_report

# Generate complete report suite
result = build_full_report(
    models=["lcdm", "pbuf"],
    output_dir="reports/output/",
    formats=["html", "md", "json"],  # Skip PDF for now
    verbose=True
)
```

#### Option 2: Individual Components

```python
from reports.html_report import build_html_report
from reports.markdown_writer import write_markdown_summary
from reports.plotter import generate_all_plots
from reports.summary_builder import compute_model_stats, collect_fit_results

# Generate HTML dashboard
build_html_report(stats, plot_dir, "reports/output/report.html")

# Generate markdown summary
write_markdown_summary(stats, "reports/output/summary.md")

# Generate plots
generate_all_plots(stats, "reports/output/plots/")
```

## 📁 Output Files

The report system generates multiple output formats in `reports/output/`:

```
reports/output/
├── report.html              # Interactive HTML dashboard (16KB)
├── summary_table.md         # Markdown summary tables (1.6KB)
├── results.json            # Machine-readable JSON data (2.9KB)
└── plots/                  # Scientific plots (PNG)
    ├── hubble_comparison.png      # H(z) evolution
    ├── sn_distance_modulus.png    # Supernova Hubble diagram
    ├── bao_isotropic.png          # BAO D_V/r_d measurements
    ├── bao_anisotropic.png        # BAO D_M/r_d and H(z)r_d/c
    ├── cmb_observables.png        # CMB distance priors
    ├── rsd_growth.png            # fσ₈(z) growth rate
    └── joint_chi2_breakdown.png  # χ² contribution by dataset
```

## 🔧 Module Architecture

### Core Components

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `summary_builder.py` | Statistical analysis | `collect_fit_results()`, `compute_model_stats()` |
| `plotter.py` | Scientific visualization | `generate_all_plots()` |
| `html_report.py` | Interactive dashboard | `build_html_report()` |
| `markdown_writer.py` | Publication tables | `write_markdown_summary()` |
| `json_exporter.py` | Data export | `export_json()` |
| `pdf_exporter.py` | PDF generation | `export_pdf()` |
| `report_pipeline.py` | Orchestration | `build_full_report()` |

### Data Flow

```
Fit Results (data/results/*.json)
    ↓
collect_fit_results() → model statistics
    ↓
compute_model_stats() → χ², AIC, BIC, ΔAIC
    ↓
generate_all_plots() → scientific figures
    ↓
build_html_report() → interactive dashboard
write_markdown_summary() → publication tables
export_json() → reproducible data
```

## 📊 Report Contents

### HTML Dashboard (`report.html`)

- **Per-dataset statistics**: χ², AIC, BIC tables for each dataset (CMB, SN, BAO, CC, RSD)
- **Global model comparison**: Total χ², ΔAIC, ΔBIC, model preference metrics
- **Best-fit parameters**: Parameter cards showing fitted cosmological parameters
- **Diagnostic plots**: All key scientific visualizations with captions
- **Audit trail**: Expandable raw JSON data for reproducibility
- **Methodology notes**: Scientific context and disclaimers
- **Self-contained**: No external dependencies, works offline

### Markdown Summary (`summary_table.md`)

- **Statistical tables**: Formatted for easy inclusion in papers
- **Model comparison**: Δχ², ΔAIC, ΔBIC analysis with interpretation
- **Parameter summary**: Best-fit values with proper significant figures
- **LaTeX compatible**: Can be converted using Pandoc

### JSON Export (`results.json`)

- **Machine readable**: Structured data for further analysis
- **Complete statistics**: All computed metrics and parameters
- **Reproducible**: Full audit trail for scientific validation
- **Version controlled**: Can be committed to git for reproducibility

## 🧪 Testing

### Run All Tests

```bash
# Using virtual environment
source venv/bin/activate
cd /home/fabian/PBUF4
PYTHONPATH=/home/fabian/PBUF4 python -m pytest tests/ -v
```

### Example Usage Script

```bash
# Generate example reports with mock data
PYTHONPATH=/home/fabian/PBUF4 python reports/example_usage.py
```

This creates sample reports demonstrating all functionality with realistic cosmological data.

## 🔬 Scientific Features

### Supported Datasets

- **CMB**: Planck 2018 distance priors (R, ℓ_A, θ*)
- **SN**: Supernova distance moduli (Pantheon+ style)
- **BAO**: Baryon acoustic oscillations (isotropic and anisotropic)
- **CC**: Cosmic chronometers (H(z) measurements)
- **RSD**: Redshift space distortions (fσ₈ growth rate)

### Model Comparison Metrics

- **χ² analysis**: Absolute and reduced χ² per dataset
- **Information criteria**: AIC and BIC with parameter penalties
- **Model preference**: ΔAIC and ΔBIC analysis
- **Statistical significance**: Evidence ratios and thresholds
- **Parameter estimation**: Best-fit values with proper uncertainties

### Visualization Suite

- **Expansion history**: H(z) evolution comparison between models
- **Distance ladder**: Supernova Hubble diagram μ(z) with residuals
- **BAO rulers**: D_V/r_d and anisotropic measurements
- **CMB constraints**: Distance priors in cosmological parameter space
- **Growth rate**: fσ₈(z) from redshift space distortions
- **χ² diagnostics**: Breakdown of contributions by dataset

## 🛠 Development

### Adding New Datasets

1. Create data loader in `data_interface/`
2. Add converter in `data_interface/` following existing patterns
3. Update `collect_fit_results()` to scan for new data types
4. Add dataset-specific plots in `plotter.py`

### Customizing Reports

1. **CSS styling**: Modify `_base_css()` in `html_report.py`
2. **New sections**: Add section builders in `html_report.py`
3. **Table formats**: Customize in `markdown_writer.py`
4. **Plot types**: Extend visualization in `plotter.py`

### Dependencies

All dependencies are managed in `requirements.txt`:

```bash
# Core scientific computing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Plotting and visualization
matplotlib>=3.7.0

# Testing
pytest>=7.0.0
```

Install with:
```bash
./venv/bin/pip install -r requirements.txt
```

## 📚 Scientific Standards

The reporting system follows cosmological literature conventions:

- **Statistical rigor**: Proper χ², AIC, BIC calculations with degrees of freedom
- **Model comparison**: ΔAIC/ΔBIC analysis with evidence thresholds
- **Numerical precision**: Consistent significant figures (4-6 digits)
- **Notation standards**: χ², not chi-squared; proper Greek symbols
- **Transparency**: Dataset-level statistics for validation
- **Reproducibility**: Complete audit trails and structured exports

## 🎯 Integration with CLI

The report system is designed to integrate with the project's CLI:

```bash
# Future CLI integration
python cli.py report --models lcdm,pbuf --formats html,pdf,md,json
```

This will be implemented as the CLI system is developed.

## 📖 Examples

See the generated example reports in `reports/output/` after running:

```bash
PYTHONPATH=/home/fabian/PBUF4 python reports/example_usage.py
```

The HTML dashboard can be opened directly in any web browser for interactive exploration of the cosmological model comparison results.
