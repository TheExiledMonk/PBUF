#!/usr/bin/env python3
"""
Example usage of the PBUF reports module.

This script demonstrates how to generate publication-ready
Markdown and HTML summaries of cosmological model comparisons.
"""

import numpy as np
from reports.markdown_writer import write_markdown_summary
from reports.html_report import build_html_report


def create_example_stats():
    """Create example statistics for demonstration."""
    return {
        "models": {
            "LCDM": {
                "chi2_total": 719334.6211,
                "AIC_total": 719364.6211,
                "BIC_total": 719394.6211,
                "chi2_reduced_total": 1.002345,
                "n_data_total": 717500,
                "n_params_total": 6
            },
            "PBUF": {
                "chi2_total": 1568831.2328,
                "AIC_total": 1568891.2328,
                "BIC_total": 1568941.2328,
                "chi2_reduced_total": 2.004123,
                "n_data_total": 782500,
                "n_params_total": 7
            }
        },
        "datasets": {
            "CMB": {
                "LCDM": {"chi2": 0.0008, "AIC": 4.0008, "BIC": 7.0008,
                        "parameters": {"H0": 67.4, "Omega_m": 0.315, "Omega_b": 0.049}},
                "PBUF": {"chi2": 0.0011, "AIC": 6.0011, "BIC": 10.0011,
                        "parameters": {"H0": 68.1, "Omega_m": 0.298, "Omega_buf": 0.05}}
            },
            "SN": {
                "LCDM": {"chi2": 1.4338, "AIC": 5.4338, "BIC": 8.4338},
                "PBUF": {"chi2": 24.9069, "AIC": 28.9069, "BIC": 31.9069}
            },
            "BAO_ISO": {
                "LCDM": {"chi2": 7.1495e5, "AIC": 7.1495e5, "BIC": 7.1495e5},
                "PBUF": {"chi2": 1.5602e6, "AIC": 1.5602e6, "BIC": 1.5602e6}
            },
            "BAO_ANISO": {
                "LCDM": {"chi2": 4125.0992, "AIC": 4129.0992, "BIC": 4132.0992},
                "PBUF": {"chi2": 7793.2360, "AIC": 7799.2360, "BIC": 7805.2360}
            },
            "CC": {
                "LCDM": {"chi2": 14.4223, "AIC": 18.4223, "BIC": 21.4223},
                "PBUF": {"chi2": 134.4425, "AIC": 138.4425, "BIC": 141.4425}
            },
            "RSD": {
                "LCDM": {"chi2": 242.8277, "AIC": 248.8277, "BIC": 251.8277},
                "PBUF": {"chi2": 639.6805, "AIC": 645.6805, "BIC": 648.6805}
            }
        },
        "global": {
            "comparison": {
                "ΔAIC (PBUF-LCDM)": 849526.6117,
                "ΔBIC (PBUF-LCDM)": 849546.6117,
                "preferred_model_AIC": "LCDM",
                "preferred_model_BIC": "LCDM"
            }
        }
    }


def create_example_plots(plot_dir):
    """Create dummy plot files for demonstration."""
    import os
    os.makedirs(plot_dir, exist_ok=True)

    # Create dummy PNG files (1x1 pixel transparent PNGs)
    dummy_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'

    plots = [
        "hubble_comparison.png",
        "sn_distance_modulus.png",
        "bao_isotropic.png",
        "bao_anisotropic.png",
        "cmb_observables.png",
        "rsd_growth.png",
        "joint_chi2_breakdown.png"
    ]

    for plot in plots:
        with open(os.path.join(plot_dir, plot), 'wb') as f:
            f.write(dummy_png)


def main():
    """Generate example markdown and HTML summaries."""
    print("Generating example cosmological model comparison reports...")
    print("=" * 60)

    # Create example statistics
    stats = create_example_stats()

    # Generate markdown summary
    print("\n1. Generating Markdown summary...")
    md_output_file = "reports/output/example_summary.md"
    md_result_path = write_markdown_summary(stats, md_output_file)
    print(f"   ✓ Markdown summary: {md_result_path}")

    # Generate HTML report
    print("\n2. Generating HTML dashboard...")
    plot_dir = "reports/output/plots"
    create_example_plots(plot_dir)

    html_output_file = "reports/output/example_report.html"
    html_result_path = build_html_report(stats, plot_dir, html_output_file)
    print(f"   ✓ HTML dashboard: {html_result_path}")

    print("\n3. Report Statistics:")
    import os
    print(f"   • Markdown file: {get_file_size(md_result_path)}")
    print(f"   • HTML file: {get_file_size(html_result_path)}")
    print(f"   • Plot files: {len([f for f in os.listdir(plot_dir) if f.endswith('.png')])}")

    print("\n📄 Generated files can be used for:")
    print("   • Scientific papers and presentations")
    print("   • arXiv submissions and preprints")
    print("   • Project documentation and README.md")
    print("   • Interactive web dashboards")
    print("   • Reviewer responses and supplements")

    print(f"\n📂 All files saved to: reports/output/")
    print(f"   • Open {html_result_path} in a browser for the interactive dashboard")


def get_file_size(filepath):
    """Get human-readable file size."""
    import os
    size = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


if __name__ == "__main__":
    import os
    os.makedirs("reports/output", exist_ok=True)
    main()
