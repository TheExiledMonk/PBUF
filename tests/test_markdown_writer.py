"""
Tests for markdown_writer module.
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from reports.markdown_writer import write_markdown_summary, _fmt


def test_fmt_function():
    """Test the _fmt helper function."""
    # Test normal numbers
    assert _fmt(1.2345) == "1.2345"
    assert _fmt(1.234567, precision=2) == "1.23"

    # Test large numbers (scientific notation)
    assert _fmt(123456.789) == "1.23e+05"

    # Test None and NaN
    assert _fmt(None) == "—"
    assert _fmt(np.nan) == "—"


def test_markdown_summary_basic():
    """Test basic markdown summary generation."""
    # Create mock stats dictionary
    stats = {
        "models": {
            "LCDM": {
                "chi2_total": 100.0,
                "AIC_total": 104.0,
                "BIC_total": 108.0,
                "chi2_reduced_total": 1.0
            },
            "PBUF": {
                "chi2_total": 120.0,
                "AIC_total": 126.0,
                "BIC_total": 132.0,
                "chi2_reduced_total": 1.2
            }
        },
        "datasets": {
            "CMB": {
                "LCDM": {"chi2": 10.0, "AIC": 14.0},
                "PBUF": {"chi2": 12.0, "AIC": 18.0}
            },
            "SN": {
                "LCDM": {"chi2": 90.0, "AIC": 90.0},
                "PBUF": {"chi2": 108.0, "AIC": 108.0}
            }
        },
        "global": {
            "comparison": {
                "ΔAIC (PBUF-LCDM)": 22.0,
                "ΔBIC (PBUF-LCDM)": 24.0,
                "preferred_model_AIC": "LCDM",
                "preferred_model_BIC": "LCDM"
            }
        }
    }

    # Test with temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Generate markdown
        output_path = write_markdown_summary(stats, tmp_path)

        # Check file was created
        assert os.path.exists(output_path)

        # Read and verify content
        with open(output_path, 'r') as f:
            content = f.read()

        # Check key sections are present
        assert "# Cosmological Model Comparison" in content
        assert "## Dataset-Level χ² Summary" in content
        assert "## Global Fit Summary" in content
        assert "## Model Comparison Metrics" in content
        assert "LCDM" in content
        assert "PBUF" in content
        assert "10.0000" in content  # CMB χ² for LCDM
        assert "12.0000" in content  # CMB χ² for PBUF

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_insufficient_models():
    """Test error when insufficient models provided."""
    stats = {
        "models": {
            "LCDM": {
                "chi2_total": 100.0,
                "AIC_total": 104.0,
                "BIC_total": 108.0,
                "chi2_reduced_total": 1.0
            }
        },
        "datasets": {}
    }

    with pytest.raises(ValueError, match="Expected at least two models"):
        write_markdown_summary(stats)


def test_default_output_path():
    """Test default output path creation."""
    stats = {
        "models": {
            "LCDM": {
                "chi2_total": 100.0,
                "AIC_total": 104.0,
                "BIC_total": 108.0,
                "chi2_reduced_total": 1.0
            },
            "PBUF": {
                "chi2_total": 120.0,
                "AIC_total": 126.0,
                "BIC_total": 132.0,
                "chi2_reduced_total": 1.2
            }
        },
        "datasets": {}
    }

    output_path = write_markdown_summary(stats)
    assert os.path.exists(output_path)
    assert "reports/output/summary_table.md" in output_path

    # Clean up
    if os.path.exists(output_path):
        os.unlink(output_path)
    # Remove directory if empty
    output_dir = Path(output_path).parent
    try:
        output_dir.rmdir()
    except:
        pass
