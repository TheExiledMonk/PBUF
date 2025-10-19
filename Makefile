# ============================================================
#  PBUF Cosmology Unified Fit — Master Makefile
# ============================================================

# -------------------------------
# Core Variables
# -------------------------------
PYTHON ?= python
PYTHONPATH ?= .

PP_DATASET ?= pantheon_plus
BAO_DATASET ?= bao
FIT_OUT ?= proofs/results
REPORT_OUT ?= reports/output/unified_report.html

PP_DATA_RAW_DIR ?= data/supernovae/raw/pantheon_plus
PP_DATA_DERIVED_DIR ?= data/supernovae/derived

BAO_DATA_RAW_DIR ?= data/bao/raw
BAO_DATA_DERIVED_DIR ?= data/bao/derived

FETCH_RELEASE ?= main
PP_PREP_RELEASE ?= PantheonPlusDR1
Z_PREFER ?= z_cmb
COV_COMPONENTS ?= stat,sys

# -------------------------------
# Phony Targets
# -------------------------------
.PHONY: prepare-data fit-sn fit-cmb fit-bao fit-bao-ani fit-joint fit-all run-all report clean-all

# -------------------------------
# Data Preparation
# -------------------------------
prepare-data:
	@echo "📦 Fetching and preparing Pantheon+ supernova data..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/data/fetch_sn_data.py \
		--source pantheon_plus \
		--out $(PP_DATA_RAW_DIR) \
		--release-tag $(FETCH_RELEASE)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/data/prepare_sn_data.py \
		--raw $(PP_DATA_RAW_DIR) \
		--derived $(PP_DATA_DERIVED_DIR) \
		--z-prefer $(Z_PREFER) \
		--compose-cov $(COV_COMPONENTS) \
		--release-tag $(PP_PREP_RELEASE)

	@echo "🌌 Fetching and preparing BAO datasets..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/data/fetch_bao_data.py \
		--out $(BAO_DATA_RAW_DIR) \
		--release-tag $(FETCH_RELEASE)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/data/prepare_bao_data.py \
		--raw $(BAO_DATA_RAW_DIR) \
		--derived $(BAO_DATA_DERIVED_DIR) \
		--release-tag $(FETCH_RELEASE)

# -------------------------------
# Supernova Fits (Pantheon+)
# -------------------------------
fit-sn:
	@echo "🌠 Running Supernova (Pantheon+) fits..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fit_sn.py --dataset $(PP_DATASET) --model lcdm --out $(FIT_OUT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fit_sn.py --dataset $(PP_DATASET) --model pbuf --out $(FIT_OUT)

# -------------------------------
# CMB Distance Priors (Planck 2018)
# -------------------------------
fit-cmb:
	@echo "🌌 Running Planck 2018 CMB fits..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fit_cmb.py --model lcdm --priors planck2018 --recomb PLANCK18 --out $(FIT_OUT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fit_cmb.py --model pbuf --priors planck2018 --recomb PLANCK18 --out $(FIT_OUT)

# -------------------------------
# BAO Isotropic (DV/rs)
# -------------------------------
fit-bao:
	@echo "📏 Running BAO isotropic fits..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fit_bao.py --model lcdm --priors bao_mixed --out $(FIT_OUT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fit_bao.py --model pbuf --priors bao_mixed --out $(FIT_OUT)

# -------------------------------
# BAO Anisotropic (DM/rs, H·rs)
# -------------------------------
fit-bao-ani:
	@echo "🌀 Running BAO anisotropic fits..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fit_aniso.py --model lcdm --priors bao_ani --out $(FIT_OUT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fit_aniso.py --model pbuf --priors bao_ani --out $(FIT_OUT)

# -------------------------------
# Joint Calibration (SN + BAO + CMB)
# -------------------------------
fit-joint:
	@echo "🔗 Running joint SN + BAO + CMB fits..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fit_joint.py \
		--model lcdm \
		--datasets sn,bao,bao_ani,cmb \
		--sn-dataset $(PP_DATASET) \
		--bao-priors bao_mixed \
		--bao-ani-priors bao_ani \
		--cmb-priors planck2018 \
		--out $(FIT_OUT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fit_joint.py \
		--model pbuf \
		--datasets sn,bao,bao_ani,cmb \
		--sn-dataset $(PP_DATASET) \
		--bao-priors bao_mixed \
		--bao-ani-priors bao_ani \
		--cmb-priors planck2018 \
		--out $(FIT_OUT)


# -------------------------------
# Combined Fit Stage (All Components)
# -------------------------------
fit-all: fit-sn fit-cmb fit-bao fit-bao-ani fit-joint

# -------------------------------
# Unified Report
# -------------------------------
report:
	@echo "📊 Generating unified comparison report..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/generate_unified_report.py \
		--inputs "$(FIT_OUT)/**/fit_results.json" \
		--out $(REPORT_OUT)
	@echo "✅ Unified report written to $(REPORT_OUT)"

# -------------------------------
# Full Pipeline (Prepare + All Fits + Report)
# -------------------------------
run-all: prepare-data fit-all report

# -------------------------------
# Clean All Outputs
# -------------------------------
clean-all:
	@echo "🧹 Cleaning all generated results and reports..."
	@rm -rf $(FIT_OUT)/*
	@rm -rf $(REPORT_OUT)
	@find . -type f \( -name "*.png" -o -name "*.npy" -o -name "*.html" \) -path "*/proofs/results/*" -delete
	@echo "✅ Clean complete: proofs/results and reports/output cleared."
