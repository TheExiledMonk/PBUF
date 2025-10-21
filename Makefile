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

RSD_DATA_RAW_DIR ?= data/rsd/raw
RSD_DATA_DERIVED_DIR ?= data/rsd/derived
RSD_PREP_RELEASE ?= nesseris2017

CC_DATA_RAW_DIR ?= data/chronometers/raw
CC_DATA_DERIVED_DIR ?= data/chronometers/derived
CC_RAW_FILE ?= $(CC_DATA_RAW_DIR)/OHD_Moresco2022.dat
CC_META ?= $(CC_DATA_DERIVED_DIR)/chronometers_index.meta.json

FETCH_RELEASE ?= main
PP_PREP_RELEASE ?= PantheonPlusDR1
Z_PREFER ?= z_cmb
COV_COMPONENTS ?= stat,sys

# -------------------------------
# Phony Targets
# -------------------------------
.PHONY: prepare-data fit-sn fit-cmb fit-bao fit-bao-ani fit-rsd fit-rsd-lcdm fit-rsd-pbuf fit-chronometers fit-joint fit-all run-all report clean-all

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
	@echo "📈 Preparing growth-rate (RSD) derived artefacts..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/data/prepare_rsd.py \
		--raw $(RSD_DATA_RAW_DIR) \
		--derived $(RSD_DATA_DERIVED_DIR) \
		--release-tag $(RSD_PREP_RELEASE)
	@echo "⏱ Fetching and preparing chronometer H(z) data..."
	$(MAKE) $(CC_META)

$(CC_RAW_FILE):
	@echo "⏬ Fetching chronometer H(z) catalog..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/data/fetch_chronometers_data.py \
		--out $(CC_DATA_RAW_DIR)

$(CC_META): $(CC_RAW_FILE)
	@echo "🛠 Preparing chronometer H(z) derived artefacts..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/data/prepare_chronometers_data.py \
		--raw $(CC_DATA_RAW_DIR) \
		--derived $(CC_DATA_DERIVED_DIR)
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
# Growth-rate (RSD) Pipeline
# -------------------------------
fit-rsd-lcdm: $(RSD_DATA_DERIVED_DIR)/rsd_index.csv $(RSD_DATA_DERIVED_DIR)/rsd_index.cov.npy
	@echo "[INFO] Running growth-rate (RSD) fit for ΛCDM..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fitters/fit_rsd.py \
		--model lcdm \
		--data-dir $(RSD_DATA_DERIVED_DIR) \
		--out $(FIT_OUT) \
		--fit

fit-rsd-pbuf: $(RSD_DATA_DERIVED_DIR)/rsd_index.csv $(RSD_DATA_DERIVED_DIR)/rsd_index.cov.npy
	@echo "[INFO] Running growth-rate (RSD) fit for PBUF..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fitters/fit_rsd.py \
		--model pbuf \
		--data-dir $(RSD_DATA_DERIVED_DIR) \
		--out $(FIT_OUT) \
		--fit

fit-rsd: fit-rsd-lcdm fit-rsd-pbuf

# -------------------------------
# Chronometer H(z) Pipeline
# -------------------------------


.PHONY: chronometers-data
chronometers-data: $(CC_META)

.PHONY: fit-chronometers
fit-chronometers: $(CC_META)
	@echo "📉 Fitting chronometer H(z) data..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fitters/fit_chronometers.py \
		--model lcdm \
		--data-dir $(CC_DATA_DERIVED_DIR) \
		--out $(FIT_OUT) \
		--fit
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fitters/fit_chronometers.py \
		--model pbuf \
		--data-dir $(CC_DATA_DERIVED_DIR) \
		--out $(FIT_OUT) \
		--fit
	@echo "✅ Chronometer fits written under $(FIT_OUT)"

# -------------------------------
# Joint Calibration (SN + BAO + CMB)
# -------------------------------
fit-joint:
	@echo "🔗 Running joint CMB+SN+BAO+BAO_ANI+CC+RSD fits..."
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fit_joint.py \
			--model lcdm \
			--datasets cmb,sn,bao,bao_ani,cc,rsd \
			--sn-dataset $(PP_DATASET) \
			--bao-priors bao_mixed \
			--bao-ani-priors bao_ani \
			--cmb-priors planck2018 \
			--chronometer-dataset moresco2022 \
			--rsd-dataset nesseris2017 \
			--out $(FIT_OUT)
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) pipelines/fit_joint.py \
			--model pbuf \
			--datasets cmb,sn,bao,bao_ani,cc,rsd \
			--sn-dataset $(PP_DATASET) \
			--bao-priors bao_mixed \
			--bao-ani-priors bao_ani \
			--cmb-priors planck2018 \
			--chronometer-dataset moresco2022 \
			--rsd-dataset nesseris2017 \
			--out $(FIT_OUT)


# -------------------------------
# Combined Fit Stage (All Components)
# -------------------------------
.PHONY: fit-all
fit-all:
	@echo "🚀 Running full fit suite in canonical order..."
	$(MAKE) fit-cmb
	$(MAKE) fit-sn
	$(MAKE) fit-bao
	$(MAKE) fit-bao-ani
	$(MAKE) fit-chronometers
	$(MAKE) fit-rsd
	$(MAKE) fit-joint

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
.PHONY: run-all
run-all:
	@echo "🏁 Running full pipeline in canonical order..."
	$(MAKE) prepare-data
	$(MAKE) fit-cmb
	$(MAKE) fit-sn
	$(MAKE) fit-bao
	$(MAKE) fit-bao-ani
	$(MAKE) fit-chronometers
	$(MAKE) fit-rsd
	$(MAKE) fit-joint
	$(MAKE) report

# -------------------------------
# Clean All Outputs
# -------------------------------
clean-all:
	@echo "🧹 Cleaning all generated results and reports..."
	@rm -rf $(FIT_OUT)/*
	@rm -rf $(REPORT_OUT)
	@find . -type f \( -name "*.png" -o -name "*.npy" -o -name "*.html" \) -path "*/proofs/results/*" -delete
	@echo "✅ Clean complete: proofs/results and reports/output cleared."
