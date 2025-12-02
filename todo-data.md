# 📊 COSMOS2 DATA INVENTORY & PLUGIN REQUIREMENTS

## 🔍 **SCIENCE RUN ANALYSIS**
**Run Directory**: `data/science_runs/full_joint/2025-11-29T013844_enhanced_jackknife_config`
**Models**: LCDM, PBUF
**Datasets**: CMB, SN, BAO_iso, CC, RSD
**Jackknife**: Level 1 (65 draws) + Level 2 (3 draws)

---

## 📋 **COMPLETE DATA INVENTORY**

### **🔧 CONFIGURATION FILES**

#### **1. config_used.json** (8,437 bytes)
- **Purpose**: Main run configuration
- **Content**: 
  - Models: [`lcdm`, `pbuf`]
  - Fits: [`cmb`, `sn`, `bao_iso`, `cc`, `rsd`]
  - Engine: basin walker with 20 threads
  - Jackknife: Level 1 + Level 2 enabled
  - Parameter bounds for both models
- **Plugin Need**: `ConfigurationPlugin` - Display run settings

#### **2. joint_config_used.json** (75 bytes)
- **Purpose**: Joint fit configuration
- **Content**: Minimal config reference
- **Plugin Need**: Include in ConfigurationPlugin

#### **3. engine_settings.json** (165 bytes)
- **Purpose**: Engine-specific settings
- **Content**: Basin walker parameters
- **Plugin Need**: Include in ConfigurationPlugin

#### **4. datasets_used.json** (96 bytes)
- **Purpose**: Dataset list
- **Content**: Active datasets
- **Plugin Need**: Include in ConfigurationPlugin

#### **5. checkpoint.json** (357 bytes)
- **Purpose**: Run state/progress
- **Content**: Checkpoint information
- **Plugin Need**: Include in ConfigurationPlugin

---

### **📈 MODEL RESULTS**

#### **6. model_summaries.json** (686,221 bytes) ⭐ **PRIMARY DATA**
- **Purpose**: Complete model results
- **Content**:
  ```json
  {
    "lcdm": {
      "best": {
        "parameters": {H0, Omega_m0, Omega_b0, Omega_k0},
        "chi_squared": 1206.38,
        "fit_results": {
          "cmb": {chi2: 7.22, predictions, observed, residuals},
          "sn": {chi2: 1113.0, ...},
          "bao_iso": {chi2: 3.37, ...},
          "cc": {chi2: 29.86, ...},
          "rsd": {chi2: 22.52, ...}
        }
      }
    },
    "pbuf": {
      "best": {
        "parameters": {H0, Omega_m0, Omega_b0, Rmax},
        "chi_squared": 948.0,
        "fit_results": {...}
      }
    }
  }
  ```
- **Plugin Need**: ✅ **COVERED** - `ModelDetailsPlugin` + `ModelComparisonPlugin`

---

### **🔬 JACKKNIFE ANALYSIS**

#### **7. jackknife_level1_results.json** (58,405 bytes) ⭐ **CRITICAL**
- **Purpose**: Data-level jackknife (65 draws)
- **Content**:
  ```json
  {
    "level": "data",
    "n_draws": 65,
    "parameter_shifts": {
      "Omega_m0": {mean: 0.297, std: 0.045, min: 0.25, max: 0.35},
      "Omega_b0": {mean: 0.038, std: 0.011, min: 0.02, max: 0.05},
      "H0": {mean: 70.46, std: 3.57, min: 65.0, max: 75.0}
    },
    "chi2_changes": {
      "sn_random_percentage": {mean: 1149.74, std: 14.41},
      "sn_redshift_bands": {...},
      "cmb_specific_points": {...},
      "bao_random_percentage": {...},
      "cc_random_percentage": {...},
      "rsd_random_percentage": {...}
    },
    "draws": [
      {
        "draw_index": 0,
        "model_specific_results": {
          "lcdm": {parameters, chi_squared: 1143.73},
          "pbuf": {parameters: null, chi_squared: "inf"}  // ❌ PBUF FAILED
        }
      }
    ]
  }
  ```
- **Issues**: PBUF jackknife results are `null`/`inf` - needs investigation
- **Plugin Need**: ❌ **MISSING** - `JackknifeAnalysisPlugin` (CRITICAL)

#### **8. jackknife_level2_results.json** (25,752 bytes) ⭐ **IMPORTANT**
- **Purpose**: Optimization-level jackknife (3 draws)
- **Content**: Walker/seed/minima removal analysis
- **Plugin Need**: ❌ **MISSING** - `JackknifeAnalysisPlugin` (EXTENSION)

#### **9. jackknife_summary.json** (4,799 bytes)
- **Purpose**: Aggregated jackknife summary
- **Content**: Summary statistics across levels
- **Plugin Need**: Include in JackknifeAnalysisPlugin

#### **10. jackknife_combined_results.json** (79,283 bytes)
- **Purpose**: Combined analysis across levels
- **Content**: Comprehensive jackknife results
- **Plugin Need**: Include in JackknifeAnalysisPlugin

#### **11. jackknife_report.md** (4,538 bytes)
- **Purpose**: Text-based jackknife summary
- **Content**: Human-readable jackknife analysis
- **Plugin Need**: Include in JackknifeAnalysisPlugin

---

### **📊 DATA TABLES**

#### **12. best_fit_parameters.csv** (211 bytes)
- **Purpose**: Parameter comparison table
- **Content**: LCDM vs PBUF parameters (all zeros - broken)
- **Issues**: All values are 0.000 - data not populated
- **Plugin Need**: ✅ **COVERED** - `DataTablesPlugin` (with validation)

#### **13. chi2_breakdown_detailed.csv** (112 bytes)
- **Purpose**: χ² breakdown by dataset
- **Content**: Dataset χ² contributions (all zeros - broken)
- **Issues**: All values are 0.000 - data not populated
- **Plugin Need**: ✅ **COVERED** - `DataTablesPlugin` (with validation)

#### **14. model_comparison.csv** (141 bytes)
- **Purpose**: Model comparison metrics
- **Content**: χ², AIC, BIC comparison (all zeros - broken)
- **Issues**: All values are 0.000 - data not populated
- **Plugin Need**: ✅ **COVERED** - `DataTablesPlugin` (with validation)

#### **15. full_data_summary.csv** (189 bytes)
- **Purpose**: Overall data summary
- **Content**: Summary statistics (all zeros - broken)
- **Issues**: All values are 0.000 - data not populated
- **Plugin Need**: ✅ **COVERED** - `DataTablesPlugin` (with validation)

#### **16. quantum_engine_input.csv** (1,286 bytes) ⭐ **PBUF-SPECIFIC**
- **Purpose**: PBUF quantum engine parameters
- **Content**:
  ```csv
  Parameter,Value,Unit,Description,Source
  Regulator Type,—,,Type of UV regularization scheme
  UV Cutoff (f_cut),0.00e+00,GeV,Maximum frequency scale
  Coupling Scale (f_coup),0.00e+00,,Quantum-gravity coupling strength
  Maximum Rigidity (R_max),0.00e+00,GeV⁻¹,Maximum spacetime rigidity
  LUT Type,Bootstrap,,Type of lookup table used
  Interpolation Method,—,,Interpolation scheme for thermal quantities
  ```
- **Issues**: All numerical values are 0.000 - data not populated
- **Plugin Need**: ❌ **MISSING** - `QuantumEnginePlugin` (PBUF-specific)

#### **17. quantum_engine_output.csv** (1,694 bytes) ⭐ **PBUF-SPECIFIC**
- **Purpose**: PBUF thermal LUT output
- **Content**:
  ```csv
  z,a,T [GeV],ε₀(T) [GeV⁴],α(T),g*,g*S,dε₀/dT,dα/dT,Provenance
  0.000,1.0000,1.00,3.09e-19,0.1000,3.4,3.9,0.00e+00,0.00e+00,bootstrap_low_T
  999.000,0.0010,1.00e+06,2.04e+06,0.1500,106.8,106.8,937.55,6.07e-09,bootstrap_intermediate
  ```
- **Status**: ✅ **HAS REAL DATA** - Thermal evolution table
- **Plugin Need**: ❌ **MISSING** - `QuantumEnginePlugin` (PBUF-specific)

**Note**: All tables have `.md` and `.tex` versions too

---

### **📊 VISUALIZATIONS**

#### **18. hubble_diagram.png** (168,779 bytes) ⭐ **ONLY FIGURE**
- **Purpose**: Hubble diagram visualization
- **Content**: Distance vs redshift plot
- **Plugin Need**: ❌ **MISSING** - `FiguresPlugin`

---

### **📄 EXISTING REPORTS**

#### **19. model_comparison_report.html** (262,175 bytes)
- **Purpose**: Old monolithic report
- **Content**: Previous generation report
- **Plugin Need**: Reference for migration

#### **20. cosmos2_report_20251129_104217.html** (25,093 bytes)
- **Purpose**: New modular report (our output)
- **Content**: Successfully generated modular report
- **Plugin Need**: ✅ **SUCCESS** - Our system works!

---

## 🎯 **PLUGIN DEVELOPMENT PLAN**

### **❌ MISSING CRITICAL PLUGINS**

#### **1. JackknifeAnalysisPlugin** ⭐ **HIGH PRIORITY**
- **Purpose**: Display jackknife stability analysis
- **Data Sources**: 
  - `jackknife_level1_results.json` (primary)
  - `jackknife_level2_results.json` (extension)
  - `jackknife_summary.json` (summary)
- **Content Needed**:
  - Parameter stability tables (mean, std, range)
  - χ² variation analysis
  - Dataset removal impact
  - Model-specific jackknife results
- **Issues to Address**: PBUF jackknife failures (null/inf results)
- **Section Type**: `individual` (under each model)

#### **2. QuantumEnginePlugin** ⭐ **MEDIUM PRIORITY**
- **Purpose**: Display PBUF quantum engine data
- **Data Sources**:
  - `quantum_engine_input.csv` (parameters)
  - `quantum_engine_output.csv` (thermal LUT)
- **Content Needed**:
  - Quantum parameter tables
  - Thermal evolution plots
  - PBUF-specific analysis
- **Section Type**: `individual` (under PBUF model only)

#### **3. FiguresPlugin** ⭐ **MEDIUM PRIORITY**
- **Purpose**: Display all generated figures
- **Data Sources**:
  - `figures/` directory
  - Currently: `hubble_diagram.png`
- **Content Needed**:
  - Figure gallery
  - Embedded images with captions
  - Figure metadata
- **Section Type**: `standalone`

#### **4. ConfigurationPlugin** ⭐ **LOW PRIORITY**
- **Purpose**: Display run configuration
- **Data Sources**:
  - `config_used.json`
  - `engine_settings.json`
  - `datasets_used.json`
- **Content Needed**:
  - Run parameters
  - Dataset information
  - Engine settings
- **Section Type**: `standalone`

---

## 🔧 **DATA QUALITY ISSUES**

### **❌ BROKEN TABLES**
- **Issue**: All CSV tables (except quantum) contain only zeros
- **Affected**: `best_fit_parameters.csv`, `chi2_breakdown_detailed.csv`, `model_comparison.csv`, `full_data_summary.csv`
- **Cause**: Table generation pipeline issue
- **Solution**: Fix table generation OR extract from `model_summaries.json`

### **❌ PBUF JACKKNIFE FAILURES**
- **Issue**: PBUF jackknife results are `null`/`inf`
- **Location**: `jackknife_level1_results.json`
- **Cause**: PBUF optimization fails in jackknife draws
- **Solution**: Debug PBUF jackknife optimization

### **✅ WORKING DATA**
- **model_summaries.json** ✅ - Complete model results
- **jackknife_level1_results.json** ✅ - LCDM jackknife works
- **quantum_engine_output.csv** ✅ - Real thermal data
- **hubble_diagram.png** ✅ - Figure exists

---

## 📋 **IMPLEMENTATION PRIORITY**

### **🚀 IMMEDIATE (Critical)**
1. **JackknifeAnalysisPlugin** - Essential for stability analysis
2. **Fix table data extraction** - Extract real data from JSON
3. **Debug PBUF jackknife failures** - Investigate PBUF issues

### **⚡ SHORT TERM (Important)**
4. **QuantumEnginePlugin** - PBUF-specific quantum data
5. **FiguresPlugin** - Visual content display
6. **ConfigurationPlugin** - Run metadata

### **📈 MEDIUM TERM (Enhancement)**
7. **Data validation improvements** - Better error handling
8. **Additional output formats** - PDF, LaTeX
9. **More themes** - Academic, presentation themes

---

## 🎯 **SUCCESS METRICS**

### **✅ CURRENTLY WORKING**
- [x] Model comparison (neutral language)
- [x] Individual model details
- [x] Basic table display (with validation)
- [x] Professional HTML theme
- [x] Modular plugin architecture

### **❌ MISSING FOR COMPLETENESS**
- [ ] Jackknife stability analysis
- [ ] PBUF quantum engine data
- [ ] Figure gallery
- [ ] Configuration display
- [ ] Fixed table data extraction

### **🎯 END STATE**
- [ ] Complete data coverage (100% of files displayed)
- [ ] All data types properly validated
- [ ] Professional publication-ready reports
- [ ] Easy plugin extension system
- [ ] Multiple output formats

---

## 📊 **DATA SUMMARY**

| **Category** | **Files** | **Size** | **Status** | **Plugin Coverage** |
|-------------|----------|---------|------------|-------------------|
| **Model Results** | 1 | 686KB | ✅ Complete | ✅ Covered |
| **Jackknife** | 4 | 168KB | ⚠️ Partial | ❌ Missing |
| **Tables** | 12 | 8KB | ❌ Broken | ⚠️ Partial |
| **Figures** | 1 | 169KB | ✅ Complete | ❌ Missing |
| **Config** | 5 | 9KB | ✅ Complete | ❌ Missing |
| **Quantum** | 2 | 3KB | ⚠️ Partial | ❌ Missing |

**Total Data**: 1.04MB across 25 files
**Current Coverage**: ~40% (model results only)
**Target Coverage**: 100% (all data types)

---

## 🚀 **NEXT STEPS**

1. **Implement JackknifeAnalysisPlugin** - Critical for stability analysis
2. **Fix table data extraction** - Extract from JSON instead of broken CSVs
3. **Create QuantumEnginePlugin** - PBUF-specific quantum data
4. **Add FiguresPlugin** - Display visualizations
5. **Add ConfigurationPlugin** - Run metadata display

**Goal**: Complete data coverage with professional publication-ready reports! 🎯
