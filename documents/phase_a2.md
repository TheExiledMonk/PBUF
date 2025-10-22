🧩 PBUF Phase A.2 — Dataset Fetching, Provenance, Validation & Jackknife System
🎯 Goal

Create a unified, fault-tolerant dataset management subsystem that:

Handles both automated and manual dataset acquisition,

Enforces provenance and reproducibility via metadata + checksums,

Automatically generates jackknife subsets for robustness testing,

Prevents the joint fitter from running unless all required datasets are valid and complete,

Integrates provenance and jackknife info into both HTML and PDF reports.

🧱 Directory Layout
data/
 ├── supernovae/
 │   ├── raw/
 │   ├── derived/
 │   ├── jackknife/                 ← generated subsets
 │   └── meta/
 ├── bao/
 │   ├── raw/
 │   ├── derived/
 │   ├── jackknife/
 │   └── meta/
 ├── cmb/
 │   └── meta/
 ├── cc/
 │   ├── derived/
 │   ├── jackknife/
 │   └── meta/
 ├── rsd/
 │   ├── derived/
 │   ├── jackknife/
 │   └── meta/
 ├── manual/
 │   ├── cc_moresco2022_extracted.csv
 │   ├── rsd_nesseris2017_extracted.csv
 │   └── metadata/
 │       ├── cc_moresco2022_meta.json
 │       ├── rsd_nesseris2017_meta.json
 │       └── README_MANUAL_SOURCES.md
 └── manifest.json

⚙️ 1. Automated Dataset Downloaders

dataio/downloaders/ contains:

fetch_sn_pantheonplus.py
fetch_bao_dr16.py
fetch_cmb_planck2018.py
fetch_cc_moresco2022.py
fetch_rsd_nesseris2017.py
utils.py


Each defines a DATASET_INFO block and:

Downloads the dataset and saves checksums to meta/dataset_info.json.

Updates data/manifest.json with status = ok.

Calls the jackknife generator after verification.

🔁 2. Jackknife Subset Generation
2.1 File: dataio/utils/jackknife.py
def create_jackknife_splits(dataset_csv, n_splits=10, outdir="jackknife"):
    """
    Generate jackknife subsets by sequentially omitting 1/n_splits of the data.
    Saves each subset as CSV in data/<dataset>/jackknife/.
    """

2.2 Output

For a dataset with N entries:

Generates jackknife_01.csv … jackknife_10.csv

Each metadata JSON records:

{
  "dataset": "pantheon_plus",
  "jackknife_splits": 10,
  "points_per_split": 170,
  "method": "leave-one-group-out",
  "checksum": "sha256:..."
}

2.3 Purpose

Enables robustness checks (fit_sn_jackknife.py)

Allows χ² variance analysis across subsets

Detects sensitivity to outlier data points

🧾 3. Manual Dataset Integration

Manual tables live under data/manual/.
Each has paired metadata in data/manual/metadata/.

A helper tools/pdf_table_to_csv.py cleans text, saves CSV, computes SHA-256, and produces metadata.

Manual datasets can also use jackknifing (optional if dataset ≥ 10 points).

🧩 4. Global Manifest Management

After any successful fetch or manual import:

data/manifest.json

{
  "datasets": [
    {"name": "pantheon_plus", "type": "auto", "path": "data/supernovae/derived", "checksum": "...", "jackknife": 10},
    {"name": "cc_moresco2022", "type": "manual", "path": "data/manual/cc_moresco2022_extracted.csv", "checksum": "...", "jackknife": 0}
  ]
}

🧪 5. Validation System
5.1 Metadata + Checksum Tests

(as previously defined — unchanged)

5.2 Jackknife Validation
def test_jackknife_subsets_exist():
    for ds in manifest["datasets"]:
        jdir = Path(f"data/{ds['name']}/jackknife")
        if ds.get("jackknife",0) > 0:
            assert jdir.exists()
            files = list(jdir.glob("jackknife_*.csv"))
            assert len(files) == ds["jackknife"]

5.3 Joint Fit Data Check

Before any joint run:

REQUIRED_DATASETS = ["pantheon_plus","bao_dr16","bao_dr16_aniso",
                     "cmb_planck2018","cc_moresco2022","rsd_nesseris2017"]

def ensure_all_datasets_available(manifest_path="data/manifest.json"):
    manifest = json.load(open(manifest_path))
    available = {d["name"] for d in manifest["datasets"] if d.get("status","ok")=="ok"}
    missing = [d for d in REQUIRED_DATASETS if d not in available]
    if missing:
        raise RuntimeError(f"Joint fit aborted: missing datasets {missing}. "
                           "Run `make fetch-all` or verify manual files.")


✅ If any dataset or metadata is missing, or validation fails → abort immediately.

📊 6. Reporting Integration
6.1 HTML / PDF Provenance Table

Add jackknife column:

Dataset	DOI	License	Source	Type	SHA-256	Jackknife	Status
Pantheon+ DR1	10.3847/1538-4357/ac8b5d	CC-BY-4.0	GitHub	Auto	ab12…	10	✓
Moresco 2022	10.1093/mnras/stab3088	Fair Use	PDF Table 3	Manual	de45…	0	✓

Include in dossier note:

“Each dataset includes validated metadata and checksum records.
Jackknife subsets (N = 10 by default) were generated to test statistical robustness.”

🧰 7. CLI Integration
make fetch-all             # run all downloaders
make fetch-sn              # Pantheon+
make fetch-bao             # BAO DR16 iso+aniso
make fetch-cc              # Cosmic Chronometers (manual)
make fetch-rsd             # RSD Nesseris 2017 (manual)
make verify-datasets       # run checksum + jackknife validation
make jackknife             # regenerate all jackknife subsets
make clean-data            # safely remove derived + jackknife


If verification fails:

❌ Missing or invalid datasets: ['cc_moresco2022']
→ Run make fetch-cc before attempting joint fits.

📈 8. Jackknife Analysis Extension

Later in Phase A.3+, optional scripts:

pipelines/analyze_jackknife.py


Computes:

Mean & σ of best-fit parameters over all jackknife splits,

Δχ² variance,

Jackknife bias estimates.

Results appear in reports under “Robustness Tests”.

✅ Acceptance Criteria

All datasets produce valid metadata + checksums.

Jackknife subsets auto-generate for datasets ≥ 10 points.

make verify-datasets confirms jackknife completeness.

Joint fit aborts if required data missing or unverified.

Reports display provenance + jackknife summary.

Unit tests cover metadata, checksums, and jackknife existence.