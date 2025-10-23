# Dataset Registry CLI Usage Guide

The Dataset Registry CLI provides comprehensive management capabilities for the PBUF dataset registry system. This tool allows you to list datasets, check status, re-verify data integrity, perform cleanup operations, and generate various types of reports.

## Installation and Setup

The CLI is automatically available when the dataset registry is installed. You can use it in two ways:

1. **Direct Python execution:**
   ```bash
   python3 pipelines/dataset_registry/cli.py [command] [options]
   ```

2. **Wrapper script (recommended):**
   ```bash
   ./dataset_cli [command] [options]
   ```

## Available Commands

### 1. List Datasets (`list`)

List all datasets in the registry with optional filtering and formatting.

```bash
# List all datasets
./dataset_cli list

# List only verified datasets
./dataset_cli list --status verified

# List CMB datasets with detailed information
./dataset_cli list --type cmb --details

# Export dataset list as CSV
./dataset_cli list --format csv

# List failed datasets
./dataset_cli list --status failed
```

**Options:**
- `--format`: Output format (`table`, `json`, `csv`)
- `--status`: Filter by status (`verified`, `failed`, `corrupted`)
- `--type`: Filter by dataset type (`cmb`, `bao`, `sn`)
- `--details`: Show detailed information

### 2. Dataset Status (`status`)

Show detailed status information for a specific dataset.

```bash
# Show detailed status for a dataset
./dataset_cli status cmb_planck2018
```

This displays:
- Verification status and timestamps
- File paths and checksums
- Source information
- Environment details
- Manifest metadata (if available)

### 3. Dataset Verification (`verify`, `verify-all`)

Re-verify dataset integrity and update registry entries.

```bash
# Re-verify a specific dataset
./dataset_cli verify cmb_planck2018

# Force re-verification even if recently verified
./dataset_cli verify cmb_planck2018 --force

# Re-verify all datasets
./dataset_cli verify-all

# Force re-verification of all datasets
./dataset_cli verify-all --force
```

**Options:**
- `--force`: Force verification even if recently verified

### 4. Integrity Check (`integrity`)

Check the integrity of the registry system and datasets.

```bash
# Check registry and manifest integrity
./dataset_cli integrity
```

This performs:
- Registry file integrity validation
- Manifest schema validation
- Orphaned file detection
- Consistency checks

### 5. Cleanup Operations (`cleanup`)

Clean up orphaned files and unused data.

```bash
# Show what would be cleaned up (dry run)
./dataset_cli cleanup --dry-run

# Actually perform cleanup
./dataset_cli cleanup --no-dry-run
```

**Options:**
- `--dry-run`: Show what would be deleted without deleting (default)
- `--no-dry-run`: Actually delete orphaned files

### 6. Export Data (`export`)

Export registry data in various formats for different purposes.

```bash
# Export registry summary as JSON
./dataset_cli export

# Export manifest table as Markdown
./dataset_cli export --type manifest --format markdown

# Export provenance bundle for publication
./dataset_cli export --type provenance --output provenance_bundle.json

# Export audit trail as CSV
./dataset_cli export --type audit --format csv --output audit_trail.csv

# Export registry summary as LaTeX table
./dataset_cli export --format latex --output registry_table.tex
```

**Export Types:**
- `registry`: Complete registry summary (default)
- `manifest`: Manifest dataset table
- `provenance`: Provenance bundle for publications
- `audit`: Audit trail data

**Formats:**
- `json`: JSON format (default)
- `markdown`: Markdown tables
- `csv`: Comma-separated values
- `latex`: LaTeX tables
- `yaml`: YAML format

### 7. Audit Trail (`audit`)

View audit trail and operation history.

```bash
# Show recent audit trail
./dataset_cli audit

# Show audit trail for specific dataset
./dataset_cli audit --dataset cmb_planck2018

# Show last 100 audit entries
./dataset_cli audit --limit 100
```

**Options:**
- `--dataset`: Filter by specific dataset
- `--limit`: Maximum number of entries to show (default: 50)

### 8. Generate Summaries (`summary`)

Generate specialized summary reports for different audiences.

```bash
# Generate operational summary
./dataset_cli summary --type operational

# Generate executive summary as Markdown
./dataset_cli summary --type executive --format markdown

# Generate publication summary
./dataset_cli summary --type publication --output pub_summary.json

# Generate compliance report
./dataset_cli summary --type compliance --format markdown --output compliance_report.md

# Generate technical summary
./dataset_cli summary --type technical
```

**Summary Types:**
- `operational`: For monitoring and operations (default)
- `executive`: High-level overview for management
- `publication`: For papers and reports
- `compliance`: For audit and compliance
- `technical`: Detailed technical information

## Global Options

All commands support these global options:

- `--registry-path`: Path to registry directory (default: `data/registry`)
- `--manifest-path`: Path to manifest file (default: `data/datasets_manifest.json`)

## Examples

### Daily Operations

```bash
# Check system health
./dataset_cli integrity

# List any failed datasets
./dataset_cli list --status failed

# Re-verify failed datasets
./dataset_cli verify-all --force
```

### Publication Preparation

```bash
# Generate dataset table for paper
./dataset_cli export --type manifest --format latex --output dataset_table.tex

# Create provenance bundle
./dataset_cli export --type provenance --output provenance_bundle.json

# Generate publication summary
./dataset_cli summary --type publication --format markdown --output dataset_summary.md
```

### Compliance and Audit

```bash
# Generate compliance report
./dataset_cli summary --type compliance --output compliance_report.json

# Export complete audit trail
./dataset_cli export --type audit --format csv --output full_audit_trail.csv

# Check registry integrity
./dataset_cli integrity
```

### Troubleshooting

```bash
# Check for orphaned files
./dataset_cli cleanup --dry-run

# Show detailed status for problematic dataset
./dataset_cli status problematic_dataset

# View recent activity
./dataset_cli audit --limit 20

# Re-verify specific dataset with force
./dataset_cli verify problematic_dataset --force
```

## Output Formats

### Table Format (default for `list`)
```
Name                Status        Source       
cmb_planck2018     ✓ VERIFIED    downloaded   
bao_compilation    ✗ FAILED      downloaded   
```

### JSON Format
```json
{
  "datasets": [
    {
      "name": "cmb_planck2018",
      "status": "verified",
      "source_type": "downloaded"
    }
  ]
}
```

### Markdown Format
```markdown
| Dataset | Status | Source Type |
|---------|--------|-------------|
| cmb_planck2018 | ✓ verified | downloaded |
```

### CSV Format
```csv
name,status,source_type
cmb_planck2018,verified,downloaded
```

## Error Handling

The CLI provides clear error messages and suggestions:

- **Dataset not found**: Lists available datasets
- **Verification failures**: Shows specific errors and suggestions
- **File permission issues**: Provides resolution steps
- **Configuration problems**: Suggests fixes

## Integration with PBUF Workflows

The CLI integrates seamlessly with PBUF workflows:

- Use `verify-all` before important analysis runs
- Generate compliance reports for publication submissions
- Export provenance bundles for reproducibility documentation
- Monitor system health with operational summaries

## Performance Considerations

- Large registries: Use filtering options to limit output
- Batch operations: `verify-all` processes datasets efficiently
- Export operations: Large exports may take time, consider using `--output` files
- Audit trails: Use `--limit` to control output size

## Security and Permissions

- The CLI respects file system permissions
- Cleanup operations require explicit confirmation
- Audit trails are append-only and tamper-evident
- Registry modifications are logged automatically