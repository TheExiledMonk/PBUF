#!/usr/bin/env bash
# =======================================================
# PBUF Environment Freezer + Registry Integrator v1.1
# -------------------------------------------------------
# Locks Python environment and writes provenance entry
# into data/registry/environment_<timestamp>.json
# =======================================================

set -e

echo "🔒 Freezing verified PBUF environment..."
FREEZE_DIR="./proofs/environment"
REGISTRY_DIR="./data/registry"
mkdir -p "$FREEZE_DIR" "$REGISTRY_DIR"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
ARCHIVE_FILE="pbuf_environment_${TIMESTAMP}.tar.gz"

# --------------------------------------------
# 1. Capture system + Python metadata
# --------------------------------------------
echo "📋 Capturing environment info..."
{
  echo "# PBUF Environment Snapshot"
  echo "# Date: $TIMESTAMP"
  echo "# Git Commit: $COMMIT_HASH"
  echo "# User: $(whoami)"
  echo "# Host: $(hostname)"
  echo "# Python: $(python3 -V 2>&1)"
  echo "# Platform: $(uname -a)"
  echo "# GCC: $(gcc --version | head -n 1 2>/dev/null || echo 'N/A')"
  echo "# Virtual Env: ${VIRTUAL_ENV:-'(none)'}"
  echo ""
} > "$FREEZE_DIR/ENVIRONMENT_INFO.txt"

# --------------------------------------------
# 2. Freeze dependencies
# --------------------------------------------
echo "📦 Exporting requirements.txt..."
pip freeze | sort > "$FREEZE_DIR/requirements.txt"

if ! pip show pip-tools >/dev/null 2>&1; then
  echo "Installing pip-tools for lockfile generation..."
  pip install -q pip-tools
fi

echo "🔐 Generating requirements.lock (with hashes)..."
pip-compile --generate-hashes \
  --output-file "$FREEZE_DIR/requirements.lock" \
  "$FREEZE_DIR/requirements.txt" >/dev/null 2>&1 || echo "⚠️ pip-compile failed, skipping lockfile."

REQ_HASH=$(sha256sum "$FREEZE_DIR/requirements.txt" | awk '{print $1}')
LOCK_HASH=$(sha256sum "$FREEZE_DIR/requirements.lock" 2>/dev/null | awk '{print $1}' || echo "none")

echo "Requirements SHA256: $REQ_HASH" >> "$FREEZE_DIR/ENVIRONMENT_INFO.txt"
echo "Lockfile SHA256:     $LOCK_HASH" >> "$FREEZE_DIR/ENVIRONMENT_INFO.txt"

# --------------------------------------------
# 3. Archive environment snapshot
# --------------------------------------------
echo "📦 Creating archive..."
tar -czf "$FREEZE_DIR/$ARCHIVE_FILE" -C "$FREEZE_DIR" requirements.txt requirements.lock ENVIRONMENT_INFO.txt

ARCHIVE_HASH=$(sha256sum "$FREEZE_DIR/$ARCHIVE_FILE" | awk '{print $1}')

# --------------------------------------------
# 4. Write registry provenance entry
# --------------------------------------------
REGISTRY_FILE="$REGISTRY_DIR/environment_${TIMESTAMP}.json"

echo "🧾 Writing provenance registry entry..."
cat > "$REGISTRY_FILE" <<EOF
{
  "registry_type": "environment",
  "registry_version": "1.0",
  "created_at": "$TIMESTAMP",
  "git_commit": "$COMMIT_HASH",
  "archive": {
    "file": "$(basename "$ARCHIVE_FILE")",
    "path": "proofs/environment/$(basename "$ARCHIVE_FILE")",
    "sha256": "$ARCHIVE_HASH"
  },
  "requirements": {
    "file": "proofs/environment/requirements.txt",
    "sha256": "$REQ_HASH"
  },
  "lockfile": {


