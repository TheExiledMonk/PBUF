"""System status plugin that renders the dashboard for controller health."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).with_name("manifest.json")
MANIFEST = json.loads(MANIFEST_PATH.read_text())

SHARED_STYLE = """
body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#0f1217; color:#f4f6fb; margin:0; }
.page { max-width:1100px; margin:0 auto; padding:2rem; }
.layout { display:grid; grid-template-columns:minmax(0,2fr) minmax(280px,1fr); gap:1.5rem; }
.content-column { display:flex; flex-direction:column; }
.sidebar .panel { margin-top:0; }
.config-panel table { font-size:0.85rem; }
.config-panel td, .config-panel th { padding:0.4rem 0.3rem; }
.config-summary { margin-top:0.6rem; font-size:0.9rem; color:#8fa3c1; }
.config-actions { display:flex; gap:0.4rem; margin-top:0.8rem; }
.config-actions button {
  background:#1d2431;
  border:1px solid rgba(255,255,255,0.1);
  color:#f4f6fb;
  padding:0.5rem 0.8rem;
  border-radius:4px;
  cursor:pointer;
}
.config-action-button {
  background:#0d8b6a;
  color:#fff;
  border:none;
  padding:0.35rem 0.8rem;
  border-radius:4px;
  cursor:pointer;
  transition: opacity 0.15s;
}
.config-action-button:disabled {
  opacity:0.6;
  cursor:not-allowed;
}
h1 { margin-bottom:0.4rem; }
h2 { margin:1rem 0 0.2rem; }
.panel { background:#161a24; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:1rem; margin-top:1rem; box-shadow:0 6px 25px rgba(0,0,0,0.4); }
.panel .grid { display:grid; grid-template-columns:auto auto auto; gap:1rem; margin-top:1rem; }
.metric { background:#1d2431; padding:1rem; border-radius:6px; }
.metric span { display:block; font-size:1.4rem; font-weight:600; }
table { width:100%; border-collapse:collapse; margin-top:0.5rem; }
th, td { padding:0.6rem 0.4rem; border-bottom:1px solid rgba(255,255,255,0.08); font-size:0.95rem; }
th { text-align:left; font-size:0.85rem; letter-spacing:0.05em; text-transform:uppercase; color:#a8b2c6; }
pre { background:#0f1217; border:1px dashed rgba(255,255,255,0.1); padding:0.4rem; border-radius:4px; margin:0; font-size:0.9rem; overflow:auto; }
.logs { max-height:200px; overflow:auto; }
.muted { color:#8fa3c1; }
.status-badge { padding:0.1rem 0.6rem; border-radius:999px; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; background:#222b3b; }
.status-badge.ready { background:#0d8b6a; }
.status-badge.warning { background:#f05d5d; }
.transport-details { font-size:0.7rem; margin-top:0.3rem; color:#a8b2c6; }
.alerts { margin-top:0.6rem; display:flex; flex-direction:column; gap:0.5rem; }
.alert-row { border-radius:6px; border:1px solid rgba(255,255,255,0.08); padding:0.8rem; background:#13161d; }
.alert-row.warning { border-color:#d19f3b; }
.alert-row.critical { border-color:#f05d5d; background:#2c151a; }
"""

DASHBOARD_STYLE = """
.dashboard-shell { min-height:100vh; display:flex; background:#0f1217; }
.nav-shell { width:260px; padding:2rem; border-right:1px solid rgba(255,255,255,0.05); background:#11141d; display:flex; flex-direction:column; gap:1.5rem; }
.nav-brand h1 { margin:0; font-size:1.4rem; }
.dashboard-nav { display:flex; flex-direction:column; gap:0.4rem; }
.nav-button { background:#1d2431; color:#f4f6fb; border:1px solid rgba(255,255,255,0.1); border-radius:6px; padding:0.65rem 0.8rem; text-align:left; font-size:0.95rem; cursor:pointer; transition: background 0.2s, border-color 0.2s; }
.nav-button:hover { border-color:#1d8f76; }
.nav-button.active { background:#0d8b6a; border-color:#0d8b6a; }
.dashboard-main { flex:1; padding:2rem; display:flex; flex-direction:column; }
.main-header { margin-bottom:1.5rem; }
.content-panel { display:none; }
.content-panel.active { display:block; }
.sidebar-note { font-size:0.85rem; color:#a8b2c6; background:#141926; border:1px solid rgba(255,255,255,0.05); border-radius:6px; padding:0.8rem; }
"""

STATUS_SECTIONS = """
<section class="panel">
  <h2>Controller</h2>
  <div class="grid">
    <div class="metric">
      <div class="muted">Ready</div>
      <span id="controller-ready">—</span>
    </div>
    <div class="metric">
      <div class="muted">Uptime</div>
      <span id="controller-uptime">—</span>
    </div>
    <div class="metric">
      <div class="muted">Last refreshed</div>
      <span id="controller-updated">—</span>
    </div>
  </div>
</section>
<section class="panel">
  <h2>Jobs</h2>
  <div class="grid">
    <div class="metric">
      <div class="muted">Queued</div>
      <span id="jobs-queued">0</span>
    </div>
    <div class="metric">
      <div class="muted">Running</div>
      <span id="jobs-running">0</span>
    </div>
    <div class="metric">
      <div class="muted">Completed</div>
      <span id="jobs-completed">0</span>
    </div>
    <div class="metric">
      <div class="muted">Failed</div>
      <span id="jobs-failed">0</span>
    </div>
  </div>
</section>
<section class="panel">
  <h2>Basin Walker Jobs</h2>
  <div id="job-metadata">
    <p class="muted">Loading…</p>
  </div>
</section>
<section class="panel">
  <h2>Workers</h2>
  <p class="muted">Total cores: <span id="workers-cores">0</span> • Allocated slots: <span id="workers-slots">0</span> • Active slices: <span id="slices-active">0</span></p>
  <table>
    <thead>
      <tr>
        <th>Worker</th>
        <th>Cores</th>
        <th>Slots</th>
        <th>Active Slices</th>
        <th>State</th>
        <th>Last Seen</th>
        <th>Retries</th>
        <th>Error</th>
        <th>Datasets</th>
      </tr>
    </thead>
    <tbody id="worker-table">
      <tr><td colspan="9">Loading…</td></tr>
    </tbody>
  </table>
</section>
<section class="panel">
  <h2>Worker Alerts</h2>
  <div id="worker-alerts" class="alerts">
    <p class="muted">Loading…</p>
  </div>
</section>
"""

CONFIG_PANEL_BODY = """
<p class="muted">Configs under <code>config/science_runs</code>.</p>
<div id="config-summary" class="config-summary">Loading configs…</div>
<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Engine</th>
      <th>Run type</th>
      <th>Dataset</th>
      <th>Fits</th>
      <th>Action</th>
    </tr>
  </thead>
  <tbody id="config-table">
    <tr><td colspan="6">Loading configs…</td></tr>
  </tbody>
</table>
<div class="config-actions">
  <button id="config-refresh" type="button">Refresh</button>
  <button id="config-create" type="button">Create new</button>
</div>
"""

LOG_PANEL_BODY = """
<div class="logs">
  <pre id="log-history">Loading…</pre>
</div>
"""

STATUS_SCRIPT = """
(() => {
  const statusEndpoint = '/system/status';
  const configEndpoint = '/controller/configs';
  const jsonHeaders = { Accept: 'application/json' };
  const readyEl = document.getElementById('controller-ready');
  const uptimeEl = document.getElementById('controller-uptime');
  const updatedEl = document.getElementById('controller-updated');
  const jobs = {
    queued: document.getElementById('jobs-queued'),
    running: document.getElementById('jobs-running'),
    completed: document.getElementById('jobs-completed'),
    failed: document.getElementById('jobs-failed'),
  };
  const workersCores = document.getElementById('workers-cores');
  const workersSlots = document.getElementById('workers-slots');
  const slicesActive = document.getElementById('slices-active');
  const workerTable = document.getElementById('worker-table');
  const logHistory = document.getElementById('log-history');
  const workerAlerts = document.getElementById('worker-alerts');
  const configSummary = document.getElementById('config-summary');
  const configTable = document.getElementById('config-table');
  const jobMetadata = document.getElementById('job-metadata');
  const configRefreshBtn = document.getElementById('config-refresh');
  const configCreateBtn = document.getElementById('config-create');
  const panelButtons = document.querySelectorAll('[data-panel-target]');
  const panelContainers = document.querySelectorAll('[data-panel]');
  const sanitizer = document.createElement('span');

  function escapeHTML(value) {
    sanitizer.textContent = value ?? '';
    return sanitizer.innerHTML;
  }

  function formatDatasets(entry) {
    const data = entry || {};
    const keys = Object.keys(data);
    if (!keys.length) {
      return '<span class="muted">empty</span>';
    }
    return keys
      .map(key => `<span>${escapeHTML(key)}: ${escapeHTML(data[key])}</span>`)
      .join('<br>');
  }

  function formatTimestamp(value) {
    if (!value) {
      return '—';
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return escapeHTML(value);
    }
    return escapeHTML(parsed.toLocaleString());
  }

  function _createCell(content) {
    const cell = document.createElement('td');
    if (content instanceof Node) {
      cell.appendChild(content);
    } else {
      cell.innerHTML = content ?? '';
    }
    return cell;
  }

  function formatNumber(value) {
    return typeof value === 'number' ? value : '—';
  }

  function summarizeMetadata(metadata) {
    if (!metadata || typeof metadata !== 'object') {
      return '';
    }
    const parts = [];
    if (metadata.package_type) {
      const label = metadata.package_id
        ? `${metadata.package_type} (${metadata.package_id})`
        : metadata.package_type;
      parts.push(label);
    }
    if (metadata.seed_start !== undefined && metadata.seed_end !== undefined) {
      parts.push(`seeds ${metadata.seed_start}–${metadata.seed_end}`);
    }
    const jackknife = metadata.jackknife_draw || metadata.jackknife;
    if (jackknife) {
      parts.push(`jackknife draw ${jackknife.index != null ? jackknife.index + 1 : 'unknown'}`);
    }
    if (metadata.prediction_modules) {
      const modules = Array.isArray(metadata.prediction_modules)
        ? metadata.prediction_modules.join(', ')
        : String(metadata.prediction_modules);
      parts.push(`predictions: ${modules}`);
    }
    if (metadata.engine) {
      parts.push(`engine: ${metadata.engine}`);
    }
    return parts.join(' • ');
  }

  function renderJobMetadata(jobs) {
    if (!jobMetadata) {
      return;
    }
    if (!jobs?.length) {
      jobMetadata.innerHTML = '<p class="muted">No basin walker jobs recorded</p>';
      return;
    }
    jobMetadata.innerHTML = jobs
      .map(job => {
        const badgeClass = job.status === 'running' ? 'ready' : 'warning';
        const jobMeta = job.metadata || {};
        const summary = summarizeMetadata(jobMeta);
        const timestamp = formatTimestamp(job.created_at);
        return `
          <div class="job-entry">
            <strong>${escapeHTML(job.package_id || job.execution_id)}</strong>
            <span class="status-badge ${badgeClass}" style="margin-left:0.5rem;">${escapeHTML(job.status)}</span>
            <div class="muted" style="margin-top:0.2rem;">${escapeHTML(timestamp)}${summary ? ' • ' + escapeHTML(summary) : ''}</div>
          </div>
        `;
      })
      .join('');
  }

  function renderWorkers(rows) {
    if (!workerTable) {
      return;
    }
    workerTable.innerHTML = '';
    if (!rows?.length) {
      workerTable.innerHTML = '<tr><td colspan="9" class="muted">No workers connected</td></tr>';
      return;
    }
    rows.forEach(row => {
      const stateLabel = row.state || 'unknown';
      const statusClass = stateLabel === 'connected' ? 'ready' : 'warning';
      const lastSeen = formatTimestamp(row.last_seen);
      const retries = typeof row.retry_count === 'number' ? row.retry_count : '—';
      const endpoint =
        row.transport_status && row.transport_status.controller_endpoint
          ? escapeHTML(row.transport_status.controller_endpoint)
          : 'endpoint unknown';
      const errorBlock = document.createElement('div');
      errorBlock.textContent = row.last_error || 'No recent errors';
      if (!row.last_error) {
        errorBlock.classList.add('muted');
      }
      const statusWrapper = document.createElement('div');
      statusWrapper.innerHTML = `<span class="status-badge ${statusClass}">${escapeHTML(stateLabel)}</span>`;
      const endpointInfo = document.createElement('div');
      endpointInfo.className = 'transport-details';
      endpointInfo.textContent = endpoint;
      statusWrapper.appendChild(endpointInfo);
      const cells = [
        _createCell(escapeHTML(row.worker_id || '—')),
        _createCell(escapeHTML(formatNumber(row.cores))),
        _createCell(escapeHTML(formatNumber(row.allocated_slots))),
        _createCell(escapeHTML(formatNumber(row.active_slices))),
        (() => {
          const cell = document.createElement('td');
          cell.appendChild(statusWrapper);
          return cell;
        })(),
        _createCell(lastSeen),
        _createCell(escapeHTML(retries)),
        (() => {
          const cell = document.createElement('td');
          cell.appendChild(errorBlock);
          return cell;
        })(),
        _createCell(formatDatasets(row.datasets)),
      ];
      const tr = document.createElement('tr');
      cells.forEach(cell => tr.appendChild(cell));
      workerTable.appendChild(tr);
    });
  }

  function renderAlerts(alerts) {
    if (!workerAlerts) {
      return;
    }
    if (!alerts?.length) {
      workerAlerts.innerHTML = '<p class="muted">No alerts.</p>';
      return;
    }
    workerAlerts.innerHTML = '';
    alerts.forEach(alert => {
      const wrapper = document.createElement('div');
      const severity = alert.severity === 'critical' ? 'critical' : 'warning';
      wrapper.className = `alert-row ${severity}`;
      const strong = document.createElement('strong');
      strong.textContent = alert.worker_id || 'worker';
      wrapper.appendChild(strong);
      const message = document.createElement('span');
      message.textContent = alert.message || 'Alert triggered';
      wrapper.appendChild(document.createTextNode(': '));
      wrapper.appendChild(message);
      const timestamp = document.createElement('div');
      timestamp.className = 'muted';
      timestamp.textContent = formatTimestamp(alert.timestamp);
      wrapper.appendChild(timestamp);
      workerAlerts.appendChild(wrapper);
    });
  }

  function renderLogs(logs) {
    if (!logHistory) {
      return;
    }
    if (!logs?.length) {
      logHistory.textContent = 'No logs yet.';
      return;
    }
    logHistory.innerHTML = logs.map(line => `<div>${escapeHTML(line)}</div>`).join('');
  }

  async function startConfigRun(configName, button) {
    if (!configName) {
      return;
    }
    const headers = {
      ...jsonHeaders,
      'Content-Type': 'application/json',
    };
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = 'Starting…';
    try {
      const res = await fetch(
        `${configEndpoint}/${encodeURIComponent(configName)}/run`,
        {
          method: 'POST',
          headers,
          body: JSON.stringify({}),
        },
      );
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(payload.error || res.statusText || 'Failed to start job');
      }
      window.alert(
        `Job started from ${configName}: ${payload.execution_id} (${payload.run_id})`,
      );
    } catch (error) {
      window.alert(`Failed to start config ${configName}: ${error.message}`);
    } finally {
      button.disabled = false;
      button.textContent = originalText;
    }
  }

  function renderConfigs(configs) {
    if (!configTable) {
      return;
    }
    configTable.innerHTML = '';
    if (!configs?.length) {
      configTable.innerHTML = '<tr><td colspan="5" class="muted">No configs found.</td></tr>';
      if (configSummary) {
        configSummary.textContent = '0 configs';
      }
      return;
    }
    configs.forEach(cfg => {
      const tr = document.createElement('tr');
      const fits = Array.isArray(cfg.fits) ? cfg.fits.join(', ') : '';
      const runType = cfg.run_type || cfg.engine_settings?.run_type || 'default';
      const dataset = cfg.dataset_id || cfg.engine_settings?.dataset_id || '—';
      [
        cfg.name || '—',
        cfg.engine || '—',
        runType,
        dataset,
        fits || '—',
      ].forEach(value => {
        const td = document.createElement('td');
        td.textContent = value;
        tr.appendChild(td);
      });
      const actionCell = document.createElement('td');
      const actionButton = document.createElement('button');
      actionButton.type = 'button';
      actionButton.className = 'config-action-button';
      actionButton.textContent = 'Start run';
      actionButton.addEventListener('click', event => {
        event.stopPropagation();
        startConfigRun(cfg.name, actionButton);
      });
      actionCell.appendChild(actionButton);
      tr.appendChild(actionCell);
      tr.addEventListener('click', () => {
        window.alert('View config via cosmos_cli.py config view ' + (cfg.name || 'unknown'));
      });
      configTable.appendChild(tr);
    });
    if (configSummary) {
      configSummary.textContent = `${configs.length} config${configs.length === 1 ? '' : 's'}`;
    }
  }

  function setupPanelNavigation() {
    if (!panelButtons.length || !panelContainers.length) {
      return;
    }
    const activatePanel = target => {
      panelContainers.forEach(panel => {
        panel.classList.toggle('active', panel.dataset.panel === target);
      });
      panelButtons.forEach(button => {
        button.classList.toggle('active', button.dataset.panelTarget === target);
      });
    };
    panelButtons.forEach(button => {
      button.addEventListener('click', () => {
        activatePanel(button.dataset.panelTarget);
      });
    });
    activatePanel(panelButtons[0].dataset.panelTarget);
  }

  async function refresh() {
    try {
      const res = await fetch(statusEndpoint, { headers: jsonHeaders });
      if (!res.ok) {
        throw new Error(res.statusText || 'Failed to load status');
      }
      const payload = await res.json();
      if (readyEl) {
        readyEl.textContent = payload.controller_ready ? 'Yes' : 'No';
      }
      if (uptimeEl) {
        uptimeEl.textContent = payload.controller_uptime || '—';
      }
      if (updatedEl) {
        updatedEl.textContent = payload.last_updated || '—';
      }
      if (jobs.queued) {
        jobs.queued.textContent = payload.jobs?.queued ?? 0;
      }
      if (jobs.running) {
        jobs.running.textContent = payload.jobs?.running ?? 0;
      }
      if (jobs.completed) {
        jobs.completed.textContent = payload.jobs?.completed ?? 0;
      }
      if (jobs.failed) {
        jobs.failed.textContent = payload.jobs?.failed ?? 0;
      }
      if (workersCores) {
        workersCores.textContent = payload.worker_summary?.total_cores ?? 0;
      }
      if (workersSlots) {
        workersSlots.textContent = payload.worker_summary?.allocated_slots ?? 0;
      }
      if (slicesActive) {
        slicesActive.textContent = payload.slices_active ?? 0;
      }
      renderWorkers(payload.workers || []);
      renderAlerts(payload.worker_alerts || []);
      renderLogs(payload.last_logs || []);
      renderJobMetadata(payload.recent_jobs || []);
    } catch (error) {
      console.error('system-status refresh failed', error);
      if (readyEl) {
        readyEl.textContent = 'Error';
      }
      if (logHistory) {
        logHistory.textContent = `Status fetch failed: ${error.message}`;
      }
      if (workerTable) {
        workerTable.innerHTML = '<tr><td colspan="9" class="muted">Unable to load workers</td></tr>';
      }
      if (workerAlerts) {
        workerAlerts.innerHTML = '<p class="muted">Unable to load alerts.</p>';
      }
    }
  }

  async function refreshConfigs() {
    if (!configSummary) {
      return;
    }
    configSummary.textContent = 'Refreshing configs…';
    try {
      const res = await fetch(configEndpoint, { headers: jsonHeaders });
      if (!res.ok) {
        throw new Error(res.statusText || 'Failed to load configs');
      }
      const payload = await res.json();
      renderConfigs(payload.configs || []);
    } catch (error) {
      console.error('config refresh failed', error);
      configSummary.textContent = 'Failed to load configs';
      if (configTable) {
        configTable.innerHTML = '<tr><td colspan="4" class="muted">Unable to load configs</td></tr>';
      }
    }
  }

  setupPanelNavigation();
  refresh();
  refreshConfigs();
  if (configRefreshBtn) {
    configRefreshBtn.addEventListener('click', refreshConfigs);
  }
  if (configCreateBtn) {
    configCreateBtn.addEventListener('click', () => {
      window.alert('Use `cosmos_cli.py config new <name>` to create a new science-run config.');
    });
  }
  setInterval(refresh, 5000);
})();
"""

SCRIPT_BLOCK = f"<script>{STATUS_SCRIPT}</script>"


def manifest() -> dict[str, Any]:
    return MANIFEST


def render_html() -> str:
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Cosmos Control — System Status</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <style>{SHARED_STYLE}</style>
</head>
<body>
  <main class=\"page layout\">
    <div class=\"content-column\">
      <header>
        <h1>System Status</h1>
        <p class=\"muted\">Controller health, workers, and synchronization overview.</p>
      </header>
      {STATUS_SECTIONS}
      <section class=\"panel\">
        <h2>Recent Logs</h2>
        {LOG_PANEL_BODY}
      </section>
    </div>
    <aside class=\"sidebar\">
      <section class=\"panel config-panel\">
        <h2>Science Run Configs</h2>
        {CONFIG_PANEL_BODY}
      </section>
    </aside>
  </main>
  {SCRIPT_BLOCK}
</body>
</html>"""


def render_dashboard_html() -> str:
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Cosmos Control — Dashboard</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <style>{SHARED_STYLE}{DASHBOARD_STYLE}</style>
</head>
<body>
  <div class=\"dashboard-shell\">
    <aside class=\"nav-shell\">
      <div class=\"nav-brand\">
        <h1>Cosmos Control</h1>
        <p class=\"muted\">Monitor controller & worker health.</p>
      </div>
      <nav class=\"dashboard-nav\">
        <button type=\"button\" class=\"nav-button active\" data-panel-target=\"panel-status\">System Status</button>
        <button type=\"button\" class=\"nav-button\" data-panel-target=\"panel-configs\">Configs</button>
        <button type=\"button\" class=\"nav-button\" data-panel-target=\"panel-logs\">Logs</button>
      </nav>
      <div class=\"sidebar-note\">
        <p>Controller writes run data under <code>data/science_runs/</code>.</p>
        <p>Workers compute via the HPC transport and never touch controller folders.</p>
      </div>
    </aside>
    <div class=\"dashboard-main\">
      <header class=\"main-header\">
        <h1>Operational Dashboard</h1>
        <p class=\"muted\">Status metrics, config inventories, and log history in one place.</p>
      </header>
      <section id=\"panel-status\" data-panel=\"panel-status\" class=\"content-panel active\">
        {STATUS_SECTIONS}
      </section>
      <section id=\"panel-configs\" data-panel=\"panel-configs\" class=\"content-panel\">
        <div class=\"panel config-panel\">
          <h2>Science Run Configs</h2>
          {CONFIG_PANEL_BODY}
        </div>
      </section>
      <section id=\"panel-logs\" data-panel=\"panel-logs\" class=\"content-panel\">
        <div class=\"panel\">
          <h2>Recent Logs</h2>
          {LOG_PANEL_BODY}
        </div>
      </section>
    </div>
  </div>
  {SCRIPT_BLOCK}
</body>
</html>"""
