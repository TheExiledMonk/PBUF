"""Simple HTTP API exposing controller state for UIs."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import urlparse

from hpc_comms.core.messages import deserialize_message

from .controller import Controller
from .hpc_bridge import HPCControllerBridge
from .plugins.system_status import plugin as system_status_plugin

CONFIG_RUNS_DIR = Path("config/science_runs")


def _ensure_config_dir() -> Path:
    CONFIG_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_RUNS_DIR


def _config_metadata() -> list[dict[str, Any]]:
    _ensure_config_dir()
    configs: list[dict[str, Any]] = []
    for path in sorted(CONFIG_RUNS_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        configs.append(
            {
                "name": path.name,
                "engine": payload.get("engine"),
                "engine_settings": payload.get("engine_settings") or {},
                "fits": payload.get("fits") or [],
                "description": payload.get("description", ""),
            }
        )
    return configs


class ControllerHTTPHandler(BaseHTTPRequestHandler):
    controller: Controller | None = None
    bridge: HPCControllerBridge | None = None

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        payload = self.rfile.read(length)
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception:
            return {}

    def _send_json(self, data: object, status: int = 200) -> None:
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        controller = self.controller
        if controller is None:
            self.send_error(500, "Controller not configured")
            return

        path = urlparse(self.path).path.rstrip("/")
        if path == "/controller/jobs":
            payload = self._read_json()
            try:
                job = controller.submit_job(
                    config_path=payload.get("config_path"),
                    config_payload=payload.get("config"),
                    slice_count=payload.get("slice_count"),
                    dataset_id=payload.get("dataset_id"),
                )
                self._send_json({"execution_id": job.execution_id, "run_id": job.run_id}, status=201)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        parts = [part for part in path.split("/") if part]
        if (
            len(parts) == 4
            and parts[0] == "controller"
            and parts[1] == "jobs"
            and parts[3] == "cancel"
        ):
            execution_id = parts[2]
            controller.cancel_job(execution_id)
            self._send_json({"status": "canceled"})
            return

            self.send_error(404, "Not found")
            return

        if path == "/api/v1/hpc":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b""
            try:
                message = deserialize_message(body.decode("utf-8"))
            except Exception as exc:
                self.send_error(400, "Invalid HPC payload")
                return
            if self.bridge is None:
                self.send_error(500, "Bridge not configured")
                return
            responses = self.bridge.handle_message(message)
            payload = {"messages": [resp.to_dict() for resp in responses]}
            self._send_json(payload)
            return

        parts = [part for part in path.split("/") if part]
        if (
            len(parts) == 4
            and parts[0] == "controller"
            and parts[1] == "configs"
            and parts[3] == "run"
        ):
            config_name = parts[2]
            config_path = (CONFIG_RUNS_DIR / config_name).resolve()
            if not config_path.exists() or config_path.parent != CONFIG_RUNS_DIR.resolve():
                self.send_error(404, "Config not found")
                return
            payload = self._read_json()
            try:
                job = controller.submit_job(
                    config_path=config_path,
                    slice_count=payload.get("slice_count"),
                    dataset_id=payload.get("dataset_id"),
                )
                self._send_json(
                    {"execution_id": job.execution_id, "run_id": job.run_id},
                    status=201,
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

    def do_GET(self) -> None:
        controller = self.controller
        if controller is None:
            self.send_error(500, "Controller not configured")
            return

        path = urlparse(self.path).path.rstrip("/")
        if path == "/controller/jobs":
            self._send_json(controller.list_jobs())
            return

        if path == "/api/v1/hpc":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b""
            try:
                message = deserialize_message(body.decode("utf-8"))
            except Exception as exc:
                self.send_error(400, "Invalid HPC payload")
                return
            if self.bridge is None:
                self.send_error(500, "Bridge not configured")
                return
            responses = self.bridge.handle_message(message)
            payload = {"messages": [resp.to_dict() for resp in responses]}
            self._send_json(payload)
            return

        if path == "/system/status":
            status = controller.system_status()
            accept = self.headers.get("Accept", "").lower()
            wants_json = "application/json" in accept or self.headers.get("X-Requested-With") == "XMLHttpRequest"
            if wants_json:
                self._send_json(status)
                return

            html = system_status_plugin.render_html()
            payload = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if path == "/dashboard":
            html = system_status_plugin.render_dashboard_html()
            payload = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if path == "/controller/configs":
            self._send_json({"configs": _config_metadata()})
            return

        parts = [part for part in path.split("/") if part]
        if len(parts) == 3 and parts[0] == "controller" and parts[1] == "jobs":
            execution_id = parts[2]
            job = controller.get_job(execution_id)
            if job is None:
                self.send_error(404, "Job not found")
            else:
                self._send_json(job)
            return

        if len(parts) == 5 and parts[0] == "controller" and parts[1] == "jobs" and parts[3] == "slices":
            execution_id = parts[2]
            slice_id = parts[4]
            slice_data = controller.get_slice(execution_id, slice_id)
            if slice_data is None:
                self.send_error(404, "Slice not found")
            else:
                self._send_json(slice_data)
            return

        if len(parts) == 4 and parts[0] == "controller" and parts[1] == "jobs" and parts[3] == "logs":
            execution_id = parts[2]
            logs = controller.get_job_logs(execution_id)
            if logs is None:
                self.send_error(404, "Logs not found")
            else:
                self._send_json({"logs": logs})
            return

        if (
            len(parts) == 6
            and parts[0] == "controller"
            and parts[1] == "jobs"
            and parts[3] == "slices"
            and parts[5] == "logs"
        ):
            execution_id = parts[2]
            slice_id = parts[4]
            logs = controller.get_slice_logs(execution_id, slice_id)
            if logs is None:
                self.send_error(404, "Slice logs not found")
            else:
                self._send_json({"logs": logs})
            return

        self.send_error(404, "Not found")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def run_controller_api(controller: Controller, host: str | None = None, port: int = 8080) -> ThreadedHTTPServer:
    """Start a threaded HTTP server that exposes controller REST APIs."""

    handler = ControllerHTTPHandler
    handler.controller = controller
    handler.bridge = HPCControllerBridge(controller)
    address = (host or "0.0.0.0", port)
    server = ThreadedHTTPServer(address, handler)
    return server
