"""The serve smoke test has to work before it costs a GPU session.

It cannot be run against vLLM here: there is no GPU, and vLLM cannot even share
an environment with appworld. So it is run against a stub that speaks the same
HTTP API, over real sockets, and each stub behaviour is one the real server can
show: answering only after a long startup, honouring the thinking flag or not,
or never coming up at all.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SMOKE_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "serve_smoke.py"


def load_smoke():
    spec = importlib.util.spec_from_file_location("serve_smoke", SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubServer:
    """An OpenAI-compatible endpoint with the behaviours worth testing."""

    def __init__(self, *, honours_thinking_flag: bool = True):
        self.honours_thinking_flag = honours_thinking_flag
        self.payloads: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, {})
                elif self.path == "/version":
                    self._send(200, {"version": "stub"})
                else:
                    self._send(404, {})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                stub.payloads.append(payload)
                kwargs = payload.get("chat_template_kwargs") or {}
                thinking = not (stub.honours_thinking_flag
                                and kwargs.get("enable_thinking") is False)
                content = ("<think>let me see</think>ready" if thinking else "ready")
                tokens = payload["max_tokens"] if payload.get("ignore_eos") else 2
                self._send(200, {
                    "choices": [{"message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": tokens},
                })

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def free_port_with_nothing_listening() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run(smoke, tmp_path, base_url, **kwargs):
    code = smoke.main("Qwen/Qwen3-8B-AWQ", base_url=base_url, out_dir=tmp_path,
                      n_concurrent=4, max_tokens=32, poll_s=0.05, **kwargs)
    return code, json.loads((tmp_path / "serve_smoke.json").read_text())


def test_a_healthy_server_that_honours_the_flag_passes(tmp_path):
    smoke = load_smoke()
    with StubServer() as server:
        code, report = run(smoke, tmp_path, server.base_url)

    assert code == 0
    assert report["ok"] is True
    assert report["server_version"] == "stub"
    assert report["thinking_off"]["has_think_block"] is False


def test_throughput_counts_every_generated_token(tmp_path):
    """4 requests of 32 forced tokens each is 128 tokens, whatever the timing."""
    smoke = load_smoke()
    with StubServer() as server:
        _, report = run(smoke, tmp_path, server.base_url)

    measured = report["throughput"]
    assert measured["completion_tokens"] == 4 * 32
    assert measured["tok_per_s"] > 0


def test_every_request_asks_for_thinking_off(tmp_path):
    """The audit's main condition. Qwen3's template defaults to thinking on, so a
    request that forgot to say otherwise would quietly change the condition."""
    smoke = load_smoke()
    with StubServer() as server:
        run(smoke, tmp_path, server.base_url)

    assert server.payloads
    assert all(p["chat_template_kwargs"] == {"enable_thinking": False}
               for p in server.payloads)


def test_a_server_that_ignores_the_flag_fails_the_smoke_test(tmp_path):
    """If thinking cannot be switched off, every episode spends its tokens on
    reasoning text and runs a condition the audit did not specify."""
    smoke = load_smoke()
    with StubServer(honours_thinking_flag=False) as server:
        code, report = run(smoke, tmp_path, server.base_url)

    assert code == 1
    assert report["ok"] is False
    assert report["thinking_off"]["has_think_block"] is True


def test_no_server_is_recorded_not_raised(tmp_path):
    """The likeliest first failure on a T4 is a server that never comes up. The
    record still has to reach disk, with the reason in it."""
    smoke = load_smoke()
    base_url = f"http://127.0.0.1:{free_port_with_nothing_listening()}"

    code, report = run(smoke, tmp_path, base_url, wait_s=0.3)

    assert code == 1
    assert report["ok"] is False
    assert "TimeoutError" in report["error"]


@pytest.mark.parametrize("missing", ["choices", "empty"])
def test_a_reply_without_choices_is_a_failure(tmp_path, missing, monkeypatch):
    smoke = load_smoke()
    body = {} if missing == "choices" else {"choices": []}
    monkeypatch.setattr(smoke, "_post", lambda url, payload, timeout: body)
    with StubServer() as server:
        code, report = run(smoke, tmp_path, server.base_url)

    assert code == 1
    assert "no choices" in report["error"]


# --- watching the server's log while waiting --------------------------------
# the server downloads the model and compiles before /health answers, and it does
# so in a background process whose output goes to a log file. the first version
# showed that log only after waiting up to 20 minutes, so a server downloading 6
# GB and a server that crashed in its first second looked identical for 20
# minutes. on kaggle that read as "it's not downloading".


def test_progress_from_the_server_log_is_shown_while_waiting(tmp_path, capsys):
    """Download progress arrives as carriage-return updates on one line; the
    latest one is what matters."""
    smoke = load_smoke()
    log = tmp_path / "vllm.log"
    log.write_text("INFO starting\nmodel.safetensors:  10%|#  \rmodel.safetensors:  40%|####  ")
    base_url = f"http://127.0.0.1:{free_port_with_nothing_listening()}"

    smoke.main("Qwen/Qwen3-8B-AWQ", base_url=base_url, out_dir=tmp_path,
               wait_s=0.3, poll_s=0.05, log_path=log, report_every=0)

    shown = capsys.readouterr().out
    assert "40%" in shown


def test_a_server_that_exited_stops_the_wait_at_once(tmp_path):
    """The readme's launch line writes this after vllm exits. A server that has
    exited is never coming up, and waiting out the full timeout after that wastes
    GPU quota and hides the cause."""
    smoke = load_smoke()
    log = tmp_path / "vllm.log"
    log.write_text(
        "INFO loading weights\n"
        "Traceback (most recent call last):\n"
        '  File "worker.py", line 1, in <module>\n'
        "RuntimeError: CUDA error: no kernel image is available for execution\n"
        "server exited with status 1\n"
    )
    base_url = f"http://127.0.0.1:{free_port_with_nothing_listening()}"

    started = time.monotonic()
    code = smoke.main("Qwen/Qwen3-8B-AWQ", base_url=base_url, out_dir=tmp_path,
                      wait_s=60, poll_s=0.05, log_path=log)
    elapsed = time.monotonic() - started

    report = json.loads((tmp_path / "serve_smoke.json").read_text())
    assert code == 1
    assert elapsed < 10
    assert "exited during startup" in report["error"]
    assert "no kernel image" in report["error"]


def test_a_missing_binary_stops_the_wait_at_once(tmp_path):
    """What the launch line leaves when the install failed: the shell cannot find
    vllm, and the exit line follows."""
    smoke = load_smoke()
    log = tmp_path / "vllm.log"
    log.write_text("sh: 1: /tmp/vllm-env/bin/vllm: not found\n"
                   "server exited with status 127\n")
    base_url = f"http://127.0.0.1:{free_port_with_nothing_listening()}"

    code = smoke.main("Qwen/Qwen3-8B-AWQ", base_url=base_url, out_dir=tmp_path,
                      wait_s=60, poll_s=0.05, log_path=log)

    report = json.loads((tmp_path / "serve_smoke.json").read_text())
    assert code == 1
    assert "vllm: not found" in report["error"]


def test_nohup_failing_to_start_the_command_stops_the_wait_at_once(tmp_path):
    """A launch without the exit line, run through nohup: nohup writes this when
    the binary is missing, and then nothing else ever runs."""
    smoke = load_smoke()
    log = tmp_path / "vllm.log"
    log.write_text("nohup: failed to run command '/tmp/vllm-env/bin/vllm': "
                   "No such file or directory\n")
    base_url = f"http://127.0.0.1:{free_port_with_nothing_listening()}"

    code = smoke.main("Qwen/Qwen3-8B-AWQ", base_url=base_url, out_dir=tmp_path,
                      wait_s=60, poll_s=0.05, log_path=log)

    report = json.loads((tmp_path / "serve_smoke.json").read_text())
    assert code == 1
    assert "failed to run command" in report["error"]


def test_a_traceback_the_server_survives_does_not_stop_the_wait(tmp_path):
    """A logged traceback is not a dead server: libraries log exceptions they
    recover from. Stopping on one would report a crash while the server was still
    loading, and every re-run would stop on the same line until the log was
    replaced. Only the exit line is proof."""
    smoke = load_smoke()
    log = tmp_path / "vllm.log"
    log.write_text(
        "Traceback (most recent call last):\n"
        "ImportError: optional kernel unavailable, falling back\n"
        "INFO loading weights\n"
    )
    base_url = f"http://127.0.0.1:{free_port_with_nothing_listening()}"

    code = smoke.main("Qwen/Qwen3-8B-AWQ", base_url=base_url, out_dir=tmp_path,
                      wait_s=0.3, poll_s=0.05, log_path=log)

    report = json.loads((tmp_path / "serve_smoke.json").read_text())
    assert code == 1
    assert "TimeoutError" in report["error"]


def test_a_log_that_does_not_exist_yet_is_not_an_error(tmp_path):
    """The log appears a moment after the server starts. Until then, keep waiting."""
    smoke = load_smoke()
    base_url = f"http://127.0.0.1:{free_port_with_nothing_listening()}"

    code = smoke.main("Qwen/Qwen3-8B-AWQ", base_url=base_url, out_dir=tmp_path,
                      wait_s=0.3, poll_s=0.05, log_path=tmp_path / "absent.log")

    report = json.loads((tmp_path / "serve_smoke.json").read_text())
    assert code == 1
    assert "TimeoutError" in report["error"]
