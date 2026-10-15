"""Smoke-test a running vLLM server on the hardware the pilot will use. GPU.

Three things the model choice and the budget rest on, none of them measured yet:

1. Does the server answer at all on this GPU. A T4 is compute capability 7.5;
   vLLM 0.31 still lists it, with attention falling back from FlashAttention to
   Triton and AWQ falling back from Marlin to its unoptimised kernels.
2. Does thinking stay off. The audit's main condition is thinking off, and
   Qwen3's chat template defaults to on, so the request has to say so.
3. What aggregate decode throughput the server sustains under concurrent load.
   The budget in tasks/todo.md assumed 200 to 500 tok/s and has never been
   checked against this hardware.

Standard library only, on purpose. vLLM needs pydantic 2 and appworld pins
pydantic 1, so the server runs in its own environment and this runs from the
notebook kernel, talking to it over HTTP the way the pilot will.

    !python notebooks/serve_smoke.py --model Qwen/Qwen3-8B-AWQ
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
THINK_TAG = "<think>"


def _get(url: str, timeout: float) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", "replace")


def _post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_health(base_url: str, wait_s: float, poll_s: float = 5.0) -> float:
    """Seconds until /health answers 200. Startup on a T4 compiles and captures
    graphs, which takes minutes, so this polls rather than failing on the first
    refused connection."""
    started = time.monotonic()
    while True:
        try:
            status, _ = _get(f"{base_url}/health", timeout=10)
            if status == 200:
                return round(time.monotonic() - started, 1)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        if time.monotonic() - started >= wait_s:
            raise TimeoutError(f"no healthy server at {base_url} after {wait_s:.0f} s")
        time.sleep(poll_s)


def chat(base_url: str, model: str, prompt: str, max_tokens: int,
         *, ignore_eos: bool = False, timeout: float = 600) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # the audit's main condition. qwen3's template defaults to thinking on,
        # and a model that is not qwen3 ignores an unknown template variable
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    started = time.monotonic()
    body = _post(f"{base_url}/v1/chat/completions", payload, timeout)
    seconds = time.monotonic() - started
    if not body.get("choices"):
        raise RuntimeError(f"server returned no choices: {body}")
    content = (body["choices"][0].get("message") or {}).get("content") or ""
    tokens = int((body.get("usage") or {}).get("completion_tokens") or 0)
    return {"content": content, "completion_tokens": tokens, "seconds": seconds}


def thinking_stays_off(base_url: str, model: str) -> dict[str, Any]:
    reply = chat(base_url, model, "Reply with the single word: ready", 64)
    content = reply["content"]
    return {"ok": bool(content.strip()) and THINK_TAG not in content,
            "has_think_block": THINK_TAG in content,
            "content_head": content[:120]}


def throughput(base_url: str, model: str, n: int, max_tokens: int) -> dict[str, Any]:
    """Aggregate decode tok/s with n requests in flight at once.

    ignore_eos makes every request generate exactly max_tokens, so the figure is
    a decode rate rather than a measure of how soon the model chose to stop.
    """
    prompt = "Count upward from 1, one number per line."
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        replies = list(pool.map(
            lambda _: chat(base_url, model, prompt, max_tokens, ignore_eos=True),
            range(n),
        ))
    wall = time.monotonic() - started
    tokens = sum(r["completion_tokens"] for r in replies)
    return {"n_concurrent": n, "max_tokens": max_tokens,
            "completion_tokens": tokens, "wall_s": round(wall, 1),
            "tok_per_s": round(tokens / wall, 1) if wall > 0 else None,
            "mean_latency_s": round(sum(r["seconds"] for r in replies) / n, 1)}


def main(model: str, base_url: str = "http://localhost:8000", wait_s: float = 1200,
         n_concurrent: int = 16, max_tokens: int = 256,
         out_dir: Path | None = None, poll_s: float = 5.0) -> int:
    result_path = (out_dir or RESULTS_DIR) / "serve_smoke.json"
    report: dict[str, Any] = {"model": model, "base_url": base_url, "ok": False}
    try:
        print(f"waiting up to {wait_s:.0f} s for {base_url}/health", flush=True)
        report["health_wait_s"] = wait_for_health(base_url, wait_s, poll_s)
        print(f"  healthy after {report['health_wait_s']} s", flush=True)

        try:
            report["server_version"] = json.loads(
                _get(f"{base_url}/version", timeout=10)[1]).get("version")
        except Exception as exc:  # the version is a nicety, not a check
            report["server_version"] = f"unavailable: {type(exc).__name__}"

        report["thinking_off"] = thinking_stays_off(base_url, model)
        print(f"  thinking off: {report['thinking_off']}", flush=True)

        report["throughput"] = throughput(base_url, model, n_concurrent, max_tokens)
        print(f"  throughput: {report['throughput']}", flush=True)

        report["ok"] = bool(report["thinking_off"]["ok"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(f"  FAILED: {report['error']}", flush=True)
    finally:
        # written whatever happened: a smoke test that dies without its record is
        # how the gate's evidence was lost last time
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n{'SMOKE TEST PASSED' if report['ok'] else 'SMOKE TEST FAILED'}"
              f"\nwrote {result_path.resolve()}", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="the name the server was started with")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--wait", type=float, default=1200,
                        help="seconds to wait for the server to become healthy")
    parser.add_argument("--concurrent", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the result (default: results/)")
    args = parser.parse_args()
    raise SystemExit(main(args.model, args.base_url, args.wait, args.concurrent,
                          args.max_tokens, args.out))
