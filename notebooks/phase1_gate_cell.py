# phase 1 gate v4, paste-able single cell, cpu only. kill condition C.
#
# set N = 3 for a smoke test first, then N = 20 for the real gate.
#
# the action source took three attempts, each failure silent:
#   v1  api_calls              -> http record dicts; execute() reads them as
#                                 no-op literals, so nothing ran
#   v2  compiled_solution_code -> a `def solution(apis, requester)` wrapper;
#                                 execute() defined it and never called it
#   v3  the wrapper, then `solution(apis, requester)`. verified on 2026-09-20:
#       the probe task went from 1/2 to 2/2 passing, so the world moved.
#
# v4 is about survivability rather than correctness. v3 built four worlds per
# task, printed nothing for minutes, and wrote its result only at the very end,
# so an interrupted run left no evidence at all. this builds two worlds per
# task, prints one line per task, and rewrites the json after every task.
import contextlib, hashlib, io, itertools, json, os, sys, tempfile, time, traceback
from appworld import AppWorld, load_task_ids

N, EXP = 3, "phase1_gate_v4"
RESULT = "phase1_gate.json"
F = {"n_requested": N,
     "action_source": "compiled_solution_code + solution(apis, requester)"}


@contextlib.contextmanager
def silenced():
    # evaluate() prints a full report per call, and the package also reports on
    # world construction and close. redirect_stdout alone does not stop it, so
    # the descriptors are duplicated too. literal 1 and 2, because a notebook
    # stream has no usable fileno() and asking it raises, which silently
    # skipped the whole redirect in v3.
    # three layers, because the first two were not enough. the package reports
    # through a stream object captured before any redirect, so neutralising the
    # object's own write is what actually catches it.
    patched = []
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
        try:
            original = stream.write
            stream.write = lambda *a, **k: None
            patched.append((stream, original))
        except Exception:
            pass

    saved = []
    try:
        with tempfile.TemporaryFile() as sink:
            try:
                for fd in (1, 2):
                    with contextlib.suppress(OSError, ValueError):
                        saved.append((fd, os.dup(fd)))
                        os.dup2(sink.fileno(), fd)
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    yield
            finally:
                for fd, backup in saved:
                    with contextlib.suppress(OSError, ValueError):
                        os.dup2(backup, fd)
                        os.close(backup)
    finally:
        for stream, original in patched:
            with contextlib.suppress(Exception):
                stream.write = original


def outcomes(items):
    # entries are not always strings. once a task completes appworld returns
    # dicts, and sorted() raises "'<' not supported between instances of 'dict'
    # and 'dict'". serialising each entry first orders either shape.
    return sorted(json.dumps(i, sort_keys=True, default=str) for i in (items or []))


def digest(w):
    p = w.evaluate().to_dict()
    return hashlib.sha256(json.dumps({
        "passes": outcomes(p.get("passes")),
        "failures": outcomes(p.get("failures")),
        "success": p.get("success"),
        "num_tests": p.get("num_tests"),
    }, sort_keys=True, default=str).encode()).hexdigest()


def run_gold(w, code):
    """Define the wrapper, then call it. Defining alone leaves the world untouched."""
    w.execute(code)
    return str(w.execute("solution(apis, requester)"))[-300:]


_sequence = itertools.count()


def world(tid):
    """A world with an experiment name of its own.

    Every world gets a unique name. Reusing one across two runs of the same task
    made the second construction raise, and it is also wrong on its own terms:
    a determinism test must not let run two inherit run one's output directory.
    """
    return AppWorld(
        task_id=tid,
        experiment_name=f"{EXP}_{next(_sequence)}",
        ground_truth_mode="full",
    )


def gate_one(tid):
    """One task. Two worlds: baseline and run one, then a fresh world for run two.

    Returns a verdict. `no_effect` means the actions never moved the state, so
    the two runs agreeing proves nothing; that is void, not a pass.
    """
    with world(tid) as w:
        code = getattr(w.task.ground_truth, "compiled_solution_code", None)
        if not code:
            return "skipped"
        code = str(code)
        baseline = digest(w)
        run_gold(w, code)
        first = digest(w)

    with world(tid) as w:
        run_gold(w, code)
        second = digest(w)

    if first == baseline:
        return "no_effect"
    return "matched" if first == second else "diverged"


ids = list(load_task_ids("train"))[:N]
buckets = {k: [] for k in ("matched", "diverged", "no_effect", "skipped", "errored")}
F["errors"] = []

print(f"gate over {len(ids)} tasks, two worlds each\n")
for index, tid in enumerate(ids, start=1):
    started, detail = time.time(), ""
    try:
        with silenced():
            verdict = gate_one(tid)
    except Exception as exc:
        verdict = "errored"
        detail = f"{type(exc).__name__}: {exc}"[:200]
        # the first failing task carries a full traceback; a summary printed
        # only at the end is worthless when the cell output truncates
        if not F["errors"]:
            F["first_traceback"] = traceback.format_exc()[-2000:]
        F["errors"].append([tid, detail])
    buckets[verdict].append(tid)
    print(f"[{index:2}/{len(ids)}] {tid:<14} {verdict:<10} "
          f"{time.time() - started:5.1f}s {detail}")

    total = len(buckets["matched"]) + len(buckets["diverged"])
    # counts are n_-prefixed and the bare names hold task id lists, so the
    # spread below cannot silently turn a headline count into a list
    F["gate"] = {
        **buckets,
        "total": total,
        "n_matched": len(buckets["matched"]),
        "rate": len(buckets["matched"]) / total if total else 0.0,
        # a run whose tasks never moved is void, not passing
        "passes_gate": (total == len(ids) and not buckets["diverged"]
                        and not buckets["no_effect"] and not buckets["errored"]),
    }
    json.dump(F, open(RESULT, "w"), indent=2, default=str)  # crash-safe

g = F["gate"]
print(f"\nfidelity {g['n_matched']}/{g['total']}"
      f"   no_effect={len(g['no_effect'])}"
      f"   skipped={len(g['skipped'])}"
      f"   errored={len(g['errored'])}")
if F.get("first_traceback"):
    print("\n--- first traceback ---")
    print(F["first_traceback"])
if g["no_effect"]:
    print("VOID - those tasks never moved the state; the action source is still wrong.")
elif g["passes_gate"]:
    print("GATE PASSED - Phase 3a permitted.")
else:
    print("GATE NOT PASSED - do not proceed to causal analysis.")

print(f"\nwrote {RESULT}")
print("--- paste this back ---")
print(json.dumps(F, indent=2, default=str)[:2000])
