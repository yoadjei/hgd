# phase 1 gate v3, paste-able single cell, cpu only. kill condition C.
#
# the action source took three attempts to get right, each failure silent:
#   v1  api_calls               -> http record dicts, execute() reads them as
#                                  no-op literals. nothing ran.
#   v2  compiled_solution_code  -> a `def solution(apis, requester)` wrapper.
#                                  execute() defined it and never called it.
#   v3  the same wrapper, then `solution(apis, requester)`. verified by
#       notebooks/solution_source_probe.py on 2026-09-20: the task went from
#       1/2 to 2/2 passing tests, so the world genuinely moved.
#
# the no-op guard caught v1 and v2. without it this cell would have printed
# 20/20 PASS twice while measuring nothing.
import contextlib, hashlib, io, json, os, sys, tempfile
from appworld import AppWorld, load_task_ids

N, EXP = 20, "phase1_gate_v3"
F = {"action_source": "compiled_solution_code + solution(apis, requester)"}


@contextlib.contextmanager
def silenced():
    # evaluate() prints a full report per call. redirect_stdout alone does not
    # stop it: the package reports through a stream captured before the
    # redirect. duplicate the descriptors as well, using literal 1 and 2
    # because a notebook stream has no usable fileno().
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    saved = []
    with tempfile.TemporaryFile() as sink:
        try:
            for fd in (1, 2):
                with contextlib.suppress(OSError, ValueError):
                    saved.append((fd, os.dup(fd)))
                    os.dup2(sink.fileno(), fd)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
        finally:
            for fd, backup in saved:
                with contextlib.suppress(OSError, ValueError):
                    os.dup2(backup, fd)
                    os.close(backup)


def digest(w):
    with silenced():
        p = w.evaluate().to_dict()
    return hashlib.sha256(json.dumps({
        "passes": sorted(p.get("passes") or []),
        "failures": sorted(p.get("failures") or []),
        "success": p.get("success"),
        "num_tests": p.get("num_tests"),
    }, sort_keys=True, default=str).encode()).hexdigest()


def gold_code(tid):
    with AppWorld(task_id=tid, experiment_name=EXP, ground_truth_mode="full") as w:
        gt = w.task.ground_truth
        code = getattr(gt, "compiled_solution_code", None)
        return str(code) if code else ""


def run_gold(w, code):
    """Define the wrapper, then call it. Defining alone leaves the world untouched."""
    with silenced():
        w.execute(code)
        return str(w.execute("solution(apis, requester)"))[-300:]


ids = list(load_task_ids("train"))[:N]

# sanity check: the gold solution must move the state on one task before the
# gate means anything at all.
probe_code = gold_code(ids[0])
with AppWorld(task_id=ids[0], experiment_name=EXP, ground_truth_mode="full") as w:
    before = digest(w)
    tail = run_gold(w, probe_code)
    after = digest(w)
F["gold_changes_state"] = before != after
F["probe_output_tail"] = tail
print(f"gold solution: {len(probe_code)} chars, changes state: {F['gold_changes_state']}")
print("output tail:", repr(tail[:200]))

if not F["gold_changes_state"]:
    print("\n!! the gold solution still did not move the state. the gate is void.")
    print("   paste this output back; do not read anything into a fidelity number.")
else:
    matched, diverged, no_effect, skipped, errored = 0, [], [], [], []
    for tid in ids:
        try:
            code = gold_code(tid)
            if not code:
                skipped.append(tid); continue
            with AppWorld(task_id=tid, experiment_name=EXP, ground_truth_mode="full") as w:
                baseline = digest(w)
            ds = []
            for _ in range(2):
                with AppWorld(task_id=tid, experiment_name=EXP, ground_truth_mode="full") as w:
                    run_gold(w, code)
                    ds.append(digest(w))
            if ds[0] == baseline:
                no_effect.append(tid)
            elif ds[0] == ds[1]:
                matched += 1
            else:
                diverged.append(tid)
        except Exception as e:
            errored.append((tid, f"{type(e).__name__}: {e}"[:300]))

    total = matched + len(diverged)
    F["gate"] = {"total": total, "matched": matched, "diverged": diverged,
                 "no_effect": no_effect, "skipped": skipped, "errored": errored[:5],
                 "n_errored": len(errored),
                 "rate": matched / total if total else 0.0,
                 # a run whose tasks never moved is void, not passing
                 "passes_gate": total > 0 and matched == total and not no_effect}
    print(f"\nfidelity {matched}/{total}  no_effect={len(no_effect)}"
          f"  skipped={len(skipped)}  errored={len(errored)}")
    if errored:
        print("first errors:", errored[:3])
    print("GATE PASSED - Phase 3a permitted." if F["gate"]["passes_gate"]
          else "GATE NOT PASSED - do not proceed to causal analysis.")

json.dump(F, open("phase1_gate.json", "w"), indent=2, default=str)
print("\n--- paste this back ---")
print(json.dumps(F, indent=2, default=str)[:2500])
