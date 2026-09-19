# phase 1 gate v2, paste-able single cell. fixes from the v1 run against the
# real package:
#   1. action source is compiled_solution_code (real python), not api_calls,
#      which is a list of http record dicts that execute() evaluates as no-ops
#   2. digest key is 'failures', not 'fails'
#   3. no-op guard: a task whose state never moved is void, not a match
#   4. evaluate() output suppressed; it prints a full test report per call
import contextlib, hashlib, io, json
from appworld import AppWorld, load_task_ids

N, EXP = 20, "phase1_gate_v2"
F = {}

def digest(w):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
        for field in ("compiled_solution_code", "solution_code"):
            code = getattr(gt, field, None)
            if code:
                return str(code)
    return ""

ids = list(load_task_ids("train"))[:N]

# sanity check: the gold solution must actually move the state on one task
probe_code = gold_code(ids[0])
F["gold_code_chars"] = len(probe_code)
F["gold_code_head"] = probe_code[:200]
with AppWorld(task_id=ids[0], experiment_name=EXP, ground_truth_mode="full") as w:
    before = digest(w)
    w.execute(probe_code)
    after = digest(w)
F["gold_changes_state"] = before != after
print(f"gold solution: {len(probe_code)} chars, changes state: {F['gold_changes_state']}")
print("head:", repr(probe_code[:200]))
if not F["gold_changes_state"]:
    print("\n!! gold solution did not move the state - the gate would be void again.")
    print("   Stopping. Paste the head above and we will find the right action source.")
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
                    w.execute(code)
                    ds.append(digest(w))
            if ds[0] == baseline:
                no_effect.append(tid)
            elif ds[0] == ds[1]:
                matched += 1
            else:
                diverged.append(tid)
        except Exception as e:
            errored.append((tid, f"{type(e).__name__}: {e}"))

    total = matched + len(diverged)
    F["gate"] = {"total": total, "matched": matched, "diverged": diverged,
                 "no_effect": no_effect, "skipped": skipped, "errored": errored[:5],
                 "n_errored": len(errored),
                 "rate": matched / total if total else 0.0,
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
