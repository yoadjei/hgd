# what is the executable form of the gold solution? paste-able single cell, cpu only.
#
# two action sources have already been wrong. api_calls is a list of http record
# dicts that execute() evaluates as no-op literals. compiled_solution_code is a
# `def solution(apis, requester)` wrapper that execute() defines and never calls,
# so the world never moves and the gate is void.
#
# this enumerates what the installed package actually offers and tries every
# plausible invocation, reporting which one moves the state. it decides nothing.
import contextlib, hashlib, io, json
from appworld import AppWorld, load_task_ids

EXP = "solution_source_probe"
TID = list(load_task_ids("train"))[0]
P = {"task_id": TID}


def digest(w):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        p = w.evaluate().to_dict()
    return hashlib.sha256(json.dumps({
        "passes": sorted(p.get("passes") or []),
        "failures": sorted(p.get("failures") or []),
        "success": p.get("success"),
        "num_tests": p.get("num_tests"),
    }, sort_keys=True, default=str).encode()).hexdigest()


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


# --- 1. what does the package expose? --------------------------------------
with AppWorld(task_id=TID, experiment_name=EXP, ground_truth_mode="full") as w:
    gt = w.task.ground_truth
    P["gt_fields"] = sorted(n for n in dir(gt) if not n.startswith("_"))
    P["world_members"] = sorted(n for n in dir(w) if not n.startswith("_"))
    P["task_members"] = sorted(n for n in dir(w.task) if not n.startswith("_"))
    # anything whose name hints at running the reference solution
    P["solution_like"] = sorted(
        n for n in P["world_members"] + P["task_members"] + P["gt_fields"]
        if any(k in n.lower() for k in ("solution", "gold", "gt", "reference", "oracle"))
    )
    for name in ("compiled_solution_code", "solution_code"):
        value = getattr(gt, name, None)
        P[name] = str(value) if value else None
    # what names already live in the shell the agent writes into?
    P["shell_names"] = str(w.execute("sorted(n for n in dir() if not n.startswith('_'))"))[:1200]

print("=== fields and members ===")
print(json.dumps(
    {k: P[k] for k in ("task_id", "gt_fields", "solution_like", "world_members", "task_members")},
    indent=2, default=str)[:2600])
print("\n=== shell namespace ===")
print(P["shell_names"])
print("\n=== compiled_solution_code (full) ===")
print(P["compiled_solution_code"])
print("\n=== solution_code (first 900 chars) ===")
print((P["solution_code"] or "<absent>")[:900])

# --- 2. which invocation actually moves the state? -------------------------
# each candidate runs in its own fresh world, so they cannot contaminate each other.
CANDIDATES = {
    "compiled_then_call_apis_requester":
        [P["compiled_solution_code"], "solution(apis, requester)"],
    "compiled_then_call_apis_only":
        [P["compiled_solution_code"], "solution(apis)"],
    "solution_code_as_is":
        [P["solution_code"]],
    "compiled_body_dedented":
        None,  # filled below if the wrapper shape is what we think
}

# strip the `def solution(...)` wrapper and dedent its body, if present
code = P["compiled_solution_code"] or ""
if "def solution(" in code:
    head, _, body = code.partition("def solution(")
    _, _, body = body.partition("\n")
    lines = [ln[4:] if ln.startswith("    ") else ln for ln in body.split("\n")]
    imports = "\n".join(
        ln for ln in head.split("\n") if ln.startswith(("import ", "from "))
    )
    CANDIDATES["compiled_body_dedented"] = [imports + "\n" + "\n".join(lines)]

P["attempts"] = {}
for label, steps in CANDIDATES.items():
    if not steps or not all(steps):
        P["attempts"][label] = {"skipped": "source absent"}
        continue
    try:
        with AppWorld(task_id=TID, experiment_name=EXP, ground_truth_mode="full") as w:
            before = quiet(digest, w)
            outputs = [str(quiet(w.execute, s))[-400:] for s in steps]
            after = quiet(digest, w)
        P["attempts"][label] = {
            "moved_state": before != after,
            "last_output": outputs[-1],
        }
    except Exception as exc:
        P["attempts"][label] = {"error": f"{type(exc).__name__}: {exc}"[:400]}

print("\n=== which invocation moves the state? ===")
for label, result in P["attempts"].items():
    verdict = result.get("moved_state", result.get("error", result.get("skipped")))
    print(f"  {label:36} -> {verdict}")
    if result.get("last_output"):
        print(f"      output tail: {result['last_output'][:220]!r}")

winners = [k for k, v in P["attempts"].items() if v.get("moved_state")]
print("\nWORKING SOURCE:", winners or "NONE - paste this whole output back")

json.dump(P, open("solution_source_probe.json", "w"), indent=2, default=str)
print("\n--- paste this back ---")
print(json.dumps(P, indent=2, default=str)[:3000])
