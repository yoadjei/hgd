# h* census: cpu only, no model, about 2 minutes for all 90 train tasks.
# decides the compute budget before any gpu quota is spent, and whether the
# audit's step limit of 3*h* is affordable at all.
import json, statistics as st
from appworld import AppWorld, load_task_ids

rows = []
for tid in load_task_ids("train"):
    try:
        with AppWorld(task_id=tid, experiment_name="hstar_census",
                      ground_truth_mode="full") as w:
            gt = w.task.ground_truth
            calls = getattr(gt, "api_calls", None) or []
            rows.append({
                "task_id": tid,
                "h_api": len(calls),
                "h_lines": getattr(gt, "num_solution_code_lines", None),
                "h_compiled": getattr(gt, "num_compiled_solution_code_lines", None),
                "difficulty": getattr(getattr(gt, "metadata", None), "difficulty", None),
            })
    except Exception as e:
        rows.append({"task_id": tid, "error": f"{type(e).__name__}: {e}"})

ok = [r for r in rows if "error" not in r]
print(f"{len(ok)}/{len(rows)} tasks read\n")

def summarise(key):
    vals = [r[key] for r in ok if isinstance(r.get(key), int)]
    if not vals:
        print(f"{key:12} unavailable"); return None
    vals.sort()
    print(f"{key:12} n={len(vals):3}  min={vals[0]:3}  p25={vals[len(vals)//4]:3}  "
          f"median={int(st.median(vals)):3}  p75={vals[3*len(vals)//4]:3}  max={vals[-1]:3}")
    return vals

h_api = summarise("h_api")
h_lines = summarise("h_lines")
h_compiled = summarise("h_compiled")

# budget implication. a step is one model call; ~300 output tokens is typical
# for a react turn on appworld.
if h_api:
    med = st.median(h_api)
    for mult in (2, 3):
        steps = med * mult
        # mve per audit section 22: 30 scenarios x 3 levels x 3 seeds x 2 models
        # = 540 composed + 270 atomic = 810 episodes
        tokens = 810 * steps * 300
        print(f"\nstep limit {mult}*H*: median {steps:.0f} steps/episode")
        print(f"  810 MVE episodes -> {tokens/1e6:.1f}M output tokens")
        for rate, label in ((500, "2xT4 vLLM batched"), (3, "CPU int4")):
            hours = tokens / rate / 3600
            print(f"  at {rate:4} tok/s ({label:18}): {hours:8.1f} h"
                  f"  = {hours/30:6.1f} weeks of Kaggle quota")

# tasks cheap enough to be worth running first
if h_api:
    cheap = sorted([r for r in ok if isinstance(r.get("h_api"), int)],
                   key=lambda r: r["h_api"])[:30]
    print(f"\n30 cheapest tasks: H* from {cheap[0]['h_api']} to {cheap[-1]['h_api']}, "
          f"median {int(st.median([c['h_api'] for c in cheap]))}")
    print("ids:", [c["task_id"] for c in cheap[:10]], "...")

json.dump(rows, open("hstar_census.json", "w"), indent=2, default=str)
print("\nwrote hstar_census.json")
