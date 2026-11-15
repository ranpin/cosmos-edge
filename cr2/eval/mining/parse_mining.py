#!/usr/bin/env python3
"""Parse the dgural batch-mining run artifacts pulled from orin-dog.

Timing/power parsing reuses the proven caliber of
cr2/eval/accuracy/driving/pareto/parse_pareto.py (same run_pareto.sh produced
these logs): per-request E2E from consecutive "Processing vision inputs"
timestamps; power = VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0 three-rail sum,
tegrastats window = pure processing (started after first request, stopped at
llm_inference exit).

Label parsing: model outputs are ```json-fenced JSON objects; strip fences,
json.loads, count compliance. Labels are MODEL OUTPUTS (no ground truth for
these public clips) — the labeled evidence for event detection reliability is
the n184 per-cluster MC accuracy in ../accuracy/driving/RESULTS.json.

Usage: python3 parse_mining.py  (writes mining_results.json)
"""
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = ["01d3588e", "26f8dc7e", "69cf2443", "c3ca785e",
         "ecafce84", "76f4a312", "0dd3eb7f", "1f7ef408"]
TS = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]")
RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")


def log_seconds(line):
    m = TS.match(line)
    if not m:
        return None
    h, mi, s, ms = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def parse_run():
    starts, complete = [], None
    for line in open(os.path.join(HERE, "run_pareto_mining_dgural.log")):
        if "Processing vision inputs" in line:
            starts.append(log_seconds(line))
        elif "Processing complete" in line:
            complete = log_seconds(line)
    assert len(starts) == 8 and complete is not None
    e2e = [b - a for a, b in zip(starts, starts[1:])]
    e2e.append(complete - starts[-1])
    return e2e, complete - starts[0]


def parse_power():
    totals = []
    for line in open(os.path.join(HERE, "pw_pareto_mining_dgural.log")):
        mw = {}
        for rail in RAILS:
            m = re.search(rail + r"\s+(\d+)mW", line)
            if m:
                mw[rail] = int(m.group(1))
        if len(mw) == len(RAILS):
            totals.append(sum(mw.values()) / 1000)
    assert totals
    return statistics.mean(totals), max(totals), len(totals)


def parse_labels():
    d = json.load(open(os.path.join(HERE, "mining_out.json")))
    out, ok = [], 0
    for r in d["responses"]:
        txt = r.get("output_text", "")
        body = re.sub(r"^```json\s*|\s*```$", "", txt.strip())
        try:
            lab = json.loads(body)
            ok += 1
        except Exception:
            lab = {"parse_error": True, "raw": txt[:300]}
        out.append({"finish_reason": r.get("finish_reason"), "labels": lab})
    return out, ok


def main():
    e2e, wall = parse_run()
    p_avg, p_peak, n_pw = parse_power()
    labels, ok = parse_labels()
    j_clip = p_avg * wall / len(e2e)
    res = {
        "run": {
            "device": "Jetson AGX Orin 32GB (MAXN)",
            "engine": "CR2-8B INT4 AWQ backbone + INT4 lm_head (engines/8b_lmhead), visual fp16 shared",
            "runtime": "TensorRT-Edge-LLM 0.9.0 shared install, llm_inference batch_size=1",
            "protocol": "8 clips x 6 frames @448 long edge, greedy (temp=0, top_k=1), max_generate_length=300",
            "n_clips": 8, "n_frames": 48,
            "json_compliant": ok, "json_compliance_pct": 100.0 * ok / 8,
            "wall_s": round(wall, 2),
            "e2e_per_clip_s": [round(x, 3) for x in e2e],
            "e2e_mean_s": round(statistics.mean(e2e), 3),
            "e2e_median_s": round(statistics.median(e2e), 3),
            "avg_power_W": round(p_avg, 2), "peak_power_W": round(p_peak, 2),
            "n_power_samples": n_pw,
            "j_per_clip": round(j_clip, 1),
            "clips_per_hour": round(3600 / statistics.mean(e2e), 1),
            "j_per_1000_clips_kj": round(j_clip, 1),
        },
        "clips": [{"clip_id": c, "e2e_s": round(e, 3), **labels[i]}
                  for i, (c, e) in enumerate(zip(CLIPS, e2e))],
    }
    json.dump(res, open(os.path.join(HERE, "mining_results.json"), "w"),
              indent=1, ensure_ascii=False)
    r = res["run"]
    print("wall %.1fs  E2E mean %.2fs (median %.2f)  P avg %.1fW peak %.1fW (%d samples)"
          % (wall, r["e2e_mean_s"], r["e2e_median_s"], p_avg, p_peak, n_pw))
    print("JSON compliance %d/8  J/clip %.1f  clips/hour %.0f"
          % (ok, j_clip, r["clips_per_hour"]))


if __name__ == "__main__":
    main()
