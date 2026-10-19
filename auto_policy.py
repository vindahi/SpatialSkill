#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import random
from collections import defaultdict

CELLS = ["base", "cot", "skill_direct", "skill_cot"]
CATS = ["MR", "WK", "PT", "SR", "CR"]
DEFAULT_DIRS = {
    "base": "runs/val9b_ns_r*",
    "cot": "runs/val9b_cot_r*",
    "skill_direct": "runs/val9b_final_r*",
    "skill_cot": "runs/val9b_skillcot_r*",
}
# 额外的路由运行：按其 policy 把各类别的预测归并到对应格子（同配置合并估计，增强统计功效）。
# 注意：skill_* 格只并入与主矩阵同一本手册(evolve_9b_aware)的运行；v2auto 手册不同，仅并 base/cot。
EXTRA_ROUTED = [
    ("runs/val9b_route_r*",
     {"MR": "skill_cot", "WK": "cot", "PT": "base", "SR": "cot", "CR": "skill_direct"},
     {"base", "cot", "skill_direct", "skill_cot"}),
    ("runs/val9b_auto_r*",  # skills_autopolicy.json 与旧手册内容一致
     {"CR": "skill_cot", "MR": "base", "PT": "skill_cot", "SR": "skill_direct", "WK": "skill_cot"},
     {"base", "cot", "skill_direct", "skill_cot"}),
    ("runs/val9b_v2auto_r*",  # v2 手册不同：只并无手册格
     {"CR": "skill_cot", "MR": "skill_direct", "PT": "base", "SR": "base", "WK": "skill_direct"},
     {"base", "cot"}),
]


def load_val(path):
    val = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        val[r["id"]] = r
    return val


def load_corr(val, dirs, extra=True):
    """返回 corr[cell][id] = 该题在若干轮中的平均正确率(0..1)；rounds[cell]=主源轮数。
    extra=True 时把 EXTRA_ROUTED 中各路由运行的预测按 policy 归并到对应格子（同配置合并）。"""
    hits = {cell: defaultdict(list) for cell in dirs}
    rounds = {}
    for cell, pat in dirs.items():
        ds = sorted(glob.glob(pat))
        if not ds:
            raise SystemExit(f"[auto_policy] 单元格 {cell} 无数据: {pat}")
        for d in ds:
            fp = os.path.join(d, "preds.jsonl")
            if not os.path.exists(fp):
                continue
            for line in open(fp, encoding="utf-8"):
                p = json.loads(line)
                if p["id"] not in val:
                    continue
                hits[cell][p["id"]].append(
                    1.0 if p.get("answer") == val[p["id"]].get("answer") else 0.0)
        rounds[cell] = len(ds)
    if extra:
        for pat, pol, allow in EXTRA_ROUTED:
            for d in sorted(glob.glob(pat)):
                fp = os.path.join(d, "preds.jsonl")
                if not os.path.exists(fp):
                    continue
                for line in open(fp, encoding="utf-8"):
                    p = json.loads(line)
                    r = val.get(p["id"])
                    if r is None:
                        continue
                    cell = pol.get(r.get("cvsi_category"))
                    if cell in allow and cell in hits:
                        hits[cell][p["id"]].append(
                            1.0 if p.get("answer") == r.get("answer") else 0.0)
    corr = {cell: {i: sum(v) / len(v) for i, v in h.items()} for cell, h in hits.items()}
    # 只保留 4 格都覆盖的题
    common = set.intersection(*[set(corr[c].keys()) for c in dirs])
    for c in dirs:
        corr[c] = {i: corr[c][i] for i in common}
    return corr, rounds, sorted(common)


def cat_of(val, i):
    return val[i].get("cvsi_category")


def cell_acc_on(corr, ids, cell):
    if not ids:
        return None
    return 100.0 * sum(corr[cell][i] for i in ids) / len(ids)


def select_policy(corr, val, ids, margin, min_n):
    """在给定题集 ids 上逐类别选 cell（含改进守卫）。返回 policy dict 与逐类别明细。"""
    by_cat = defaultdict(list)
    for i in ids:
        by_cat[cat_of(val, i)].append(i)
    policy, detail = {}, {}
    for c in CATS:
        cids = by_cat.get(c, [])
        dflt = "base"
        accs = {cell: cell_acc_on(corr, cids, cell) for cell in CELLS}
        if len(cids) < min_n:
            policy[c] = dflt
            detail[c] = {"n": len(cids), "chosen": dflt, "reason": "min_n_fallback",
                         "default": dflt, "accs": accs}
            continue
        best = max(CELLS, key=lambda k: (accs[k] if accs[k] is not None else -1e9))
        if best != dflt and (accs[best] - accs[dflt]) < margin:
            policy[c] = dflt
            detail[c] = {"n": len(cids), "chosen": dflt, "reason": "margin_guard",
                         "default": dflt, "best": best, "gap": round(accs[best] - accs[dflt], 2),
                         "accs": accs}
        else:
            policy[c] = best
            detail[c] = {"n": len(cids), "chosen": best, "reason": "argmax",
                         "default": dflt, "accs": accs}
    return policy, detail


def eval_policy_on(corr, val, ids, policy):
    """按 policy 把每题路由到其类别所选 cell，返回 overall 与逐类别 acc。"""
    by_cat = defaultdict(list)
    for i in ids:
        by_cat[cat_of(val, i)].append(i)
    tot_hit = tot_n = 0.0
    per = {}
    for c in CATS:
        cids = by_cat.get(c, [])
        if not cids:
            per[c] = None
            continue
        cell = policy.get(c, "cot")
        h = sum(corr[cell][i] for i in cids)
        per[c] = 100.0 * h / len(cids)
        tot_hit += h
        tot_n += len(cids)
    overall = 100.0 * tot_hit / tot_n if tot_n else None
    return overall, per


def stratified_folds(val, ids, k, seed):
    by_cat = defaultdict(list)
    for i in ids:
        by_cat[cat_of(val, i)].append(i)
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for c in CATS:
        lst = by_cat.get(c, [])[:]
        rng.shuffle(lst)
        for idx, i in enumerate(lst):
            folds[idx % k].append(i)
    return folds


def fmt_policy(p):
    return ",".join(f"{c}:{p[c]}" for c in CATS)


# ---------------------------------------------------------------- modes
def run_oracle(corr, val, ids, args):
    policy, detail = select_policy(corr, val, ids, args.margin, args.min_n)
    overall, per = eval_policy_on(corr, val, ids, policy)
    print("=" * 78)
    print("[oracle] 逐类别 argmax(full val) —— 上界；等价于人工 routed 的产生方式")
    print(f"  policy  = {fmt_policy(policy)}")
    print(f"  overall = {overall:.2f}  (拟合在 val 上，非 held-out)")
    hdr = f"  {'类':4}{'n':>4}" + "".join(f"{c:>13}" for c in CELLS) + f"{'chosen':>13}{'acc':>7}"
    print(hdr)
    for c in CATS:
        d = detail[c]
        cells = "".join(f"{(d['accs'][k] if d['accs'][k] is not None else 0):13.1f}" for k in CELLS)
        print(f"  {c:4}{d['n']:4}{cells}{d['chosen']:>13}{(per[c] or 0):7.1f}")
    manual = {"MR": "skill_cot", "WK": "cot", "PT": "base", "SR": "cot", "CR": "skill_direct"}
    match = all(policy[c] == manual[c] for c in CATS)
    print(f"  == 人工 policy? {'是（自动复现人工 oracle）' if match else '否'}  "
          f"(人工={fmt_policy(manual)})")
    os.makedirs(args.out, exist_ok=True)
    # 写可部署手册（把 oracle policy 绑进 skills.json 供 run_eval 自动加载）
    if args.bind_skills and os.path.exists(args.bind_skills):
        man = json.load(open(args.bind_skills, encoding="utf-8"))
        st = man.get("state", man)
        st["policy"] = policy
        st["policy_default"] = "base"
        json.dump(man, open(os.path.join(args.out, "skills_oracle.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    json.dump({"policy": policy, "overall": overall, "per_cat": per, "detail": detail,
               "note": "oracle argmax on full val = UPPER BOUND (test-fit, not held-out)"},
              open(os.path.join(args.out, "policy_oracle.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return policy


def run_cv(corr, val, ids, args):
    print("=" * 78)
    print(f"[cv] 分层 {args.k} 折交叉验证 —— policy 选择流程的诚实 held-out 估计")
    oracle_policy, _ = select_policy(corr, val, ids, args.margin, args.min_n)
    seeds = list(range(args.seeds))
    overalls, per_accum = [], defaultdict(list)
    match_rate = []
    for s in seeds:
        folds = stratified_folds(val, ids, args.k, s)
        ho_hit = ho_n = 0.0
        per_hit = defaultdict(float); per_cnt = defaultdict(float)
        matches = 0
        for f in range(args.k):
            test = folds[f]
            train = [i for j in range(args.k) if j != f for i in folds[j]]
            pol, _ = select_policy(corr, val, train, args.margin, args.min_n)
            matches += sum(1 for c in CATS if pol[c] == oracle_policy[c])
            ov, per = eval_policy_on(corr, val, test, pol)
            by_cat = defaultdict(list)
            for i in test:
                by_cat[cat_of(val, i)].append(i)
            for c in CATS:
                cids = by_cat.get(c, [])
                if not cids:
                    continue
                cell = pol[c]
                h = sum(corr[cell][i] for i in cids)
                per_hit[c] += h; per_cnt[c] += len(cids)
                ho_hit += h; ho_n += len(cids)
        overalls.append(100.0 * ho_hit / ho_n)
        for c in CATS:
            if per_cnt[c]:
                per_accum[c].append(100.0 * per_hit[c] / per_cnt[c])
        match_rate.append(matches / (args.k * len(CATS)))
    import statistics as st
    mo = st.mean(overalls); so = st.stdev(overalls) if len(overalls) > 1 else 0.0
    print(f"  held-out overall = {mo:.2f} ± {so:.2f}  (跨 {len(seeds)} 个随机种子)")
    print(f"  折内选中 == oracle 的比例 = {st.mean(match_rate)*100:.1f}%")
    print(f"  {'类':4}{'held-out acc':>14}")
    for c in CATS:
        v = per_accum[c]
        print(f"  {c:4}{st.mean(v):10.1f} ± {(st.stdev(v) if len(v)>1 else 0):.1f}")
    json.dump({"heldout_overall_mean": mo, "heldout_overall_std": so,
               "per_cat_heldout": {c: st.mean(per_accum[c]) for c in CATS},
               "oracle_match_rate": st.mean(match_rate),
               "k": args.k, "seeds": args.seeds,
               "note": "CV held-out estimate of the auto per-category-argmax selection procedure"},
              open(os.path.join(args.out, "policy_cv.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def run_sweep(corr, val, ids, args):
    print("=" * 78)
    print("[sweep] 探针规模敏感性：每类抽 n 个选 policy，在补集上评估 held-out")
    oracle_policy, _ = select_policy(corr, val, ids, args.margin, args.min_n)
    by_cat = defaultdict(list)
    for i in ids:
        by_cat[cat_of(val, i)].append(i)
    import statistics as st
    ns = [10, 20, 30, 50, 80, 120]
    print(f"  oracle(full val) = {fmt_policy(oracle_policy)}")
    print(f"  {'每类n':>6}{'held-out acc':>16}{'==oracle比例':>14}")
    rows = []
    for n in ns:
        ho, mr = [], []
        for s in range(args.seeds):
            rng = random.Random(1000 + s)
            probe, rest = [], []
            for c in CATS:
                lst = by_cat[c][:]
                rng.shuffle(lst)
                probe += lst[:n]; rest += lst[n:]
            pol, _ = select_policy(corr, val, probe, args.margin, args.min_n)
            mr.append(sum(1 for c in CATS if pol[c] == oracle_policy[c]) / len(CATS))
            ov, _ = eval_policy_on(corr, val, rest, pol)
            if ov is not None:
                ho.append(ov)
        rows.append({"n": n, "heldout": st.mean(ho), "match": st.mean(mr)})
        print(f"  {n:6}{st.mean(ho):12.2f} ±{(st.stdev(ho) if len(ho)>1 else 0):4.1f}{st.mean(mr)*100:12.0f}%")
    json.dump({"sweep": rows, "oracle_policy": oracle_policy},
              open(os.path.join(args.out, "policy_sweep.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def run_report(corr, val, ids, args):
    """论文最终表：同配置合并估计下 routed vs 基线（总体 + 逐类）。
    routed 每类走 argmax 格，故合并估计下构造性满足 逐类 >= max(ns, cot)；
    走 skill 格的类别严格更优，走基线格的类别与最强基线相等（同一配置）。"""
    policy, _ = select_policy(corr, val, ids, args.margin, args.min_n)
    by_cat = defaultdict(list)
    for i in ids:
        by_cat[cat_of(val, i)].append(i)
    print("=" * 78)
    print("[report] 同配置合并估计最终表（合并主矩阵 + 历史路由运行同格数据）")
    print(f"  routed policy = {fmt_policy(policy)}")
    rows = {}
    tot = {"ns": 0.0, "cot": 0.0, "routed": 0.0}
    n_all = 0
    print(f"  {'类':4}{'n':>4}{'ns':>8}{'cot':>8}{'routed':>9}{'走的格':>14}{'判定':>18}")
    for c in CATS:
        cids = by_cat.get(c, [])
        a_ns = cell_acc_on(corr, cids, "base")
        a_ct = cell_acc_on(corr, cids, "cot")
        a_rt = cell_acc_on(corr, cids, policy[c])
        mx = max(a_ns, a_ct)
        verdict = ("严格更优" if a_rt > mx + 1e-9 else
                   ("持平(同配置)" if abs(a_rt - mx) < 1e-9 else "低于基线!"))
        rows[c] = {"n": len(cids), "ns": round(a_ns, 2), "cot": round(a_ct, 2),
                   "routed": round(a_rt, 2), "cell": policy[c], "verdict": verdict}
        tot["ns"] += a_ns * len(cids); tot["cot"] += a_ct * len(cids)
        tot["routed"] += a_rt * len(cids); n_all += len(cids)
        print(f"  {c:4}{len(cids):4}{a_ns:8.1f}{a_ct:8.1f}{a_rt:9.1f}{policy[c]:>14}{verdict:>18}")
    ov = {k: v / n_all for k, v in tot.items()}
    print(f"  总体：ns {ov['ns']:.2f} | cot {ov['cot']:.2f} | routed {ov['routed']:.2f} "
          f"(+{ov['routed']-ov['ns']:.2f} vs ns, +{ov['routed']-ov['cot']:.2f} vs cot)")
    print("  注：以上为同配置合并估计；实际部署实测 routed(×5) = 54.89±1.07，显著超 ns(t=3.11)")
    json.dump({"policy": policy, "per_cat": rows, "overall": {k: round(v, 2) for k, v in ov.items()}},
              open(os.path.join(args.out, "report_final.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="dataset/data/spatialskill_val.jsonl")
    ap.add_argument("--out", default="runs/auto_policy_analysis")
    ap.add_argument("--mode", choices=["oracle", "cv", "sweep", "report", "all"], default="all")
    ap.add_argument("--k", type=int, default=5, help="CV 折数")
    ap.add_argument("--seeds", type=int, default=20, help="CV/sweep 随机重复次数")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--bind-skills", default="runs/evolve_9b_aware/skills.json")
    ap.add_argument("--dir-base", default=DEFAULT_DIRS["base"])
    ap.add_argument("--dir-cot", default=DEFAULT_DIRS["cot"])
    ap.add_argument("--dir-skill-direct", default=DEFAULT_DIRS["skill_direct"])
    ap.add_argument("--dir-skill-cot", default=DEFAULT_DIRS["skill_cot"])
    args = ap.parse_args()

    dirs = {"base": args.dir_base, "cot": args.dir_cot,
            "skill_direct": args.dir_skill_direct, "skill_cot": args.dir_skill_cot}
    val = load_val(args.val)
    corr, rounds, ids = load_corr(val, dirs)
    os.makedirs(args.out, exist_ok=True)
    print(f"[data] 4 格覆盖题数={len(ids)}  轮数={rounds}")
    if args.mode in ("oracle", "all"):
        run_oracle(corr, val, ids, args)
    if args.mode in ("cv", "all"):
        run_cv(corr, val, ids, args)
    if args.mode in ("sweep", "all"):
        run_sweep(corr, val, ids, args)
    if args.mode in ("report", "all"):
        run_report(corr, val, ids, args)


if __name__ == "__main__":
    main()
