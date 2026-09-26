#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""技能感知进化螺旋（EmbodiSkill 主循环，训练无关）。

窗口式演化（修订间隔 B = --interval）：
  每个窗口内 manual 冻结 -> 执行器对窗口内样本推理（串行 infer_one 或并行 run_executor）
  -> 逐样本 LLM 反思（K=1，窗口内并发，库冻结故等价）-> 窗口末 LLM 修订 manual（版本+1）-> 快照。
反思/修订全部由 Qwen3-8B 反思器完成；执行器（Qwen VLM）只答题。
skill-aware：四类反思 + 执行提醒 appendix；skill-unaware 消融：单类反思、只写板块、不注入提醒。
进化只用 train；val 交给 run_eval 在 manual 冻结后评测。
"""
import argparse
import json
import os
import random

from . import config as C
from . import data as D
from . import executor as E
from . import llm as L
from . import retrieval as R
from . import reflection as RF
from . import revision as RV
from . import manual as MAN


def _snapshot(manual, out_dir, window, running_acc, rev_info, processed, log_path):
    manual.save(os.path.join(out_dir, f"skills_win{window}.json"))
    manual.save(os.path.join(out_dir, "skills.json"))  # 最新
    entry = {"window": window, "processed": processed, "train_acc": running_acc,
             "manual": manual.summary(), "revision": rev_info}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[win {window}] processed={processed} train_acc={running_acc} "
          f"manual={manual.summary()} rev={rev_info}", flush=True)
    return entry


def _atomic_json(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _serialize_streak(streak):
    return {key: {"count": count, "title": title, "item": item}
            for key, (count, title, item) in streak.items()}


def _deserialize_streak(raw):
    streak = {}
    for key, value in (raw or {}).items():
        if not isinstance(value, dict):
            continue
        try:
            streak[key] = (int(value["count"]), value["title"], value["item"])
        except (KeyError, TypeError, ValueError):
            continue
    return streak


def _load_prediction_checkpoint(path, valid_ids):
    results = {}
    if not os.path.exists(path):
        return results
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                result = json.loads(line)
                rid = result.get("id")
                if rid in valid_ids:
                    results[rid] = result
            except (json.JSONDecodeError, AttributeError):
                continue
    return results


def _run_api_window_checkpointed(jobs, gpus, tag, workdir, enable_thinking=False):
    """逐条追加窗口 API 结果；中断或最终失败后下次调用只补未完成题。"""
    os.makedirs(workdir, exist_ok=True)
    checkpoint_path = os.path.join(workdir, "preds.jsonl")
    job_ids = {job["id"] for job in jobs}
    results = _load_prediction_checkpoint(checkpoint_path, job_ids)
    pending = [job for job in jobs if job["id"] not in results]
    if results:
        print(f"[evolve] {tag} resumed API checkpoint: {len(results)}/{len(jobs)} completed",
              flush=True)
    if not pending:
        return results

    with open(checkpoint_path, "a", encoding="utf-8") as checkpoint:
        def save_result(result):
            rid = result["id"]
            if rid in results:
                return
            checkpoint.write(json.dumps(result, ensure_ascii=False) + "\n")
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
            results[rid] = result

        try:
            fresh = E.run_executor(pending, gpus=gpus, tag=tag,
                                   enable_thinking=enable_thinking,
                                   workdir=workdir, result_sink=save_result)
        except (E.ExecutorInterrupted, E.ExecutorAPIError):
            print(f"[evolve] {tag} stopped with a reusable checkpoint at {checkpoint_path}; "
                  "fix API settings and re-run with the same RUN_ID.", flush=True)
            raise
        for result in fresh.values():
            save_result(result)
    return results


def _score_batch(batch, manual, aware, direct):
    """用给定手册对一批样本打分（与部署形态一致：direct 无证据句），返回 acc。
    通过统一执行器路由，兼容本地服务与外部 API。"""
    skill_text = R.inject_text(manual, aware=aware)
    jobs = [D.build_job(r, skill_text=skill_text, cot=(not direct)) for r in batch]
    res = E.run_executor(jobs, tag="probe_score")
    ok = sum(1 for r in batch if (res.get(r["id"]) or {}).get("answer") == r.get("answer"))
    return round(100.0 * ok / max(1, len(batch)), 2)


_POLICY_CELLS = ["base", "cot", "skill_direct", "skill_cot"]


def _stratified_probe(train_recs, n, val_recs, seed):
    """按 val 五类比例分层从 train 抽取探针（与 full val 分布一致）。
    返回 (probe, rest)：rest 为剔除探针后的进化流，两者不交。"""
    from collections import defaultdict
    vc = defaultdict(int)
    for r in val_recs:
        vc[r.get("cvsi_category")] += 1
    total = sum(vc.values()) or 1
    buckets = defaultdict(list)
    for r in train_recs:
        buckets[r.get("cvsi_category")].append(r)
    rng = random.Random(seed)
    for c in buckets:
        rng.shuffle(buckets[c])
    probe = []
    for c in sorted(vc):
        k = int(round(n * vc[c] / total))
        probe.extend(buckets.get(c, [])[:k])
    probe_ids = {r["id"] for r in probe}
    rest = [r for r in train_recs if r["id"] not in probe_ids]
    return probe, rest


def _score_cells_by_cat(probe, manual, aware):
    """对探针逐个格子（base/cot/skill_direct/skill_cot）评测，返回
    (table[cat][cell]=acc, ncat[cat]=探针样本数)。每格子一遍探针（同 id 不冲突）。"""
    from collections import defaultdict
    skill_text = R.inject_text(manual, aware=aware)
    ncat = defaultdict(int)
    for r in probe:
        ncat[r.get("cvsi_category")] += 1
    cats = sorted(ncat)
    table = {c: {} for c in cats}
    for cell in _POLICY_CELLS:
        st = skill_text if cell in ("skill_direct", "skill_cot") else None
        cot = cell in ("cot", "skill_cot")
        jobs = [D.build_job(r, skill_text=st, cot=cot) for r in probe]
        res = E.run_executor(jobs, tag=f"policy_{cell}")
        hit = defaultdict(int)
        for r in probe:
            if (res.get(r["id"]) or {}).get("answer") == r.get("answer"):
                hit[r.get("cvsi_category")] += 1
        for c in cats:
            table[c][cell] = round(100.0 * hit[c] / max(1, ncat[c]), 2)
    return table, dict(ncat)


def _search_policy(table, ncat, default_cell, margin, min_probe):
    """逐类 argmax + 双防过拟合守卫：探针样本不足退默认；非默认格须高出默认格 >= margin 才切换。"""
    policy = {}
    for c, cells in table.items():
        if ncat.get(c, 0) < min_probe:
            policy[c] = default_cell
            continue
        best = max(cells, key=lambda k: cells[k])
        if best != default_cell and cells[best] - cells.get(default_cell, -1e9) < margin:
            policy[c] = default_cell
        else:
            policy[c] = best
    return policy


def _update_streaks(streak, manual):
    """逐窗更新规则存活计数（归一化文本为键）；消失的规则计数清零。"""
    new = {}
    for sec in manual.sections():
        for it in sec.get("items", []):
            k = MAN.item_key(it)
            new[k] = (streak.get(k, (0, None, None))[0] + 1, sec["title"], it)
    return new


def evolve(exp, gpus=None, interval=C.B_INTERVAL, epochs=1, aware=True,
           seed=0, init_skills=None, max_examples=0, serial=True,
           enable_thinking=False, direct=False):
    C.ensure_dirs()
    if not L.available():
        if C.SKILL_LLM_BACKEND == "vllm":
            raise SystemExit(
                f"反思器 vLLM 未就绪（{C.SKILL_LLM_URL}）。请先启动：bash src/serve_brain.sh")
        if C.SKILL_LLM_BACKEND == "siliconflow":
            raise SystemExit("SILICONFLOW_API_KEY 未设置，无法进行 LLM 反思/修订。请先 source src/env.sh。")
        raise SystemExit("DEEPSEEK_API_KEY 未设置，无法进行 LLM 反思/修订。请先 export 后再运行。")
    # exp 是当前运行根目录下的阶段名；显式绝对路径/旧 runs/ 路径仍可复用。
    out_dir = C.resolve_run_output(exp)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "evolve_log.jsonl")
    state_path = os.path.join(out_dir, "evolve_state.json")

    recs = D.load_records("train")
    random.Random(seed).shuffle(recs)
    val_recs = D.load_records("val")
    # E4v2：固定探针集（不进进化流），每窗裁决修订接受/拒绝；棘轮保证手册只升不降
    # 探针按 val 五类比例分层抽样（与 full val 分布一致），使棘轮/路由搜索与部署分布对齐
    if getattr(C, "POLICY_STRATIFIED_PROBE", False):
        probe, recs = _stratified_probe(recs, C.EVOLVE_PROBE_N, val_recs, seed)
    else:
        probe = recs[:C.EVOLVE_PROBE_N]
        recs = recs[C.EVOLVE_PROBE_N:]
    if max_examples:
        recs = recs[:max_examples]
    batches = [recs[start:start + interval] for _epoch in range(max(1, epochs))
               for start in range(0, len(recs), interval)]

    # 每窗完成后保存的状态足以精确恢复：手册、棘轮分数、连续拒绝次数及规则保护 streak。
    resume_state = None
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                resume_state = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"无法读取进化恢复状态 {state_path}: {e}") from e
    if resume_state and resume_state.get("complete"):
        print(f"[evolve] already complete -> {out_dir}", flush=True)
        return out_dir

    if resume_state:
        expected_resume_args = {
            "interval": interval,
            "epochs": epochs,
            "max_examples": max_examples,
            "seed": seed,
            "direct": direct,
            "aware": aware,
        }
        for key, current in expected_resume_args.items():
            if key in resume_state and resume_state[key] != current:
                raise RuntimeError(
                    f"恢复参数 {key}={current!r} 与保存值 {resume_state[key]!r} 不一致；"
                    "请使用原来的命令参数，或为新实验创建新的 RUN_ID。")
        skills_path = os.path.join(out_dir, "skills.json")
        if not os.path.exists(skills_path):
            raise RuntimeError(f"恢复状态存在但缺少手册：{skills_path}")
        manual = MAN.Manual.load(skills_path)
        window = int(resume_state.get("completed_windows", 0))
        processed = int(resume_state.get("processed", 0))
        ref_score = float(resume_state["ref_score"])
        reject_run = int(resume_state.get("reject_run", 0))
        streak = _deserialize_streak(resume_state.get("streak"))
        if not streak:
            streak = _update_streaks({}, manual)
        if window < 0 or window > len(batches):
            raise RuntimeError(
                f"恢复窗口数无效：{window}（当前配置共 {len(batches)} 窗）；"
                "请使用原来的 --interval/--epochs/--max-examples 参数。")
        print(f"[evolve] resumed at window {window + 1}/{len(batches)}: "
              f"processed={processed} probe={ref_score} manual={manual.summary()}", flush=True)
    else:
        # 初始化 manual：给定冷启动或空册（对齐 EmbodiSkill 无种子技能）
        if init_skills and os.path.exists(init_skills):
            manual = MAN.Manual.load(init_skills)
        else:
            manual = MAN.cold_start_manual()
        manual.save(os.path.join(out_dir, "skills_win0.json"))
        manual.save(os.path.join(out_dir, "skills.json"))
        # 规则存活追踪：连续存活 >= PROTECT_STREAK 窗的规则受保护（refactor/截断不得动）
        streak = _update_streaks({}, manual)
        ref_score = _score_batch(probe, manual, aware, direct)
        reject_run = 0
        window = 0
        processed = 0
        print(f"[init] probe基准={ref_score} manual={manual.summary()}", flush=True)

    def save_progress(complete=False):
        _atomic_json(state_path, {
            "version": 1,
            "complete": bool(complete),
            "completed_windows": window,
            "processed": processed,
            "ref_score": ref_score,
            "reject_run": reject_run,
            "streak": _serialize_streak(streak),
            "interval": interval,
            "epochs": epochs,
            "max_examples": max_examples,
            "seed": seed,
            "direct": direct,
            "aware": aware,
        })

    def _infer_window(batch, skill_text):
        """窗口内推理，返回 [(rec, result), ...]。

        direct 形态（E4）：直答优先 + 1-2 句视觉证据，与部署时的直答使用方式对齐，
        证据句仅供反思归因。"""
        cot = not direct
        jobs = [D.build_job(rec, skill_text=skill_text, cot=cot, evidence=direct)
                for rec in batch]
        if C.EXEC_BACKEND == "api":
            workdir = os.path.join(out_dir, "_exec", f"w{window + 1}")
            res = _run_api_window_checkpointed(
                jobs, gpus=gpus, tag=f"{exp}_w{window + 1}", workdir=workdir,
                enable_thinking=enable_thinking)
            return [(rec, res.get(rec["id"], {})) for rec in batch]
        if serial:
            eps = E.service_endpoints()
            if eps:
                # 常驻服务并发快路径；统一走 run_executor 以复用模型校验和恢复等待。
                res = E.run_executor(jobs, enable_thinking=enable_thinking,
                                     use_service=True)
                return [(rec, res.get(rec["id"], {})) for rec in batch]
            pairs = []
            for rec, job in zip(batch, jobs):
                pairs.append((rec, E.infer_one(job, enable_thinking=enable_thinking)))
            return pairs
        results = E.run_executor(jobs, gpus=gpus, tag=f"{exp}_w{window + 1}",
                                 enable_thinking=enable_thinking,
                                 workdir=os.path.join(out_dir, "_exec", f"w{window + 1}"))
        return [(rec, results.get(rec["id"], {})) for rec in batch]

    for batch_index, batch in enumerate(batches):
        if batch_index < window:
            continue
        # manual 窗口内冻结：注入文本对整窗一致，算一次
        skill_text = R.inject_text(manual, aware=aware)
        pairs = _infer_window(batch, skill_text)
        processed += len(batch)

        correct = sum(1 for rec, r in pairs if r.get("answer") == rec.get("answer"))
        acc = round(100.0 * correct / max(1, len(pairs)), 2)

        # 窗口内逐样本 LLM 反思（库冻结，可并发）
        reflections = L.parallel_map(
            pairs, lambda pr: RF.reflect(manual, pr[0], pr[1], aware=aware))
        buffer = [rf for rf in reflections if rf]

        # 窗口末 LLM 修订 manual（候选），探针棘轮裁决接受/拒绝
        protected = [(t, txt) for _k, (c, t, txt) in streak.items()
                     if c >= C.PROTECT_STREAK]
        candidate = RV.revise_manual(manual, buffer, aware=aware, protected=protected)
        cand_score = _score_batch(probe, candidate, aware, direct)
        accepted = cand_score >= ref_score - C.EVOLVE_PROBE_EPS
        if accepted:
            manual = candidate
            ref_score = cand_score
            reject_run = 0
        else:
            reject_run += 1
            print(f"[ratchet] 拒绝修订：probe {cand_score} < {ref_score} - "
                  f"{C.EVOLVE_PROBE_EPS}（连拒 {reject_run}）", flush=True)
            if reject_run >= C.EVOLVE_REJECT_REFRESH:
                ref_score = _score_batch(probe, manual, aware, direct)
                reject_run = 0
                print(f"[ratchet] 连拒达限，重测基准={ref_score}", flush=True)
        streak = _update_streaks(streak, manual)
        window += 1
        rev_info = {"reflections": len(buffer),
                    "by_type": _count_types(buffer),
                    "probe_cand": cand_score, "probe_ref": ref_score,
                    "accepted": accepted, "n_protected": len(protected)}
        # 周期性 val 监控（仅记录进化进展，不参与任何选择，终点即交付）
        if window % C.EVOLVE_VAL_EVERY == 0:
            rev_info["val_acc"] = _score_batch(val_recs, manual, aware, direct)
            print(f"[monitor] win{window} val_acc={rev_info['val_acc']}", flush=True)
        _snapshot(manual, out_dir, window, acc, rev_info, processed, log_path)
        save_progress()

    # 终点：探针逐类 4 格搜索二维路由 policy，双守卫防噪，写回 skills.json（与手册绑定）
    # 探针不足或增益未越过 margin 时，一律回退到无手册、直答的基线格。
    default_cell = "base"
    table, ncat = _score_cells_by_cat(probe, manual, aware)
    policy = _search_policy(table, ncat, default_cell,
                            C.POLICY_SEARCH_MARGIN, C.POLICY_MIN_PROBE)
    manual.state["policy"] = policy
    manual.state["policy_default"] = default_cell
    manual.save(os.path.join(out_dir, "skills.json"))
    with open(os.path.join(out_dir, "policy.json"), "w", encoding="utf-8") as f:
        json.dump({"policy": policy, "policy_default": default_cell,
                   "cell_table": table, "probe_by_cat": ncat,
                   "margin": C.POLICY_SEARCH_MARGIN, "min_probe": C.POLICY_MIN_PROBE},
                  f, ensure_ascii=False, indent=2)
    save_progress(complete=True)
    print(f"[policy] endpoint search default={default_cell} -> {policy}", flush=True)
    print(f"[policy] cell_table={table} probe_by_cat={ncat}", flush=True)

    print("evolution done ->", out_dir, flush=True)
    return out_dir


def _count_types(buffer):
    out = {}
    for rf in buffer:
        t = rf.get("reflection_type", "?")
        out[t] = out.get(t, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    help="阶段名（默认写入 runs/<executor>_<date>_vNNN/<exp>/）")
    ap.add_argument("--gpus", default="1", help="仅并行推理时用；串行走常驻服务")
    ap.add_argument("--interval", type=int, default=C.B_INTERVAL,
                    help="修订间隔 B（演化粒度，窗口大小）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--parallel", action="store_true", help="窗口内并发推理加速（默认串行）")
    ap.add_argument("--unaware", action="store_true", help="skill-unaware 消融")
    ap.add_argument("--direct", action="store_true",
                    help="E4：执行器直答+短证据（与 direct 部署形态对齐），默认 CoT")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-skills", default="")
    ap.add_argument("--max-examples", type=int, default=0, help="限制 train 规模（调试）")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    try:
        evolve(args.exp, gpus=gpus, interval=args.interval, epochs=args.epochs,
               aware=(not args.unaware), seed=args.seed,
               init_skills=args.init_skills or None, max_examples=args.max_examples,
               serial=(not args.parallel), enable_thinking=args.thinking,
               direct=args.direct)
    except E.ExecutorInterrupted as e:
        print(f"[evolve] interrupted: {e}. Update the API settings, then re-run with the "
              "same RUN_ID and command arguments to resume.", flush=True)
        raise SystemExit(130) from None
    except E.ExecutorAPIError as e:
        print(f"[evolve] API failed: {e}. Completed work is checkpointed; update the API "
              "settings, then re-run with the same RUN_ID and command arguments.", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
