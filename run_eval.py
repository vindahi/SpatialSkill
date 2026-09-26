#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测驱动：构建 jobs -> 多卡执行器推理 -> 计算指标 -> 落盘。

modes:
  no_skill : 零样本直接答（无 CoT）
  cot      : 零样本思维链
  skill    : 注入进化好的技能手册 Manual（需 --skills 指向 skills.json）
             skill-aware  注入 sections + execution notes；
             skill-unaware（--unaware）只注入 sections（消融执行提醒）；
             --routed  按题目 cvsi_category 路由的减量注入（每题 <=8 行）；
             --direct  手册+直接作答（无 CoT，消融手册作为先验偏置的作用）。
默认注入对齐 EmbodiSkill：整册 manual 一次性注入，manual 在评测期间冻结。
"""
import argparse
import json
import os

from . import config as C
from . import data as D
from . import metrics as M
from . import executor as E
from . import retrieval as R
from . import manual as MAN


def build_jobs(records, mode, skill_text=None, manual=None, routed=False,
               aware=True, direct=False, cot_cats=None, policy=None,
               policy_default="base"):
    """cot_cats: 一组 cvsi_category，这些类别即使 direct=True 也强制保留 CoT。
    用于"按类别路由推理模式"——序贯推理类（如 MR/WK）保留 CoT，直觉类走 direct。

    policy: dict[cvsi_category -> cell]，二维路由（推理模式 × 手册注入）。
      cell ∈ {base, cot, skill_direct, skill_cot}：
        base        = 直答 + 不注入手册（等价 no_skill）
        cot         = CoT + 不注入手册（等价 cot 基线）
        skill_direct= 直答 + 注入手册
        skill_cot   = CoT  + 注入手册
      提供 policy 时逐题按类别查表决定 (cot, 是否注入)，覆盖 direct/cot_cats 逻辑；
      未列出的类别用 policy_default。此为跨模型规模自适应路由的落地形式。
    """
    cot_cats = set(cot_cats or [])
    jobs = []
    for rec in records:
        if policy:
            cell = policy.get(rec.get("cvsi_category"), policy_default)
            st = skill_text if cell in ("skill_direct", "skill_cot") else None
            cot = cell in ("cot", "skill_cot")
            jobs.append(D.build_job(rec, skill_text=st, cot=cot))
            continue
        st = None
        if mode == "skill":
            st = (R.inject_text_routed(manual, rec, aware=aware) if routed
                  else skill_text)
        cot = (mode != "no_skill") and not direct
        # 类别级推理模式路由：命中 cot_cats 的题强制走 CoT（覆盖全局 direct）
        if mode == "skill" and direct and rec.get("cvsi_category") in cot_cats:
            cot = True
        jobs.append(D.build_job(rec, skill_text=st, cot=cot))
    return jobs


def _load_prediction_checkpoint(path, valid_ids):
    """读取逐条追加的预测 checkpoint；截断或损坏的最后一行不阻断续跑。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--mode", default="cot", choices=["no_skill", "cot", "skill"])
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（pilot）")
    ap.add_argument("--gpus", default=",".join(str(g) for g in C.EXECUTOR_GPUS))
    ap.add_argument("--skills", default="", help="skill 模式下进化好的 skills.json（Manual）路径")
    ap.add_argument("--unaware", action="store_true",
                    help="skill-unaware 消融：只注入 sections，不注入执行提醒 appendix")
    ap.add_argument("--routed", action="store_true",
                    help="按题目 cvsi_category 路由的减量注入（每题 <=8 行）")
    ap.add_argument("--direct", action="store_true",
                    help="skill 模式下直接作答（无 CoT），消融手册作为先验偏置的作用")
    ap.add_argument("--cot-cats", default="",
                    help="逗号分隔的 cvsi_category，这些类别在 --direct 下仍强制保留 CoT（如 MR,WK）")
    ap.add_argument("--policy", default="",
                    help="按类别二维路由，如 'MR:skill_cot,WK:cot,PT:base,SR:cot,CR:skill_direct'；"
                         "cell ∈ {base,cot,skill_direct,skill_cot}。设置后覆盖 --direct/--cot-cats。")
    ap.add_argument("--policy-default", default="base",
                    help="未在 --policy 中列出的类别使用的默认 cell")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shuffle-choices", action="store_true", help="打乱选项去偏（暂留接口）")
    args = ap.parse_args()

    C.ensure_dirs()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    records = D.load_records(args.split, limit=args.limit)

    skill_text = None
    manual = None
    if args.mode == "skill":
        if not (args.skills and os.path.exists(args.skills)):
            raise SystemExit(f"skill 模式需要 --skills 指向进化好的 Manual：{args.skills!r} 不存在")
        manual = MAN.Manual.load(args.skills)
        if not args.routed:
            skill_text = R.inject_text(manual, aware=(not args.unaware))
        print(f"[eval] loaded manual {manual.summary()} aware={not args.unaware} "
              f"routed={args.routed} direct={args.direct}", flush=True)
        if not args.routed and skill_text is None:
            print("[eval] 警告：manual 为空，skill 模式退化为纯 CoT。", flush=True)

    _VALID_CELLS = {"base", "cot", "skill_direct", "skill_cot"}
    policy = {}
    for kv in args.policy.split(","):
        kv = kv.strip()
        if not kv:
            continue
        c, cell = kv.split(":")
        c, cell = c.strip(), cell.strip()
        if cell not in _VALID_CELLS:
            raise SystemExit(f"--policy 非法 cell {cell!r}，合法值：{sorted(_VALID_CELLS)}")
        policy[c] = cell
    if args.policy_default not in _VALID_CELLS:
        raise SystemExit(f"--policy-default 非法：{args.policy_default!r}")
    policy_default = args.policy_default
    # 未显式传 --policy 时，自动读取与手册绑定的 policy（进化终点探针搜索写入 skills.json）
    if not policy and manual is not None and isinstance(manual.state.get("policy"), dict):
        for c, cell in manual.state["policy"].items():
            if cell in _VALID_CELLS:
                policy[c] = cell
        pd = manual.state.get("policy_default")
        if pd in _VALID_CELLS:
            policy_default = pd
        if policy:
            print(f"[eval] auto-loaded policy from skills.json: {policy} "
                  f"(default={policy_default})", flush=True)

    cot_cats = [x.strip() for x in args.cot_cats.split(",") if x.strip()]
    jobs = build_jobs(records, args.mode, skill_text=skill_text, manual=manual,
                      routed=args.routed, aware=(not args.unaware),
                      direct=args.direct, cot_cats=cot_cats,
                      policy=policy or None, policy_default=policy_default)
    tag = args.tag or f"{args.split}_{args.mode}"
    # 每次启动由 config 固定一个运行根目录；同一运行内不同轮次仍以 tag 分目录。
    out_dir = C.resolve_run_output(tag)
    os.makedirs(out_dir, exist_ok=True)
    preds_path = os.path.join(out_dir, "preds.jsonl")
    job_ids = {job["id"] for job in jobs}
    results = _load_prediction_checkpoint(preds_path, job_ids)
    pending_jobs = [job for job in jobs if job["id"] not in results]
    if results:
        print(f"[eval] resumed checkpoint: {len(results)}/{len(jobs)} completed; "
              f"{len(pending_jobs)} remaining", flush=True)

    if pending_jobs:
        # API 模式在每个响应完成后从主线程追加并 fsync；Ctrl-C 或单题最终失败后，
        # 已完成条目仍可在同一 RUN_ID/tag 下被安全跳过。
        with open(preds_path, "a", encoding="utf-8") as checkpoint:
            def save_result(result):
                rid = result["id"]
                if rid in results:
                    return
                checkpoint.write(json.dumps(result, ensure_ascii=False) + "\n")
                checkpoint.flush()
                os.fsync(checkpoint.fileno())
                results[rid] = result

            try:
                fresh = E.run_executor(
                    pending_jobs, gpus=gpus, tag=tag, enable_thinking=args.thinking,
                    workdir=os.path.join(out_dir, "_exec"), result_sink=save_result)
            except E.ExecutorInterrupted as e:
                print(f"[eval] interrupted: {e}. Re-run with the same RUN_ID and --tag to resume.",
                      flush=True)
                raise SystemExit(130) from None
            except E.ExecutorAPIError as e:
                print(f"[eval] API failed: {e}. Completed results were checkpointed; "
                      "fix the API configuration and re-run with the same RUN_ID and --tag.",
                      flush=True)
                raise SystemExit(1) from None
            for result in fresh.values():
                save_result(result)

    if len(results) != len(jobs):
        missing = len(jobs) - len(results)
        raise SystemExit(f"[eval] incomplete checkpoint: {missing} jobs remain; metrics were not written")

    # 效率轴（avg_gen_tokens）需要与执行器一致的 tokenizer 折算 raw 输出
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tok_path = C.MODEL_PATHS.get(C.SERVED_MODEL_NAME, C.EXECUTOR_MODEL)
        if not os.path.isdir(tok_path):
            tok_path = C.EXECUTOR_MODEL
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    except Exception as e:  # tokenizer 失败不阻断评测，效率轴置空
        print(f"[eval] 警告：tokenizer 加载失败（{e}），效率轴指标置空", flush=True)

    m = M.compute_metrics(records, results, tokenizer=tokenizer)
    m["mode"] = args.mode
    m["split"] = args.split
    if args.mode == "skill":
        m["skill_aware"] = (not args.unaware)
        m["skills_path"] = args.skills
        m["routed"] = args.routed
        m["direct"] = args.direct
        m["cot_cats"] = cot_cats
        m["policy"] = policy
        m["policy_default"] = policy_default
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

    print(f"=== {tag} ===")
    print(M.format_report(m))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
