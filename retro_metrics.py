#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""retro_metrics.py —— 对已完成的 run 事后补算正式三指标（零 GPU）。

从 runs/<tag>/preds.jsonl 重算 acc / avg_gen_tokens / Option Position Bias，
重写 metrics.json（保留原有 mode/policy 等元数据），并删除已撤销的
op_coverage / answer_rate 字段。用于主表重跑产物的指标体系切换。

用法：
  python -m skillevo.retro_metrics                          # 默认补算主表+修正policy全部run
  python -m skillevo.retro_metrics --runs "runs/qwen3_5_9b_*_r*"
"""
import argparse
import glob
import json
import os

from . import config as C
from . import data as D
from . import metrics as M

# tag 前缀 -> 生成该 run 的执行器模型（tokenizer 必须与生成模型一致）
MODEL_PREFIXES = ["qwen3_5_9b", "qwen3_vl_8b_instruct", "qwen3_vl_4b_instruct"]
# 重写 metrics.json 时需要保留的元数据键（其余由重算覆盖）
META_KEYS = ["mode", "split", "skill_aware", "skills_path", "routed", "direct",
             "cot_cats", "policy", "policy_default"]
DEFAULT_PATTERNS = ["runs/qwen3_vl_8b_instruct_*_r*", "runs/qwen3_5_9b_*_r*"]


def pick_tokenizer(tag, cache):
    """按 tag 前缀选执行器 tokenizer（带缓存，同规模只加载一次）。"""
    model = next((m for m in MODEL_PREFIXES if tag.startswith(m)), None)
    if model is None:
        return None
    if model not in cache:
        from transformers import AutoTokenizer
        cache[model] = AutoTokenizer.from_pretrained(
            C.MODEL_PATHS.get(model, os.path.join(C.MODEL_DIR, model)),
            trust_remote_code=True)
        print(f"[retro] loaded tokenizer: model/{model}", flush=True)
    return cache[model]


def retro_one(run_dir, records_by_id, tok_cache, dry_run=False):
    tag = os.path.basename(run_dir.rstrip("/"))
    preds_fp = os.path.join(run_dir, "preds.jsonl")
    met_fp = os.path.join(run_dir, "metrics.json")
    if not (os.path.exists(preds_fp) and os.path.exists(met_fp)):
        print(f"[retro] skip {tag}（缺 preds.jsonl 或 metrics.json）")
        return None
    old = json.load(open(met_fp, encoding="utf-8"))
    results = {}
    for line in open(preds_fp, encoding="utf-8"):
        r = json.loads(line)
        results[r["id"]] = r
    if set(results) - set(records_by_id):
        print(f"[retro] skip {tag}（preds 含未知 id，split 不匹配）")
        return None
    records = [records_by_id[i] for i in results]
    tokenizer = pick_tokenizer(tag, tok_cache)
    m = M.compute_metrics(records, results, tokenizer=tokenizer)
    for k in META_KEYS:           # 保留评测配置元数据
        if k in old:
            m[k] = old[k]
    if dry_run:
        return m
    with open(met_fp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"[retro] {tag:32s} acc={m['overall_acc']:5.2f} "
          f"tok={m['avg_gen_tokens']} jsd={m['pos_bias_jsd']} "
          f"range={m['pos_acc_range']}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_PATTERNS,
                    help="run 目录 glob 列表")
    ap.add_argument("--split", default="val")
    ap.add_argument("--dry-run", action="store_true", help="只重算不落盘")
    args = ap.parse_args()

    records_by_id = {r["id"]: r for r in D.load_records(args.split)}
    tok_cache = {}
    dirs = []
    for pat in args.runs:
        dirs += sorted(glob.glob(pat))
    dirs = sorted(set(d for d in dirs if os.path.isdir(d)
                      and not d.rstrip("/").endswith("_service")))
    print(f"[retro] {len(dirs)} run dirs matched")
    for d in dirs:
        retro_one(d + "/", records_by_id, tok_cache, dry_run=args.dry_run)
    print("[retro] done" + ("（dry-run，未写盘）" if args.dry_run else ""))


if __name__ == "__main__":
    main()
