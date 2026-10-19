#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import defaultdict

from . import config as C


def _acc(correct, total):
    return round(100.0 * correct / total, 2) if total else 0.0


def _jsd(p_counts, q_counts):
    """两个计数分布的 Jensen–Shannon 散度（log2，值域 [0, 1]）。"""
    keys = set(p_counts) | set(q_counts)
    if not keys:
        return 0.0
    P = sum(p_counts.values()) or 1
    Q = sum(q_counts.values()) or 1
    jsd = 0.0
    for k in keys:
        p = p_counts.get(k, 0) / P
        q = q_counts.get(k, 0) / Q
        m = (p + q) / 2.0
        if p > 0:
            jsd += 0.5 * p * math.log2(p / m)
        if q > 0:
            jsd += 0.5 * q * math.log2(q / m)
    return round(jsd, 4)


def _n_tokens(tokenizer, text):
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def compute_metrics(records, results, tokenizer=None):
    """records: 原始记录列表；results: {id: {answer, raw, ...}}。

    tokenizer: 与执行器一致的 tokenizer，用于把 raw 输出折算为生成 token 数；
      未提供时，如果结果中带有 vLLM/OpenAI ``usage.completion_tokens``
      对应的 ``gen_tokens``，则直接使用该精确计数，否则效率轴字段置 None。
    """
    by = {
        "relation": defaultdict(lambda: [0, 0]),
        "viewpoint": defaultdict(lambda: [0, 0]),
        "numimg": defaultdict(lambda: [0, 0]),
    }
    # CvSI 逐类累加器：acc 分子分母 + 生成 token 累计
    cvsi = defaultdict(lambda: {"correct": 0, "total": 0,
                                "tok_sum": 0, "tok_cnt": 0})
    n = correct = answered = 0
    tok_total = 0
    has_result_token_usage = tokenizer is None and any(
        isinstance((r or {}).get("gen_tokens"), (int, float))
        for r in results.values()
    )
    pred_pos = defaultdict(int)   # 预测选项字母分布（仅作答成功的题）
    gold_pos = defaultdict(int)   # 金标选项字母分布
    pos_cond = defaultdict(lambda: [0, 0])  # 金标字母 -> [答对, 全部] 条件准确率
    for rec in records:
        n += 1
        r = results.get(rec["id"])
        pred = r.get("answer") if r else None
        gold = rec["answer"]
        gold_pos[gold] += 1
        ok = int(pred == gold)
        pos_cond[gold][1] += 1
        pos_cond[gold][0] += ok
        if pred is not None:
            answered += 1
            pred_pos[pred] += 1
        correct += ok
        cs = cvsi[rec.get("cvsi_category")]
        cs["total"] += 1
        cs["correct"] += ok
        if tokenizer is not None or has_result_token_usage:
            if tokenizer is not None:
                nt = _n_tokens(tokenizer, (r or {}).get("raw") or "")
            else:
                # Missing/failed requests count as zero, matching the
                # tokenizer path where their raw output is an empty string.
                nt = (r or {}).get("gen_tokens") or 0
            tok_total += nt
            cs["tok_sum"] += nt
            cs["tok_cnt"] += 1
        for field, key in (
            ("relation", rec.get("relation_expr")),
            ("viewpoint", rec.get("viewpoint_dynamic")),
            ("numimg", str(rec.get("num_images"))),
        ):
            slot = by[field][key]
            slot[0] += ok
            slot[1] += 1

    def pack(d):
        return {k: {"acc": _acc(v[0], v[1]), "n": v[1]} for k, v in sorted(d.items())}

    # CvSI 逐类：acc + 平均生成 token + 样本数
    cvsi_pack = {}
    for k, v in sorted(cvsi.items()):
        item = {"acc": _acc(v["correct"], v["total"]), "n": v["total"]}
        item["avg_tokens"] = round(v["tok_sum"] / v["tok_cnt"], 1) if v["tok_cnt"] else None
        cvsi_pack[k] = item
    macro = [cvsi_pack[k]["acc"] for k in C.CVSI_LABELS if k in cvsi_pack]
    cvsi_macro = round(sum(macro) / len(macro), 2) if macro else 0.0

    # 偏差轴：逐金标位置条件准确率 + 极差
    pos_acc_by_gold = {k: _acc(v[0], v[1]) for k, v in sorted(pos_cond.items())}
    accs = list(pos_acc_by_gold.values())
    pos_acc_range = round(max(accs) - min(accs), 2) if accs else 0.0

    return {
        "n": n,
        "answered": answered,
        "parse_fail": n - answered,
        "overall_acc": _acc(correct, n),
        "cvsi_macro_acc": cvsi_macro,
        # 效率轴：每题平均生成 token（总体；逐类在 by_cvsi[*].avg_tokens）
        "avg_gen_tokens": (
            round(tok_total / n, 1)
            if (tokenizer is not None or has_result_token_usage) else None
        ),
        "by_cvsi": cvsi_pack,
        "by_relation": pack(by["relation"]),
        "by_viewpoint": pack(by["viewpoint"]),
        "by_numimg": pack(by["numimg"]),
        # 偏差轴：位置偏置两形式
        "pred_pos_dist": dict(pred_pos),
        "gold_pos_dist": dict(gold_pos),
        "pos_bias_jsd": _jsd(pred_pos, gold_pos),
        "pos_acc_by_gold": pos_acc_by_gold,
        "pos_acc_range": pos_acc_range,
    }


def format_report(m):
    tok = m.get("avg_gen_tokens")
    lines = [
        f"n={m['n']} answered={m['answered']} parse_fail={m['parse_fail']}",
        f"OVERALL acc = {m['overall_acc']}   CvSI-macro acc = {m['cvsi_macro_acc']}"
        f"   avg_gen_tokens = {tok}",
        f"pos_bias: JSD = {m['pos_bias_jsd']}   acc-range-by-gold-pos = {m['pos_acc_range']}",
        "by CvSI:  " + "  ".join(
            f"{k}[acc={v['acc']} tok={v.get('avg_tokens')} n{v['n']}]"
            for k, v in m["by_cvsi"].items()),
        "by rel:   " + "  ".join(f"{k}={v['acc']}(n{v['n']})" for k, v in m["by_relation"].items()),
        "by view:  " + "  ".join(f"{k}={v['acc']}(n{v['n']})" for k, v in m["by_viewpoint"].items()),
        "pred_pos: " + str(m["pred_pos_dist"]),
    ]
    return "\n".join(lines)
