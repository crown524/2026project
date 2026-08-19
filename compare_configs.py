"""配对比较两个实验配置：差值的 bootstrap 置信区间。

为什么需要这个脚本：两个配置的 CI 重叠**不等于**差异不显著，反之亦然。
判断"B1 是否真的强于 A2"必须对**差值本身**做区间估计，且必须在同一批
受试者上配对——两个配置各自的 CI 分别包含了受试者抽样波动，直接比会把
共同波动重复计入，掩盖真实差异。

依赖 run_experiment.py 写入的 per_subject 逐人预测。若结果文件里没有该字段
（旧版本产出），需用当前代码重跑对应实验。

用法:
    python compare_configs.py A.json B.json
    python compare_configs.py A.json B.json --label "LOSO"   # 只比某个协议
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import config as C
import eval_utils as E


def pick(data: dict, want: str | None) -> dict | None:
    """挑出带 per_subject 的结果行；want 为标签子串筛选。"""
    cands = [r for r in data.get("results", []) if r.get("per_subject")]
    if want:
        cands = [r for r in cands if want in r.get("label", "")]
    return cands[0] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--label", help="只比标签含此子串的行，如 LOSO / dev")
    ap.add_argument("--n-boot", type=int, default=C.N_BOOTSTRAP)
    args = ap.parse_args()

    out = []
    for f in (args.file_a, args.file_b):
        p = Path(f)
        if not p.is_absolute():
            p = C.RESULT_DIR / f
        if not p.exists():
            print(f"找不到: {p}")
            return 1
        data = json.loads(p.read_text(encoding="utf-8"))
        row = pick(data, args.label)
        if row is None:
            print(f"{p.name} 中没有含 per_subject 的结果行"
                  f"{'（标签筛选: ' + args.label + '）' if args.label else ''}。"
                  "\n该文件可能由旧版脚本产出，请用当前 run_experiment.py 重跑。")
            return 1
        out.append((p.name, data, row))

    (na, da, ra), (nb, db, rb) = out
    labels = E.load_labels()
    truth = dict(zip(labels[C.COL_PARTICIPANT_ID].astype(str),
                     labels[C.COL_PHQ8_SCORE].astype(float)))

    ids = sorted(set(ra["per_subject"]) & set(rb["per_subject"]) & set(truth))
    if len(ids) < 10:
        print(f"共同受试者仅 {len(ids)} 人，不足以比较。")
        return 1

    y = np.array([truth[i] for i in ids])
    pa = np.array([ra["per_subject"][i] for i in ids])
    pb = np.array([rb["per_subject"][i] for i in ids])

    print("=" * 62)
    print("配对比较（同一批受试者，差值的 bootstrap 95% CI）")
    print("=" * 62)
    print(f"  A: {'+'.join(da.get('lanes', []))}  [{ra['label']}]  ({na})")
    print(f"  B: {'+'.join(db.get('lanes', []))}  [{rb['label']}]  ({nb})")
    print(f"  共同受试者: {len(ids)} 人")

    rho_a, rho_b = spearmanr(y, pa)[0], spearmanr(y, pb)[0]
    rmse = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))

    # 配对 bootstrap：每次重采样受试者，两个配置用同一批索引，保留配对结构
    rng = np.random.default_rng(C.RANDOM_SEED)
    d_rho, d_rmse = [], []
    for _ in range(args.n_boot):
        idx = rng.integers(0, len(ids), len(ids))
        if len(np.unique(y[idx])) < 2:
            continue
        s_a, s_b = spearmanr(y[idx], pa[idx])[0], spearmanr(y[idx], pb[idx])[0]
        if not (np.isnan(s_a) or np.isnan(s_b)):
            d_rho.append(s_b - s_a)
        d_rmse.append(np.sqrt(np.mean((y[idx] - pb[idx]) ** 2))
                      - np.sqrt(np.mean((y[idx] - pa[idx]) ** 2)))

    def report(name: str, va: float, vb: float, diffs: list,
               lower_better: bool) -> None:
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"\n  {name}")
        print(f"    A = {va:.3f}   B = {vb:.3f}   差值(B-A) = {vb - va:+.3f}")
        print(f"    差值 95% CI [{lo:+.3f}, {hi:+.3f}]")
        if lo <= 0 <= hi:
            print("    → 区间跨零：**差异不显著**，不能声称哪个更好")
        else:
            better = "A" if (lo > 0) == lower_better else "B"
            print(f"    → 区间不跨零：{better} 显著更好")

    report("Spearman ρ（越大越好）", rho_a, rho_b, d_rho, lower_better=False)
    report("RMSE（越小越好）", rmse(pa), rmse(pb), d_rmse, lower_better=True)

    print("\n" + "=" * 62)
    print("注：配置差异不显著时，应按可解释性、成本、工程复杂度取舍，")
    print("    而不是按小数点后第二位的精度差异下结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
