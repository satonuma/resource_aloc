"""
FY2029 FTE算出モジュール
========================
売上予測を実現するために必要なMR数（FTE）と
MR/デジタル比率を品目×月別に算出する。

対象期間: FY2029（2029年4月 〜 2030年3月）
対象領域:
  CS    （希少疾患以外, 2.5 call/day, 380名）
  PS遺伝（希少疾患・遺伝, 1.5 call/day,  25名）
  PS血液（希少疾患・血液, 1.5 call/day,  40名）

品目グループ（FY2029）:
  PDT    : GLI, CUV, HYQ, TAK881（血漿分画製剤）
  NS     : INT, TRI, ENT
  OVE    : OVE
  NEW_CS : Zaso, TAK881（CS新規発売品）
  CV     : REV, ALC
  RS     : VYV, VPR
  PS遺伝 : LVM, TKZ, VPR, RPL, FIR
  PS血液 : VON, LIV, FEI, OBI, ADV, ADY, Meza

FC/SC構造:
  - ファーストコール（FC）: 主訪問、フルFTEコスト
  - セカンドコール（SC）: FC訪問に内包、FC × SC_COEFFICIENT（=0.1）のコスト

Databricks Python環境で動作する設計。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# 定数
# ============================================================

WORKING_DAYS_PER_MONTH = 20

CALLS_PER_DAY: Dict[str, float] = {
    "CS":     2.5,
    "PS遺伝": 1.5,
    "PS血液": 1.5,
}

CURRENT_MR_COUNT: Dict[str, int] = {
    "CS":     380,
    "PS遺伝":  25,
    "PS血液":  40,
}

# SC訪問はFC訪問に内包されるが、このコスト係数を乗じてFTEを計上
SC_COEFFICIENT = 0.1

# 品目グループ（全製品）
PRODUCT_GROUPS: Dict[str, List[str]] = {
    "PDT":    ["GLI", "CUV", "HYQ", "TAK881"],  # 血漿分画製剤
    "NS":     ["INT", "TRI", "ENT"],
    "OVE":   ["OVE"],
    "NEW_CS": ["Zaso", "TAK881"],     # CS新規発売品
    "CV":    ["REV", "ALC"],
    "RS":    ["VYV"],
    "PS遺伝": ["LVM", "TKZ", "VPR", "RPL", "FIR"],           # VPR: HAE製剤、TKZとセット
    "PS血液": ["VON", "LIV", "FEI", "OBI", "ADV", "ADY", "Meza"],
}

# 品目→領域
PRODUCT_AREA: Dict[str, str] = {
    # CS（希少疾患以外）
    "GLI": "CS", "CUV": "CS", "HYQ": "CS", "TAK881": "CS",
    "INT": "CS", "TRI": "CS", "ENT": "CS",
    "OVE": "CS",
    "Zaso": "CS",
    "REV": "CS", "ALC": "CS",
    "VYV": "CS",
    # PS遺伝（希少疾患・遺伝）
    "LVM": "PS遺伝", "TKZ": "PS遺伝", "VPR": "PS遺伝",  # VPR: HAE製剤
    "RPL": "PS遺伝", "FIR": "PS遺伝",
    # PS血液（希少疾患・血液）
    "VON": "PS血液", "LIV": "PS血液", "FEI": "PS血液",
    "OBI": "PS血液", "ADV": "PS血液", "ADY": "PS血液", "Meza": "PS血液",
}

# FY開始年 → 12ヶ月リストを生成するユーティリティ
def get_fy_months(fy_start: int) -> List[str]:
    """FY開始年(例: 2026) → ["2026-04", ..., "2027-03"]"""
    return (
        [f"{fy_start}-{m:02d}" for m in range(4, 13)]
        + [f"{fy_start + 1}-{m:02d}" for m in range(1, 4)]
    )


def _parse_ym(ym: str) -> datetime:
    """年月文字列を解析し datetime オブジェクトを返す。"""
    value = str(ym).strip()
    for fmt in ("%Y-%m", "%y-%m", "%b-%y", "%b-%Y", "%Y/%m", "%Y.%m"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Invalid year-month format: '{ym}'. Expected 'YYYY-MM' or 'Apr-27' style.")


def month_to_fy(month: str) -> str:
    """年月文字列をFY文字列に変換 (例: "2026-05" → "FY2026", "2027-01" → "FY2026")"""
    dt = _parse_ym(month)
    fy = dt.year if dt.month >= 4 else dt.year - 1
    return f"FY{fy}"


# FY2029の月リスト（後方互換のため保持）
FY2029_MONTHS: List[str] = get_fy_months(2029)


def apply_product_fte_caps(
    fte_df: pd.DataFrame,
    fte_caps: Dict[str, float],
) -> pd.DataFrame:
    """
    品目別FTE上限を適用し、超過分を同エリア内の非キャップ品目に比例再配分する。

    Parameters
    ----------
    fte_df   : normalize_fte_to_headcount / discretize_fte_semiannually 出力
    fte_caps : {product_id: max_fte} のdict（上限を設定する品目のみ）
    """
    if not fte_caps:
        return fte_df

    fte_col = "adjusted_fte" if "adjusted_fte" in fte_df.columns else "required_fte"
    result = fte_df.copy()

    for month in result["month"].unique():
        for area in result["area"].unique():
            mask = (result["month"] == month) & (result["area"] == area)
            if not mask.any():
                continue

            idx = result[mask].index
            excess = 0.0
            for i in idx:
                pid = result.at[i, "product_id"]
                if pid in fte_caps:
                    cap = fte_caps[pid]
                    val = result.at[i, fte_col]
                    if val > cap:
                        excess += val - cap
                        result.at[i, fte_col] = cap

            if excess > 0:
                uncapped_idx = [i for i in idx if result.at[i, "product_id"] not in fte_caps]
                uncapped_total = result.loc[uncapped_idx, fte_col].sum()
                if uncapped_total > 0:
                    for i in uncapped_idx:
                        result.at[i, fte_col] += excess * result.at[i, fte_col] / uncapped_total

    result[fte_col] = result[fte_col].round(2)
    result["adjusted_mr_fte"] = result[fte_col].round(2)
    if fte_col != "adjusted_fte":
        result["adjusted_fte"] = result[fte_col].round(2)

    return result


def normalize_fte_roi_weighted(
    fte_df: pd.DataFrame,
    headcount_targets: Dict[str, int],
    decay_params_df: "pd.DataFrame",
    combined_mode: bool = False,
) -> "pd.DataFrame":
    """
    ROI最大化（Hill関数等限界収益）ベースのFTE正規化。

    【ロジック】
    キャップ不要（月別合計 <= ヘッドカウント目標）:
        → required_fte をそのまま維持（制約なし＝レスポンスカーブ不要）
    キャップあり（月別合計 > ヘッドカウント目標）:
        → MMMパラメータあり品目: Hill関数の等限界収益条件（dR/da = λ）で最適配分
           * 限界収益の高い品目（カーブの急峻な部分）を優先保護
           * 限界収益の低い品目（飽和域）から先にFTEを削減
        → MMMパラメータなし品目: グループ内で比例スケーリング（従来方式）
        → グループ間予算: required_fte比で按分

    Parameters
    ----------
    fte_df            : calculate() または run_with_dynamic_new_product() の出力
    headcount_targets : {"CS": 380, "PS遺伝": 25, "PS血液": 40}
    decay_params_df   : mmm_decay_params.csv（beta_m, ec_m, slope_m 必須）
    """
    fte_col = "adjusted_fte" if "adjusted_fte" in fte_df.columns else "required_fte"
    result  = fte_df.copy()

    # ---- Hill 関数ヘルパ（calculate_roi_optimal_fte と同じ実装）----
    def _marginal(a: float, beta: float, ec: float, slope: float) -> float:
        if a <= 0:
            return float("inf") if slope <= 1 else 0.0
        return beta * slope * (ec ** slope) * (a ** (slope - 1)) / (ec ** slope + a ** slope) ** 2

    def _find_activity(beta: float, ec: float, slope: float, target: float) -> float:
        """dR/da = target を満たす活動量 a（バイナリサーチ）"""
        if slope > 1:
            a_inflect = ec * ((slope - 1) / (slope + 1)) ** (1.0 / slope)
            if target > _marginal(a_inflect, beta, ec, slope):
                return 0.0
            a_lo, a_hi = a_inflect, ec * 500.0
        else:
            a_lo, a_hi = 1e-4, ec * 500.0
        for _ in range(100):
            mid = (a_lo + a_hi) / 2.0
            if _marginal(mid, beta, ec, slope) > target:
                a_lo = mid
            else:
                a_hi = mid
        return (a_lo + a_hi) / 2.0

    # ---- MR面談チャネルの Hillパラメータ収録 (pid → (beta, ec, slope)) ----
    mmm_params: Dict[str, Tuple[float, float, float]] = {}
    for pid in decay_params_df["product_id"].unique():
        rows = decay_params_df[
            (decay_params_df["product_id"] == pid)
            & (decay_params_df["channel"].isin(MR_CHANNELS))
        ]
        if rows.empty:
            continue
        face = rows[rows["channel"] == "面談"]
        p = face.iloc[0] if not face.empty else rows.iloc[0]
        mmm_params[pid] = (float(p["beta_m"]), float(p["ec_m"]), float(p["slope_m"]))

    # ---- 合計モード: CS＋PS遺伝＋PS血液を1プールとして正規化 ----
    if combined_mode:
        # 品目ごとの calls/month キャップ（領域別）
        product_cap: Dict[str, float] = {}
        for pid in result["product_id"].unique():
            row_area = result.loc[result["product_id"] == pid, "area"]
            a = row_area.iloc[0] if not row_area.empty else "CS"
            product_cap[pid] = CALLS_PER_DAY.get(a, 2.0) * WORKING_DAYS_PER_MONTH

        total_budget = float(sum(headcount_targets.values()))

        for month in sorted(result["month"].unique()):
            mask      = result["month"] == month
            total_req = result.loc[mask, fte_col].sum()
            if total_req <= 0:
                continue

            if abs(total_req - total_budget) <= 0.5:
                continue  # 誤差範囲内: スキップ

            if total_req < total_budget:
                # 合計 < 予算: 比例スケールアップ（445名を常にフル活用）
                scale = total_budget / total_req
                result.loc[mask, fte_col] = (result.loc[mask, fte_col] * scale).round(2)
                result.loc[mask, "adjusted_mr_fte"] = result.loc[mask, fte_col]
                if fte_col != "adjusted_fte":
                    result.loc[mask, "adjusted_fte"] = result.loc[mask, fte_col]
                continue

            sub      = result.loc[mask, ["product_id", fte_col]].set_index("product_id")[fte_col].to_dict()
            has_mmm  = {pid: mmm_params[pid] for pid in sub if pid in mmm_params and sub[pid] > 0}
            no_mmm   = [pid for pid in sub if pid not in mmm_params]
            req_mmm   = sum(sub[pid] for pid in has_mmm) if has_mmm else 0.0
            req_nommm = sum(sub[pid] for pid in no_mmm)  if no_mmm  else 0.0

            if not has_mmm:
                scale = total_budget / total_req
                result.loc[mask, fte_col] = (result.loc[mask, fte_col] * scale).round(2)
                result.loc[mask, "adjusted_mr_fte"] = result.loc[mask, fte_col]
                if fte_col != "adjusted_fte":
                    result.loc[mask, "adjusted_fte"] = result.loc[mask, fte_col]
                continue

            bgt_mmm   = total_budget * (req_mmm   / total_req)
            bgt_nommm = total_budget * (req_nommm / total_req)

            def _total_fte_comb(lam: float, _hmm=has_mmm, _pc=product_cap) -> float:
                return sum(
                    _find_activity(b, e, s, lam) / _pc.get(pid, 50.0)
                    for pid, (b, e, s) in _hmm.items()
                )

            log_lo, log_hi = np.log(1e-12), np.log(1e8)
            for _ in range(200):
                log_mid = (log_lo + log_hi) / 2.0
                if _total_fte_comb(float(np.exp(log_mid))) > bgt_mmm:
                    log_lo = log_mid
                else:
                    log_hi = log_mid
            lam_final = float(np.exp((log_lo + log_hi) / 2.0))

            for pid, (beta, ec, slope) in has_mmm.items():
                opt_fte = max(_find_activity(beta, ec, slope, lam_final) / product_cap.get(pid, 50.0), 0.0)
                m2 = mask & (result["product_id"] == pid)
                result.loc[m2, fte_col] = round(opt_fte, 2)

            if no_mmm and req_nommm > 0:
                scale_no = bgt_nommm / req_nommm
                for pid in no_mmm:
                    m2 = mask & (result["product_id"] == pid)
                    result.loc[m2, fte_col] = (result.loc[m2, fte_col] * scale_no).round(2)

        result["adjusted_mr_fte"] = result[fte_col].round(2)
        if fte_col != "adjusted_fte":
            result["adjusted_fte"] = result[fte_col].round(2)
        return result

    # ---- 領域×月ごとに最適配分（per-area mode） ----
    for area, budget in headcount_targets.items():
        cap      = CALLS_PER_DAY.get(area, 2.0) * WORKING_DAYS_PER_MONTH
        area_mask = result["area"] == area

        for month in sorted(result.loc[area_mask, "month"].unique()):
            mask      = area_mask & (result["month"] == month)
            total_req = result.loc[mask, fte_col].sum()

            if total_req <= 0:
                continue

            # ── 合計 < 予算: 比例スケールアップ（全MRを常に配分）──
            # キャップあり = 常にヘッドカウント目標に正規化（上下ともbudget固定）
            # 例: LOE後に需要が380を下回っても380名分のMRは何かに配分される
            # ※ budget + 0.5 の許容幅: round(2)の丸め誤差（最大 ±0.01 × 品目数）が
            #   累積して total_req が budget をわずかに超えた場合に Hill 関数が
            #   誤作動するのを防ぐ。真のオーバー配分（Hill が必要）は必ず 0.5 FTE 以上超過する。
            if total_req <= budget + 0.5:
                scale = budget / total_req
                result.loc[mask, fte_col] = (result.loc[mask, fte_col] * scale).round(2)
                result.loc[mask, "adjusted_mr_fte"] = result.loc[mask, fte_col]
                if fte_col != "adjusted_fte":
                    result.loc[mask, "adjusted_fte"] = result.loc[mask, fte_col]
                continue

            # ── 合計 > 予算: ROI最適配分 ──
            sub = result.loc[mask, ["product_id", fte_col]].set_index("product_id")[fte_col].to_dict()

            # required_fte=0 の品目（post_loe_factor=0.0 でLOE後など）は
            # Hill最適化から除外する。ROI最適化は活動を止めるべき品目を考慮しないため、
            # required_fte=0 の品目を含めると誤ってFTEを割り当ててしまう。
            has_mmm = {pid: mmm_params[pid] for pid in sub
                       if pid in mmm_params and sub[pid] > 0}
            no_mmm  = [pid for pid in sub if pid not in mmm_params]

            req_mmm   = sum(sub[pid] for pid in has_mmm) if has_mmm else 0.0
            req_nommm = sum(sub[pid] for pid in no_mmm)  if no_mmm  else 0.0

            if not has_mmm:
                # 全品目MMM未収録 → 比例スケーリング
                scale = budget / total_req
                result.loc[mask, fte_col] = (result.loc[mask, fte_col] * scale).round(2)
                result.loc[mask, "adjusted_mr_fte"] = result.loc[mask, fte_col]
                if fte_col != "adjusted_fte":
                    result.loc[mask, "adjusted_fte"] = result.loc[mask, fte_col]
                continue

            # グループ間予算: required_fte 比で按分
            bgt_mmm   = float(budget) * (req_mmm   / total_req)
            bgt_nommm = float(budget) * (req_nommm / total_req)

            # ── MMM品目: 等限界収益配分（外側λサーチ）──
            def _total_fte_mmm(lam: float) -> float:
                return sum(_find_activity(b, e, s, lam) / cap for b, e, s in has_mmm.values())

            log_lo, log_hi = np.log(1e-12), np.log(1e8)
            for _ in range(200):
                log_mid = (log_lo + log_hi) / 2.0
                if _total_fte_mmm(float(np.exp(log_mid))) > bgt_mmm:
                    log_lo = log_mid
                else:
                    log_hi = log_mid
            lam_final = float(np.exp((log_lo + log_hi) / 2.0))

            for pid, (beta, ec, slope) in has_mmm.items():
                opt_fte = max(_find_activity(beta, ec, slope, lam_final) / cap, 0.0)
                m2 = mask & (result["product_id"] == pid)
                result.loc[m2, fte_col] = round(opt_fte, 2)

            # ── MMM無品目: 比例スケーリング ──
            if no_mmm and req_nommm > 0:
                scale_no = bgt_nommm / req_nommm
                for pid in no_mmm:
                    m2 = mask & (result["product_id"] == pid)
                    result.loc[m2, fte_col] = (result.loc[m2, fte_col] * scale_no).round(2)

    result["adjusted_mr_fte"] = result[fte_col].round(2)
    if fte_col != "adjusted_fte":
        result["adjusted_fte"] = result[fte_col].round(2)
    return result


def normalize_fte_to_headcount(
    fte_df: pd.DataFrame,
    headcount_targets: Dict[str, int],
    combined_mode: bool = False,
) -> pd.DataFrame:
    """
    各月の領域別合計FTEをヘッドカウント目標に正規化する。

    品目間の相対配分比率を保ちながら、毎月の（領域別または合計）FTEが
    headcount_targets に一致するよう全品目を比例スケーリングする。

    Parameters
    ----------
    fte_df           : run() or run_with_dynamic_new_product() の出力
    headcount_targets: {"CS": 380, "PS遺伝": 25, "PS血液": 40} など
    combined_mode    : True の場合、全領域を1プールとして合計ヘッドカウントに正規化

    Returns
    -------
    adjusted_fte / adjusted_mr_fte を更新したDataFrame
    """
    fte_col = "adjusted_fte" if "adjusted_fte" in fte_df.columns else "required_fte"
    result = fte_df.copy()

    if combined_mode:
        total_target = float(sum(headcount_targets.values()))
        for month in result["month"].unique():
            mask  = result["month"] == month
            total = result.loc[mask, fte_col].sum()
            if total <= 0 or abs(total - total_target) <= 0.5:
                continue
            # 上下ともに比例スケーリング（常に合計 = total_target）
            scale = total_target / total
            result.loc[mask, fte_col] = (result.loc[mask, fte_col] * scale).round(2)
        result["adjusted_mr_fte"] = result[fte_col].round(2)
        if fte_col != "adjusted_fte":
            result["adjusted_fte"] = result[fte_col].round(2)
        return result

    for area, target in headcount_targets.items():
        area_mask = result["area"] == area
        for month in result.loc[area_mask, "month"].unique():
            mask = area_mask & (result["month"] == month)
            total = result.loc[mask, fte_col].sum()
            if total > 0:
                scale = target / total
                result.loc[mask, fte_col] = (result.loc[mask, fte_col] * scale).round(2)

    # FTEはMRのみ: adjusted_mr_fte = adjusted_fte (そのままMR headcount)
    result["adjusted_mr_fte"] = result[fte_col].round(2)
    if fte_col != "adjusted_fte":
        result["adjusted_fte"] = result[fte_col].round(2)

    return result


def discretize_fte_semiannually(
    fte_df: pd.DataFrame,
    headcount_targets: Dict[str, int],
    decay_params_df: Optional["pd.DataFrame"] = None,
    combined_mode: bool = False,
) -> pd.DataFrame:
    """
    月次FTEを半期（6ヶ月）ステップ関数に変換する。

    各品目×半期について、期内の月平均FTEを計算し
    同じ値を期内全月に適用する（離散的な推移）。
    半期の定義: H1=4〜9月, H2=10〜3月（翌年）

    decay_params_df が指定された場合は再正規化に normalize_fte_roi_weighted を使用し、
    指定なしの場合は従来の比例スケーリング（normalize_fte_to_headcount）を使用する。

    Returns
    -------
    半期ステップ化 + 再正規化したDataFrame
    """
    fte_col = "adjusted_fte" if "adjusted_fte" in fte_df.columns else "required_fte"

    def _to_fy_half(ym: str) -> str:
        dt = _parse_ym(ym)
        m  = dt.month
        fy = dt.year if m >= 4 else dt.year - 1
        h  = 1 if 4 <= m <= 9 else 2
        return f"FY{fy}-H{h}"

    result = fte_df.copy()
    result["_half"] = result["month"].apply(_to_fy_half)

    # 品目×半期の平均FTEを各月に適用（ステップ関数化）
    result[fte_col] = (
        result.groupby(["product_id", "_half"])[fte_col]
        .transform("mean")
        .round(2)
    )
    result = result.drop(columns=["_half"])

    # 再正規化（領域×月合計を目標値に戻す）
    if decay_params_df is not None:
        result = normalize_fte_roi_weighted(result, headcount_targets, decay_params_df, combined_mode=combined_mode)
    else:
        result = normalize_fte_to_headcount(result, headcount_targets, combined_mode=combined_mode)
    return result


# 後方互換エイリアス
discretize_fte_quarterly = discretize_fte_semiannually


def calculate_roi_optimal_fte(
    decay_params_df: pd.DataFrame,
    product_configs: "List[ProductConfig]",
    headcount_targets: Dict[str, int],
) -> pd.DataFrame:
    """
    ROI最大化（等限界収益配分）による最適FTE配分を各年度で算出。

    各FYにおいてactiveな品目に対し、Hill関数の等限界収益条件を満たす
    FTE配分を2段階バイナリサーチで求める。

      外側: λ（限界収益閾値）のlog-scaleバイナリサーチ
            → total_fte(λ) = budget となるλを探索
      内側: 品目ごとに dR/da = λ となる活動量のバイナリサーチ

    この配分が「売上最大化を目的とするROI最大FTE」を表す。

    Returns
    -------
    DataFrame: product_id, area, fiscal_year, optimal_fte
    """
    # 各FYの代表月（中間月: 9月 = FY Q2）
    fy_rep_months = {f"FY{y}": f"{y}-09" for y in range(2026, 2036)}

    def _marginal(a: float, beta: float, ec: float, slope: float) -> float:
        """Hill関数の限界収益 dR/da"""
        if a <= 0:
            return float("inf") if slope <= 1 else 0.0
        return beta * slope * (ec ** slope) * (a ** (slope - 1)) / (ec ** slope + a ** slope) ** 2

    def _find_activity(beta: float, ec: float, slope: float, target: float) -> float:
        """dR/da = target を満たす活動量 a を返す（収穫逓減域で探索）"""
        if slope > 1:
            a_inflect = ec * ((slope - 1) / (slope + 1)) ** (1.0 / slope)
            if target > _marginal(a_inflect, beta, ec, slope):
                return 0.0  # このλでは当品目に投資価値なし
            a_lo, a_hi = a_inflect, ec * 500.0
        else:
            a_lo, a_hi = 1e-4, ec * 500.0

        for _ in range(100):
            mid = (a_lo + a_hi) / 2.0
            if _marginal(mid, beta, ec, slope) > target:
                a_lo = mid
            else:
                a_hi = mid
        return (a_lo + a_hi) / 2.0

    rows = []
    for fy, rep_month in fy_rep_months.items():
        for area, budget in headcount_targets.items():
            cap = CALLS_PER_DAY[area] * WORKING_DAYS_PER_MONTH

            # このFY・エリアのactiveな品目とHillパラメータを収集
            active: List[Tuple[str, float, float, float]] = []
            for cfg in product_configs:
                if cfg.area != area:
                    continue
                elapsed  = _months_between(cfg.launch_ym, rep_month)
                loe_rem  = cfg.loe_months - elapsed
                if elapsed < 0 or loe_rem <= 0:
                    continue
                mr_rows = decay_params_df[
                    (decay_params_df["product_id"] == cfg.product_id)
                    & (decay_params_df["channel"].isin(MR_CHANNELS))
                ]
                if mr_rows.empty:
                    continue
                # 面談チャネルを優先、なければMRチャネル群の最初の行
                face_rows = mr_rows[mr_rows["channel"] == "面談"]
                p = face_rows.iloc[0] if not face_rows.empty else mr_rows.iloc[0]
                active.append((cfg.product_id, float(p["beta_m"]), float(p["ec_m"]), float(p["slope_m"])))

            if not active:
                continue

            # total_fte(λ): λに対応する全品目FTE合計（λの単調減少関数）
            def total_fte(lam: float) -> float:
                t = 0.0
                for _, beta, ec, slope in active:
                    t += _find_activity(beta, ec, slope, lam) / cap
                return t

            # log-scaleバイナリサーチでλを探索
            log_lo, log_hi = np.log(1e-10), np.log(1e6)
            for _ in range(200):
                log_mid = (log_lo + log_hi) / 2.0
                lam_mid = float(np.exp(log_mid))
                if total_fte(lam_mid) > budget:
                    log_lo = log_mid  # λ小すぎ → 上げる
                else:
                    log_hi = log_mid  # λ大きすぎ → 下げる
            lam_star = float(np.exp((log_lo + log_hi) / 2.0))

            # 各品目の最適活動量 → FTE変換
            active_pids = {pid for pid, _, _, _ in active}
            total_opt = 0.0
            pid_fte: Dict[str, float] = {}
            for pid, beta, ec, slope in active:
                a_opt = _find_activity(beta, ec, slope, lam_star)
                pid_fte[pid] = a_opt / cap
                total_opt += pid_fte[pid]

            # budgetに合わせて正規化（バイナリサーチの近似誤差を補正）
            scale = budget / total_opt if total_opt > 0 else 1.0
            for pid in pid_fte:
                pid_fte[pid] = pid_fte[pid] * scale

            # max_fte キャップ: 上限超過分を非キャップ品目に比例再配分
            cfg_map = {cfg.product_id: cfg for cfg in product_configs}
            excess = 0.0
            for pid in list(pid_fte):
                cfg = cfg_map.get(pid)
                if cfg and cfg.max_fte is not None and pid_fte[pid] > cfg.max_fte:
                    excess += pid_fte[pid] - cfg.max_fte
                    pid_fte[pid] = cfg.max_fte
            if excess > 0:
                uncapped = [pid for pid in pid_fte
                            if not (cfg_map.get(pid) and cfg_map[pid].max_fte is not None)]
                uncapped_total = sum(pid_fte[p] for p in uncapped)
                if uncapped_total > 0:
                    for pid in uncapped:
                        pid_fte[pid] += excess * pid_fte[pid] / uncapped_total

            for pid, fte_opt in pid_fte.items():
                rows.append({
                    "product_id":  pid,
                    "area":        area,
                    "fiscal_year": fy,
                    "optimal_fte": round(fte_opt, 2),
                })

            # LOE後/未発売品目は optimal_fte = 0
            for cfg in product_configs:
                if cfg.area == area and cfg.product_id not in active_pids:
                    rows.append({
                        "product_id":  cfg.product_id,
                        "area":        area,
                        "fiscal_year": fy,
                        "optimal_fte": 0.0,
                    })

    return pd.DataFrame(rows)


# ============================================================
# データクラス
# ============================================================

@dataclass
class ProductConfig:
    """品目設定"""
    product_id: str
    area: str                   # "CS" | "PS"
    is_new: bool                # FY2029新発売か
    launch_ym: str              # 発売年月 "YYYY-MM"
    loe_months: float           # 発売からLOEまでの月数
    estimated_patients: int     # 推定患者数
    num_indications: int        # 効能数
    # 効能追加パラメータ（省略可）
    indication_add_ym: Optional[str] = None   # 効能追加年月（第1効能） "YYYY-MM"
    indication_fte_boost: float = 1.0         # 効能追加時のFTEブースト倍率（第1効能, 例: 3.0）
    indication_boost_months: int = 0          # ブースト持続月数（第1効能）
    # 第2効能追加パラメータ（省略可）
    indication_add_ym_2: Optional[str] = None   # 効能追加年月（第2効能） "YYYY-MM"
    indication_fte_boost_2: float = 1.0         # 効能追加時のFTEブースト倍率（第2効能, 例: 1.667）
    indication_boost_months_2: int = 0          # ブースト持続月数（第2効能）
    # バイオシミラー浸食耐性パラメータ（省略可）
    post_loe_factor: float = 0.0  # LOE後に維持するライフサイクル係数（0.0=小分子完全停止, 0.55=バイオ品ENT等）
    # Doctor Mindscapeターゲットパーセンタイル（新製品のみ。0=未設定 → 参照品推定）
    mindscape_target_pct: int = 0  # 例: 40 → 処方量上位40%の医師を対象
    # FTE上限（省略可）: 設定時はこの値でキャップし余剰を他品目に再配分
    max_fte: Optional[float] = None


@dataclass
class MMMDecayParams:
    """Meridian出力: 品目×チャネル別減衰曲線パラメータ"""
    product_id: str
    channel: str         # "面談" | "面談_アポ" | "説明会" | "Web講演会" | "Webinar" | "e-contents" | "メール"
    alpha: float         # Adstock減衰率
    beta_m: float        # 応答曲線スケール係数（売上換算天井）
    ec_m: float          # EC50（半最大効果到達活動量）
    eta_m: float         # チャネル効率パラメータ
    mr_time_weight: float  # MR時間換算係数（面談=1.0, e-contents=0.0等）
    slope_m: float = 1.0   # Hill係数（固定=1.0: ミカエリス-メンテン型）


# ============================================================
# MMMパラメータ ライフサイクル調整
# ============================================================

# チャネルグループ定義
MR_CHANNELS = {"面談", "面談_アポ", "説明会"}
DIGITAL_CHANNELS = {"Web講演会", "Webinar", "e-contents", "メール"}

# ライフサイクルイベント別 調整乗数定義
# beta_m: 応答天井への乗数
# ec_m:   EC50への乗数（<1で飽和が早まる = 競合で市場が難しくなる）
# eta_m:  チャネル効率への乗数

LOE_DECAY_RULES = {
    # years_since_loe -> (beta_m_mult, ec_m_mult)
    0: (0.80, 0.90),
    1: (0.55, 0.75),
    2: (0.35, 0.60),
    3: (0.20, 0.50),
}

LAUNCH_RAMP_RULES = {
    # years_since_launch -> (beta_m_mult, ec_m_mult)
    0: (0.40, 1.30),
    1: (0.60, 1.15),
    2: (0.80, 1.05),
    3: (1.00, 1.00),
}

COMPETITOR_ENTRY_EC_MULT = 0.85  # 競合参入でec_mが15%低下

DIGITAL_ETA_TREND_PER_YEAR = 0.02  # デジタルチャネル eta_m が年2%向上


class MMMParameterAdjuster:
    """
    MMMの事後分布パラメータをライフサイクルイベントに基づいて将来調整する。

    調整対象:
      - beta_m: LOE後は低下、新製品ランプアップで上昇
      - ec_m:   競合参入で低下（飽和が早まる）
      - eta_m:  デジタルチャネルは年率2%向上トレンド

    変更しないもの:
      - alpha:   チャネル固有の効果持続性（製品ライフサイクルと独立）
      - gamma_c: 施設固有特性（安定）
      - sigma:   観測ノイズ
      - slope_m: 1.0固定
    """

    def __init__(
        self,
        decay_params_df: pd.DataFrame,
        loe_schedule: Optional[Dict[str, str]] = None,
        launch_schedule: Optional[Dict[str, str]] = None,
        competitor_entry: Optional[Dict[str, str]] = None,
        base_year: int = 2026,
    ) -> None:
        """
        Args:
            decay_params_df: mmm_decay_params.csv の DataFrame
            loe_schedule:    {product_id: "YYYY-MM"} LOE発生年月
            launch_schedule: {product_id: "YYYY-MM"} 発売年月
            competitor_entry:{product_id: "YYYY-MM"} 主要競合参入年月
            base_year:       パラメータ推定の基準年度
        """
        self.base_df = decay_params_df.copy()
        self.loe_schedule = loe_schedule or {}
        self.launch_schedule = launch_schedule or {}
        self.competitor_entry = competitor_entry or {}
        self.base_year = base_year

    def _years_since(self, event_ym: str, target_fy: int) -> Optional[float]:
        """target_FY開始時点でのイベント経過年数（負なら未発生）"""
        try:
            ey, em = int(event_ym[:4]), int(event_ym[5:7])
            # FY開始は4月
            fy_start_months = target_fy * 12 + 4
            event_months = ey * 12 + em
            return (fy_start_months - event_months) / 12.0
        except Exception:
            return None

    def get_adjusted_params(self, product_id: str, fiscal_year: int) -> pd.DataFrame:
        """
        指定品目・年度の調整済みMMMパラメータを返す。

        Returns:
            decay_params_dfと同構造のDataFrame（当該品目の行のみ）
        """
        rows = self.base_df[self.base_df["product_id"] == product_id].copy()
        if rows.empty:
            return rows

        beta_mult = 1.0
        ec_mult   = 1.0

        # --- LOE調整 ---
        if product_id in self.loe_schedule:
            ysl = self._years_since(self.loe_schedule[product_id], fiscal_year)
            if ysl is not None and ysl >= 0:
                key = min(int(ysl), max(LOE_DECAY_RULES.keys()))
                b_m, e_m = LOE_DECAY_RULES[key]
                beta_mult *= b_m
                ec_mult   *= e_m

        # --- 新製品ランプアップ調整 ---
        if product_id in self.launch_schedule:
            ysl = self._years_since(self.launch_schedule[product_id], fiscal_year)
            if ysl is not None and 0 <= ysl < 4:
                key = min(int(ysl), max(LAUNCH_RAMP_RULES.keys()))
                b_m, e_m = LAUNCH_RAMP_RULES[key]
                beta_mult *= b_m
                ec_mult   *= e_m

        # --- 競合参入調整 ---
        if product_id in self.competitor_entry:
            ysl = self._years_since(self.competitor_entry[product_id], fiscal_year)
            if ysl is not None and ysl >= 0:
                ec_mult *= COMPETITOR_ENTRY_EC_MULT

        # beta_m / ec_m 適用
        rows["beta_m"] = rows["beta_m"] * beta_mult
        rows["ec_m"]   = rows["ec_m"]   * ec_mult

        # --- デジタルチャネル eta_m トレンド ---
        years_from_base = fiscal_year - self.base_year
        digital_mask = rows["channel"].isin(DIGITAL_CHANNELS)
        rows.loc[digital_mask, "eta_m"] = (
            rows.loc[digital_mask, "eta_m"] * (1 + DIGITAL_ETA_TREND_PER_YEAR) ** years_from_base
        )

        return rows

    def response_curve_points(
        self, product_id: str, fiscal_year: int, channel: str,
        n_points: int = 50, x_max: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """
        指定チャネルの調整済みレスポンスカーブ点列を返す（x=活動量, y=売上効果）。
        slope_m=1固定のミカエリス-メンテン型。
        x_max を指定すると横軸範囲を固定できる（複数年度を同一軸で比較する場合）。
        """
        params = self.get_adjusted_params(product_id, fiscal_year)
        row = params[params["channel"] == channel]
        if row.empty:
            return []
        beta = float(row["beta_m"].iloc[0])
        ec   = float(row["ec_m"].iloc[0])
        if x_max is None:
            x_max = ec * 5
        points = []
        for i in range(n_points + 1):
            x = x_max * i / n_points
            y = beta * x / (ec + x)  # slope=1固定
            points.append((round(x, 2), round(y, 2)))
        return points


def _months_between(ym_start: str, ym_end: str) -> int:
    """2つの年月文字列間の月数差（ym_end - ym_start）を返す"""
    d1 = _parse_ym(ym_start)
    d2 = _parse_ym(ym_end)
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


# ============================================================
# Module 1: ターゲット医師数算出
# ============================================================

class TargetDoctorCalculator:
    """
    品目×月ごとのターゲット医師数を算出。

    データソース優先順位:
    1. doctor_tiers（target_doctors.csv の R/W ティアデータ）← R+W の合計を使用
    2. base_target_doctors（target_doctors.csv 旧形式）← 後方互換
    3. mindscape_segments_df（Doctor Mindscape セグメントデータ） ← 新製品で使用
    4. activity_data の直近12ヶ月集計（フォールバック）

    引数:
    - base_target_doctors   : {product_id: 医師数}  target_doctors.csv から読込
    - mindscape_segments_df : product_id, pct_from, pct_to, doctor_count
    - activity_data         : フォールバック用（既存品目でbase_target_doctorsにない場合）
    - product_info          : product_id, launch_ym, loe_months, estimated_patients, num_indications
    - doctor_tiers          : {product_id: {r_doctors, w_doctors, r_visit_freq, w_visit_freq}}
    """

    def __init__(
        self,
        base_target_doctors: Dict[str, int],
        product_info: pd.DataFrame,
        mindscape_segments_df: Optional[pd.DataFrame] = None,
        activity_data: Optional[pd.DataFrame] = None,
        doctor_tiers: Optional[Dict[str, dict]] = None,
        yearly_doctor_counts: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None,
        visit_freq_data: Optional[Dict[Tuple[str, str, str], dict]] = None,
        # {(product_id, "FY2026", "R"/"W"): {"target_freq": float, "achievement_rate": float}}
    ) -> None:
        self.base_target_doctors = base_target_doctors
        self.doctor_tiers = doctor_tiers or {}
        self.yearly_doctor_counts = yearly_doctor_counts or {}
        self.product_info = product_info
        self.mindscape_segments_df = mindscape_segments_df if mindscape_segments_df is not None else pd.DataFrame()
        self.activity_data = activity_data if activity_data is not None else pd.DataFrame()
        self.visit_freq_data = visit_freq_data or {}

    # ----------------------------------------------------------
    # 既存品目用
    # ----------------------------------------------------------

    def get_doctor_tier(self, product_id: str, month: Optional[str] = None) -> Optional[dict]:
        """
        R/W医師ティアデータを返す。なければ None。

        month を指定すると target_doctor_yearly.csv の年度別データを優先参照。
        返り値: {"r_doctors": int, "w_doctors": int,
                 "r_visit_freq": float, "w_visit_freq": float}
        """
        base = self.doctor_tiers.get(product_id)
        if base is None:
            return None

        if month and self.yearly_doctor_counts:
            fy = month_to_fy(month)
            yearly = self.yearly_doctor_counts.get((product_id, fy))
            if yearly is not None:
                return {
                    "r_doctors":    yearly[0],
                    "w_doctors":    yearly[1],
                    "r_visit_freq": base["r_visit_freq"],
                    "w_visit_freq": base["w_visit_freq"],
                }

        return base

    def get_visit_freq(self, product_id: str, month: str, tier: str) -> Tuple[float, float, float]:
        """
        品目×ティアの実効訪問頻度を返す。

        visit_freq.csv に年度別データがある場合はそれを使用。
        ない場合は target_doctors.csv の base tier freq を返す。

        Returns: (effective_freq, target_freq, achievement_rate)
        """
        fy = month_to_fy(month)
        key = (product_id, fy, tier)
        if key in self.visit_freq_data:
            row = self.visit_freq_data[key]
            tgt = float(row["target_freq"])
            ach = float(row["achievement_rate"])
            return tgt * ach, tgt, ach

        # フォールバック: target_doctors.csv の base freq (lc_adj は呼び元で適用)
        base = self.doctor_tiers.get(product_id)
        if base:
            freq = float(base["r_visit_freq"] if tier == "R" else base["w_visit_freq"])
            return freq, freq, 1.0
        return 1.0, 1.0, 1.0

    def get_base_target_doctors(self, product_id: str, month: Optional[str] = None) -> int:
        """
        品目のベースターゲット医師数（R+W合計）を返す。

        優先順位:
        1. yearly_doctor_counts（年度別指定があれば優先）
        2. doctor_tiers の R+W 合計（target_doctors.csv 新形式）
        3. base_target_doctors（旧形式フォールバック）
        4. activity_data の直近12ヶ月ユニーク医師数
        """
        if month and self.yearly_doctor_counts:
            fy = month_to_fy(month)
            yearly = self.yearly_doctor_counts.get((product_id, fy))
            if yearly is not None:
                return yearly[0] + yearly[1]

        tier = self.doctor_tiers.get(product_id)
        if tier:
            return tier["r_doctors"] + tier["w_doctors"]

        if product_id in self.base_target_doctors:
            return self.base_target_doctors[product_id]

        # フォールバック: activity_data から集計
        if self.activity_data.empty:
            warnings.warn(f"{product_id}: target_doctors.csv にも活動データにもなし → 0")
            return 0

        prod_acts = self.activity_data[self.activity_data["product_id"] == product_id]
        if prod_acts.empty:
            warnings.warn(f"{product_id}: 活動データなし、ターゲット医師数=0")
            return 0

        monthly_unique = (
            prod_acts.groupby("activity_ym")["doctor_id"]
            .nunique()
            .tail(12)
            .mean()
        )
        return int(round(monthly_unique))

    def calculate(
        self,
        product_id: str,
        months: List[str],
        config: Optional[ProductConfig] = None,
        ramp_up_curve: Optional[List[float]] = None,
        reference_product_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        品目×月リストのターゲット医師数を返す。

        新発売品（config.is_new == True）の場合:
          - reference_product_id（類似品）のベース数を参照
          - 患者数・効能数で補正
          - ramp_up_curve で発売後の浸透率を適用
        既存品:
          - 過去実績のベース数を使用（月による変化なし）

        Returns
        -------
        DataFrame: product_id, month, target_doctors
        """
        if config is not None and config.is_new:
            return self._calculate_new_product(
                config, months, ramp_up_curve or [], reference_product_id
            )

        # Build per-month target doctors using yearly override when available
        records = []
        for m in months:
            base = self.get_base_target_doctors(product_id, month=m)
            records.append({"product_id": product_id, "month": m, "target_doctors": base})
        return pd.DataFrame(records)

    # ----------------------------------------------------------
    # 新発売品用
    # ----------------------------------------------------------

    def _calculate_new_product(
        self,
        config: ProductConfig,
        months: List[str],
        ramp_up_curve: List[float],
        reference_product_id: Optional[str],
    ) -> pd.DataFrame:
        """
        新発売品ターゲット医師数の算出。

        優先順位（baseの決定）:
        0. target_doctor_yearly.csv の年度別直接指定（最優先）
           その年度のエントリがあれば R+W を医師数として直接使用（ランプ不要）
        1. Doctor Mindscape セグメントデータ（config.mindscape_target_pct > 0 かつデータあり）
           base = mindscape_segments の pct_to <= target_pct の doctor_count 合計
        2. 参照品の医師数 × 患者数比 × 効能数補正（フォールバック）
        """
        pid = config.product_id
        base = 0

        # ---- 1. Doctor Mindscape ----
        if config.mindscape_target_pct > 0 and not self.mindscape_segments_df.empty:
            seg = self.mindscape_segments_df[
                self.mindscape_segments_df["product_id"] == pid
            ]
            if not seg.empty:
                base = int(
                    seg[seg["pct_to"] <= config.mindscape_target_pct]["doctor_count"].sum()
                )
                if base > 0:
                    pass  # Mindscapeデータ使用
                else:
                    warnings.warn(f"{pid}: Mindscape target_pct={config.mindscape_target_pct} でセグメントが空 → 参照品推定に切替")

        # ---- 2. 参照品推定（フォールバック）----
        if base == 0:
            if reference_product_id is None:
                base = int(config.estimated_patients * 0.10)
            else:
                ref_info = self.product_info[self.product_info["product_id"] == reference_product_id]
                new_info = self.product_info[self.product_info["product_id"] == pid]
                if ref_info.empty or new_info.empty:
                    base = int(config.estimated_patients * 0.10)
                else:
                    ref = ref_info.iloc[0]
                    nw  = new_info.iloc[0]
                    ref_base = self.get_base_target_doctors(reference_product_id)
                    patient_ratio = nw["estimated_patients"] / max(ref["estimated_patients"], 1)
                    indication_adj = np.sqrt(nw["num_indications"] / max(ref["num_indications"], 1))
                    base = int(ref_base * patient_ratio * indication_adj)

        records = []
        for month in months:
            elapsed = _months_between(config.launch_ym, month)
            if elapsed < 0:
                doctors = 0
            else:
                # ---- 0. target_doctor_yearly.csv による年度別直接指定（最優先）----
                # patient_ratio 等のフォーミュラを使わずに医師数を直接管理したい場合に使用。
                # エントリがある年度はその R+W を医師数として採用（ランプ曲線は適用しない）。
                fy = month_to_fy(month)
                yearly = self.yearly_doctor_counts.get((pid, fy))
                if yearly is not None:
                    doctors = yearly[0] + yearly[1]
                elif elapsed < len(ramp_up_curve):
                    doctors = int(base * ramp_up_curve[elapsed])
                else:
                    doctors = int(base * (ramp_up_curve[-1] if ramp_up_curve else 1.0))
            records.append({
                "product_id": config.product_id,
                "month": month,
                "target_doctors": doctors,
            })
        return pd.DataFrame(records)


# ============================================================
# Module 2: FC/SC 訪問コスト重み算出
# ============================================================

class FCScAllocator:
    """
    品目の FC/SC 訪問コスト重みを管理する。

    FC（First Call）= その品目が訪問の主目的
    SC（Second Call）= 他品目の訪問に内包される（ついで話し）

    医師をFC/SCに分類するのではなく、品目の訪問コストに重みをかける考え方。
    ある医師AにTRIはFC・INTはSCとして訪問するとき、TRIの訪問コストはフル、
    INTはその訪問に内包されるためほぼ追加コストなし。

    fc_weight = fc_ratio + (1 - fc_ratio) × SC_COEFFICIENT
    required_calls_effective = raw_required_calls × fc_weight

    fc_sc_ratio.csv の列:
      product_id, fiscal_year, fc_ratio (0.0〜1.0), note
      fiscal_year = "default" は全年度のデフォルト値
      fiscal_year = "FY2027" のように年度指定すると、その年度のみ上書き
      fc_ratio=1.0 → 全訪問がFC（主訪問）
      fc_ratio=0.0 → 全訪問がSC（他品目のついで）
    """

    def __init__(
        self,
        fc_ratio_df: Optional["pd.DataFrame"] = None,
        fc_ratios: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Args:
            fc_ratio_df: fc_sc_ratio.csv の DataFrame（product_id, fiscal_year, fc_ratio）
            fc_ratios:   旧形式の {product_id: fc_ratio} dict（後方互換用）
        """
        import pandas as _pd
        self._by_fy: Dict[Tuple[str, str], float] = {}  # (product_id, "FY2027") -> ratio
        self._default: Dict[str, float] = {}            # product_id -> default ratio

        if fc_ratio_df is not None and not fc_ratio_df.empty:
            for _, row in fc_ratio_df.iterrows():
                pid = str(row["product_id"]).strip()
                fy  = str(row.get("fiscal_year", "default")).strip()
                ratio = float(row["fc_ratio"])
                if fy == "default":
                    self._default[pid] = ratio
                else:
                    self._by_fy[(pid, fy)] = ratio
        elif fc_ratios:
            # 後方互換：旧形式のdictをdefaultとして登録
            self._default = {k: float(v) for k, v in fc_ratios.items()}

    def get_fc_ratio(self, product_id: str, month: Optional[str] = None) -> float:
        """
        品目・月のFC比率を返す。

        Args:
            product_id: 品目ID
            month:      "YYYY-MM" 形式（例: "2027-06"）。Noneならデフォルト値を返す

        Returns:
            fc_ratio (0.0〜1.0)。未指定なら 1.0（全訪問FC）
        """
        fy = None
        if month:
            try:
                y, m = int(month[:4]), int(month[5:7])
                # 会計年度: 4月始まり（2027-04〜2028-03 → FY2027）
                fy = f"FY{y}" if m >= 4 else f"FY{y - 1}"
            except (ValueError, IndexError):
                pass

        # 年度別指定 → デフォルト の優先順位
        if fy and (product_id, fy) in self._by_fy:
            return self._by_fy[(product_id, fy)]
        return float(self._default.get(product_id, 1.0))

    def get_fc_weight(self, product_id: str, month: Optional[str] = None) -> float:
        """
        訪問コスト重み = fc_ratio + (1 - fc_ratio) × SC_COEFFICIENT

        ただし fc_ratio=0.0（100%SC）の場合は weight=0.0。
        SC100%品目は FC品目の訪問に完全内包されており、
        独自のMR稼働コストは発生しない（FTE=0）。

        例:
          GLI fc_ratio=1.00 → weight=1.00（フルコスト）
          INT fc_ratio=0.00 → weight=0.00（FC訪問に完全内包、独自FTEなし）
          CUV fc_ratio=0.64 → weight=0.676
          ADV fc_ratio=0.05 → weight=0.145（一部FC）
        """
        fc_ratio = self.get_fc_ratio(product_id, month=month)
        if fc_ratio == 0.0:
            return 0.0  # 100%SC: FC訪問に完全内包、独自FTEコスト不要
        return fc_ratio + (1.0 - fc_ratio) * SC_COEFFICIENT


# ============================================================
# Module 3: 活動頻度推定
# ============================================================

class ActivityFrequencyEstimator:
    """
    品目×月ごとの必要活動頻度（回/ターゲット医師/月）を推定。

    3モード:
      "actual"            : 過去活動実績の平均
      "mmm_optimal"       : Meridianの施設別最適実施数テーブルから
      "lifecycle_adjusted": 実績ベース × ライフサイクル補正係数

    ライフサイクル補正係数:
      発売後 ≤12ヶ月  : ×1.3（ローンチ高頻度）
      LOEまで ≤12ヶ月 : ×0.7（縮小フェーズ）
      LOE後           : ×0.3
      それ以外         : ×1.0
    """

    LIFECYCLE_ADJ = {
        "launch":  1.3,
        "normal":  1.0,
        "pre_loe": 0.7,
        "post_loe": 0.3,
    }

    def __init__(
        self,
        activity_data: pd.DataFrame,
        product_info: pd.DataFrame,
        mmm_optimal_activities: Optional[pd.DataFrame] = None,
    ) -> None:
        self.activity_data = activity_data
        self.product_info = product_info
        self.mmm_optimal = mmm_optimal_activities
        self._actual_freq_cache: Dict[str, float] = {}

    def get(
        self,
        product_id: str,
        month: str,
        mode: str = "lifecycle_adjusted",
    ) -> float:
        """
        活動頻度（回/医師/月）を返す。
        mode: "actual" | "mmm_optimal" | "lifecycle_adjusted"
        """
        if mode == "actual":
            return self._actual_freq(product_id)
        if mode == "mmm_optimal":
            return self._mmm_freq(product_id, month)
        if mode == "lifecycle_adjusted":
            base = self._actual_freq(product_id)
            return base * self._lifecycle_adj(product_id, month)
        raise ValueError(f"mode={mode!r} は不正です")

    def _actual_freq(self, product_id: str) -> float:
        if product_id in self._actual_freq_cache:
            return self._actual_freq_cache[product_id]

        prod_acts = self.activity_data[
            self.activity_data["product_id"] == product_id
        ]
        if prod_acts.empty:
            return 1.0  # デフォルト: 月1回

        freq = (
            prod_acts.groupby(["doctor_id", "activity_ym"])["activity_count"]
            .sum()
            .mean()
        )
        result = float(freq) if not np.isnan(freq) else 1.0
        self._actual_freq_cache[product_id] = result
        return result

    def _mmm_freq(self, product_id: str, month: str) -> float:
        if self.mmm_optimal is None:
            warnings.warn(f"{product_id}: MMM最適データなし → 実績値を使用")
            return self._actual_freq(product_id)

        subset = self.mmm_optimal[
            (self.mmm_optimal["product_id"] == product_id)
            & (self.mmm_optimal["month"] == month)
        ]
        if subset.empty:
            return self._actual_freq(product_id)
        return float(subset["optimal_calls_per_doctor"].iloc[0])

    def _lifecycle_adj(self, product_id: str, month: str) -> float:
        """
        連続的なライフサイクル調整係数。品目の薬剤クラスに応じて LOE 周辺の
        カーブ形状が変わる（post_loe_factor で統一的にパラメータ化）。

        フェーズ別の係数推移:
          フェーズ1 - ローンチ期 (0〜18ヶ月):
            0.80 → 1.05（全品目共通）
          フェーズ2 - 成長期 (18〜36ヶ月):
            1.05 → 1.00（全品目共通）
          フェーズ3 - 成熟〜低下期 (36ヶ月〜LOE36ヶ月前):
            1.00 → phase3_floor  where phase3_floor = max(0.80, post_loe_factor)
              小分子:     1.00 → 0.80（緩やかに低下）
              バイオ/ENT: 1.00 → 0.80（同左）
              血漿分画:   1.00 → 0.90（ほぼ横ばい）
          フェーズ4 - LOE直前期 (LOE36ヶ月前〜LOE):
            phase3_floor → loe_floor  where loe_floor = max(0.35, post_loe_factor)
              小分子:     0.80 → 0.35（急速低下）
              バイオ/ENT: 0.80 → 0.55（緩慢に低下）
              血漿分画:   0.90 → 0.90（ほぼフラット、浸食なし）
          フェーズ5 - LOE後:
            post_loe_factor を維持
              小分子:     0.0（MR活動終了）
              バイオ/ENT: 0.55（バイオシミラー参入後も一定の活動を維持）
              血漿分画:   0.90（特許切れても市場浸食がほぼないため活動継続）
        """
        info = self.product_info[
            self.product_info["product_id"] == product_id
        ]
        if info.empty:
            return 1.0
        row = info.iloc[0]
        elapsed = _months_between(row["launch_ym"], month)
        loe_months = float(row["loe_months"])
        loe_remaining = loe_months - elapsed

        # post_loe_factor: LOE後に維持するライフサイクル係数
        #   小分子=0.0 / バイオ=0.55 / 血漿分画=0.90
        post_loe = float(row.get("post_loe_factor", 0.0)) if "post_loe_factor" in row.index else 0.0

        # フェーズ5 - LOE後
        if loe_remaining <= 0:
            return post_loe

        # フェーズ3〜4 の境界点を品目クラスに応じて設定
        loe_floor    = max(0.35, post_loe)   # フェーズ4の着地点（LOE直前の最低値）
        phase3_floor = max(0.80, loe_floor)  # フェーズ3の着地点 = フェーズ4の開始点

        # フェーズ4: LOE直前36ヶ月 → phase3_floor → loe_floor へ変化
        if loe_remaining <= 36:
            t = loe_remaining / 36.0
            return loe_floor + (phase3_floor - loe_floor) * t

        # フェーズ1: ローンチ後18ヶ月以内 → 0.80 から 1.05 へ線形上昇
        if elapsed <= 18:
            t = elapsed / 18.0
            return 0.80 + 0.25 * t

        # フェーズ2: 18〜36ヶ月 → 1.05 から 1.00 へ緩やかに降下
        if elapsed <= 36:
            t = (elapsed - 18) / 18.0
            return 1.05 - 0.05 * t

        # フェーズ3: 成熟期 (36ヶ月〜LOE36ヶ月前) → 1.00 から phase3_floor へ線形低下
        mature_start = 36.0
        mature_end   = max(loe_months - 36.0, mature_start + 1.0)
        if elapsed < mature_end:
            t = (elapsed - mature_start) / (mature_end - mature_start)
            return 1.00 - (1.00 - phase3_floor) * t
        return phase3_floor


# ============================================================
# Module 4: デジタル有効性スコア算出
# ============================================================

class DigitalEffectivenessScorer:
    """
    品目ごとのデジタルチャネル有効性スコア（0～1）を算出する。

    FTE算出とは独立した分析モジュール。MR FTE は全活動から直接計算し、
    本スコアはチャネル戦略上の示唆（m3等デジタル活用余地）を提供する。

    スコア構成（合計 1.0）:
      1. MMM デジタル応答比 （W=0.50）
         Hill関数の総効果量: dig_val / (mr_val + dig_val)
         活動量 x における応答量なので活動数補正済み。

      2. SOC デジタル感受性 （W=0.50）
         digital_soc_rate（soc_params.csv）を直接スコアとして使用。
         医師が1回の視聴でどれだけ想起するかの確率（0～1）。

      ライフサイクル補正（加算値）:
         LOE後          : -0.25 （投資価値なし）
         LOE <1年       : -0.20 （回収困難）
         LOE 1～3年     : -0.10 （段階縮小）
         発売 <1年      : -0.05 （MR啓発期、デジタル補完役）
         発売 1～3年    : +0.08 （成長期、デジタル補完効果大）
         成熟期（3年～）: +0.05 （安定期、デジタルで効率維持）

    最終スコア: clip(base + lc_adj, 0.0, 1.0)
    レベル判定: 0.55以上=高, 0.35以上=中, 未満=低

    decay_params_df : product_id, channel, alpha, beta, ec, slope
    soc_activity    : {product_id: {"mr": float, "digital": float}}  月次平均活動数
    soc_rates       : {product_id: {"mr": float, "digital": float}}  想起率（0～1）
    """

    W_MMM = 0.50
    W_SOC = 0.50

    DEFAULT_DIGITAL_FRACTION = 0.35   # MMMデータなし時のデジタル応答比デフォルト
    DEFAULT_DIGITAL_SOC_RATE = 0.25   # SOCデータなし時のデジタル想起率デフォルト

    def __init__(
        self,
        decay_params_df: pd.DataFrame,
        soc_activity: Optional[Dict[str, Dict[str, float]]] = None,
        soc_rates: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.params = decay_params_df
        self.soc_activity = soc_activity or {}
        self.soc_rates = soc_rates or {}

    # ----------------------------------------------------------
    # Hill関数ユーティリティ（NewProductFTEAllocator でも使用）
    # ----------------------------------------------------------

    @staticmethod
    def _hill_value(x: float, beta: float, ec: float, slope: float) -> float:
        """Hill関数の総効果量: beta × x^slope / (ec^slope + x^slope)"""
        if x <= 0:
            return 0.0
        return beta * (x ** slope) / (ec ** slope + x ** slope)

    @staticmethod
    def _hill_marginal(x: float, ec: float, slope: float) -> float:
        """Hill関数の微分（1回あたりの限界応答）。ROI計算に使用。"""
        a = max(x, 1e-6)
        num = slope * (ec ** slope) * (a ** (slope - 1))
        den = (ec ** slope + a ** slope) ** 2
        return num / den if den > 0 else 0.0

    # ----------------------------------------------------------
    # ライフサイクル補正（デジタル有効性視点）
    # ----------------------------------------------------------

    @staticmethod
    def _digital_lifecycle_adj(years_since_launch: float, years_to_loe: float) -> float:
        """LOE残年数・発売年数に基づくデジタル有効性スコアの加算調整値。"""
        if years_to_loe <= 0:
            return -0.25   # LOE後: デジタル投資価値なし
        elif years_to_loe < 1:
            return -0.20   # LOE直前: 投資回収困難
        elif years_to_loe < 3:
            return -0.10   # LOE直前3年: 段階縮小
        elif years_since_launch < 1:
            return -0.05   # 発売直後: MR啓発期、デジタル補完役
        elif years_since_launch < 3:
            return +0.08   # 成長期: デジタル補完効果大
        else:
            return +0.05   # 成熟期: デジタルで効率維持

    # ----------------------------------------------------------
    # スコア算出
    # ----------------------------------------------------------

    def score(
        self,
        product_id: str,
        months_since_launch: int = 0,
        loe_months_remaining: float = 999,
    ) -> Dict[str, object]:
        """
        品目のデジタル有効性スコアを算出する。

        Returns
        -------
        {
          "score": float (0～1),
          "level": str ("高"/"中"/"低"),
          "mmm_digital_fraction": float,
          "digital_soc_rate": float,
          "lifecycle_adj": float,
        }
        """
        years_since_launch = months_since_launch / 12.0
        years_to_loe       = loe_months_remaining / 12.0

        soc = self.soc_activity.get(product_id, {})
        mr_act  = float(soc.get("mr",      0.0))
        dig_act = float(soc.get("digital", 0.0))

        rates = self.soc_rates.get(product_id, {})
        dig_soc_rate = float(rates.get("digital", self.DEFAULT_DIGITAL_SOC_RATE))

        # ---- 1. MMM: デジタルの応答比率 ----
        _mr_rows  = self.params[
            (self.params["product_id"] == product_id) & (self.params["channel"].isin(MR_CHANNELS))
        ]
        _dig_rows = self.params[
            (self.params["product_id"] == product_id) & (self.params["channel"].isin(DIGITAL_CHANNELS))
        ]
        # 面談チャネルを優先
        _face = _mr_rows[_mr_rows["channel"] == "面談"]
        mr_row  = _face if not _face.empty else _mr_rows
        dig_row = _dig_rows
        if not mr_row.empty and not dig_row.empty:
            mr  = mr_row.iloc[0]
            dig = dig_row.iloc[0]
            mr_val  = self._hill_value(mr_act,  float(mr["beta_m"]),  float(mr["ec_m"]),  float(mr["slope_m"]))
            dig_val = self._hill_value(dig_act, float(dig["beta_m"]), float(dig["ec_m"]), float(dig["slope_m"]))
            total_mmm = mr_val + dig_val
            mmm_dig_frac = (dig_val / total_mmm) if total_mmm > 0 else self.DEFAULT_DIGITAL_FRACTION
        else:
            mmm_dig_frac = self.DEFAULT_DIGITAL_FRACTION

        # ---- 2. SOC: デジタル感受性（想起率を直接使用）----
        # digital_soc_rate は活動1件あたりの想起確率（0～1）= デジタル感受性指標

        # ---- 3. ライフサイクル補正 ----
        lc_adj = self._digital_lifecycle_adj(years_since_launch, years_to_loe)

        # ---- スコア合成 ----
        base  = self.W_MMM * mmm_dig_frac + self.W_SOC * dig_soc_rate
        final = float(np.clip(base + lc_adj, 0.0, 1.0))
        level = "高" if final >= 0.55 else ("中" if final >= 0.35 else "低")

        return {
            "score":                round(final, 3),
            "level":                level,
            "mmm_digital_fraction": round(mmm_dig_frac, 3),
            "digital_soc_rate":     round(dig_soc_rate, 3),
            "lifecycle_adj":        round(lc_adj, 3),
        }


# ============================================================
# Module 5: 新品目へのFTE配分（ドナー品目決定）
# ============================================================

class NewProductFTEAllocator:
    """
    新品目（OVE）に必要なFTEをどの既存品目から削るかを定量的に決定。

    原則:
      MMMの限界ROI（追加1コールあたりの売上増）が低い品目ほど
      活動削減による売上損失が小さい → 優先的にFTEを削る。

    配分方式:
      削減量 ∝ (1/限界ROI) × 余裕FTE（現在FTE - 最低保証FTE）

    decay_params_df  : product_id, channel, beta, ec, slope
    current_fte      : {product_id: 現在のFTE数}
    min_fte_ratio    : 各品目の最低保証FTE比率（デフォルト50%）
    """

    def __init__(
        self,
        decay_params_df: pd.DataFrame,
        current_fte: Dict[str, float],
        current_mr_activity: Dict[str, float],  # 品目 → 月次MRコール数
        min_fte_ratio: float = 0.5,
    ) -> None:
        self.params = decay_params_df
        self.current_fte = current_fte
        self.current_mr_activity = current_mr_activity
        self.min_fte_ratio = min_fte_ratio

    def marginal_roi(self, product_id: str) -> float:
        """品目の現在活動量における限界ROI"""
        _mr_rows = self.params[
            (self.params["product_id"] == product_id)
            & (self.params["channel"].isin(MR_CHANNELS))
        ]
        _face = _mr_rows[_mr_rows["channel"] == "面談"]
        mr_row = _face if not _face.empty else _mr_rows
        if mr_row.empty:
            return 0.0
        p = mr_row.iloc[0]
        x = self.current_mr_activity.get(product_id, 0.0)
        if x <= 0:
            return float("inf")
        return DigitalEffectivenessScorer._hill_marginal(x, p["ec_m"], p["slope_m"]) * p["beta_m"]

    def allocate(
        self,
        new_product_fte: float,
        donor_products: List[str],
        min_fte_override: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        新品目に必要なFTEを既存品目から按分して削減。

        Parameters
        ----------
        new_product_fte    : 新品目に必要なFTE合計
        donor_products     : ドナー候補品目リスト
        min_fte_override   : 品目別FTE下限の上書き {product_id: FTE}

        Returns
        -------
        DataFrame: product_id, current_fte, fte_reduction, new_fte, marginal_roi
        """
        rows = []
        weights = {}

        for pid in donor_products:
            cur_fte = self.current_fte.get(pid, 0.0)
            min_fte = (
                min_fte_override.get(pid, cur_fte * self.min_fte_ratio)
                if min_fte_override
                else cur_fte * self.min_fte_ratio
            )
            slack = max(0.0, cur_fte - min_fte)
            mroi  = self.marginal_roi(pid)
            # 限界ROIが高い品目は削りにくい → 逆数で重み付け
            inv_mroi = (1.0 / mroi) if (mroi > 0 and not np.isinf(mroi)) else 0.0
            weights[pid] = inv_mroi * slack

        total_weight = sum(weights.values())

        for pid in donor_products:
            cur_fte = self.current_fte.get(pid, 0.0)
            min_fte = (
                min_fte_override.get(pid, cur_fte * self.min_fte_ratio)
                if min_fte_override
                else cur_fte * self.min_fte_ratio
            )
            slack = max(0.0, cur_fte - min_fte)

            if total_weight > 0:
                raw_reduction = new_product_fte * (weights[pid] / total_weight)
            else:
                raw_reduction = new_product_fte / len(donor_products)

            reduction = min(raw_reduction, slack)
            rows.append({
                "product_id":   pid,
                "current_fte":  round(cur_fte, 1),
                "fte_reduction": round(reduction, 1),
                "new_fte":      round(cur_fte - reduction, 1),
                "marginal_roi": round(self.marginal_roi(pid), 6),
                "slack_fte":    round(slack, 1),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("fte_reduction", ascending=False)
            .reset_index(drop=True)
        )


# ============================================================
# Module 6: メイン FTE 算出クラス
# ============================================================

class FY2029FTECalculator:
    """
    複数年度（FY2026〜FY2029等）の品目×月別FTEを算出するメインクラス。
    target_months を指定することで任意の期間を計算可能。
    省略時は FY2029（後方互換）。

    算出フロー:
      1. ターゲット医師数（月別）
      2. FC/SC分割
      3. 活動頻度（ライフサイクル調整済み）
      4. 必要コール数 = FC × freq × 1.0 + SC × freq × SC_COEFFICIENT
      5. FTE = 必要コール数 ÷ (calls/day × working_days)
      6. MR/Digital比率を乗じてMR-FTE / Digital-FTEに分割

    主要メソッド:
      run()                         : 全品目×全月のFTEを算出
      run_with_new_product()        : 新品目FTE配分も含めて算出
      summarize_fy()                : 年度集計（品目別平均・最大FTE）
    """

    def __init__(
        self,
        product_configs: List[ProductConfig],
        target_doctor_calc: TargetDoctorCalculator,
        fc_sc_allocator: FCScAllocator,
        freq_estimator: ActivityFrequencyEstimator,
        product_info: pd.DataFrame,
        frequency_mode: str = "lifecycle_adjusted",
        new_product_ramp_up: Optional[Dict[str, List[float]]] = None,
        # {product_id: [発売0ヶ月目浸透率, 1ヶ月目, ...]}
        reference_products: Optional[Dict[str, str]] = None,
        # {新品目ID: 参照する類似既存品目ID}
        target_months: Optional[List[str]] = None,
        # 計算対象月リスト。None → FY2029のみ（後方互換）
        competition_schedule: Optional[Dict[str, List[Dict]]] = None,
        # {product_id: [{"launch_ym": "2028-04", "intensity": 1.25, "boost_months": 18}, ...]}
        # 競合品発売スケジュール。競合が多い品目ほど高いFTEを割り当てる。
        competition_count_factor: float = 0.05,
        # 発売済み競合品 1 品目あたりの FTE 増加率。デフォルト 0.05 = 5%/品目。
        # 例: 競合 2 品が発売済みなら 1 + 2×0.05 = 1.10（10%増）を launch-boost に乗算する。
        # 0.0 にすると競合品数カウントを無効化（launch-boost のみ）。
        supply_restrictions: Optional[Dict[str, List[Dict]]] = None,
        # {product_id: [{"start_ym": "2026-02", "end_ym": "2027-03", "factor": 0.65}, ...]}
        # 供給制限スケジュール。制限期間中はFTEを factor 倍に削減する。
        working_days_map: Optional[Dict[str, int]] = None,
        # {YYYY-MM: 実稼働日数}
        competitor_yearly: Optional[Dict[Tuple[str, str], int]] = None,
        # {(product_id, "FY2027"): n_direct_competitors}
        # 効能効果が同じ直接競合品の年度別数。n > 0 なら FTE に count_boost を乗算する。
    ) -> None:
        self.configs = {p.product_id: p for p in product_configs}
        self.target_doctor_calc = target_doctor_calc
        self._target_months = target_months
        self.fc_sc_allocator = fc_sc_allocator
        self.freq_estimator = freq_estimator
        self.product_info = product_info
        self.frequency_mode = frequency_mode
        self.new_product_ramp_up = new_product_ramp_up or {}
        self.reference_products = reference_products or {}
        self.competition_schedule = competition_schedule or {}
        self.competition_count_factor = competition_count_factor
        self.supply_restrictions = supply_restrictions or {}
        self.working_days_map = working_days_map or {}
        self.competitor_yearly: Dict[Tuple[str, str], int] = competitor_yearly or {}

    @property
    def target_months(self) -> List[str]:
        """計算対象月リスト。未指定時はFY2029のみ（後方互換）"""
        return self._target_months if self._target_months else FY2029_MONTHS

    # ----------------------------------------------------------
    # ヘルパー
    # ----------------------------------------------------------

    def _indication_boost(self, config: ProductConfig, month: str) -> float:
        """
        効能追加によるFTEブースト係数を返す（第1・第2効能の積）。

        各効能のブースト（ブースト期間中は線形に減衰）:
          boost_i = 1 + (fte_boost_i - 1) × (1 - elapsed_i / boost_months_i)

        最終ブースト = boost_1 × boost_2
        効能追加前・ブースト期間外は 1.0（中立）を返す。
        """
        def _single_boost(add_ym, fte_boost, boost_months) -> float:
            if not add_ym or boost_months <= 0:
                return 1.0
            elapsed = _months_between(add_ym, month)
            if elapsed < 0 or elapsed >= boost_months:
                return 1.0
            t = elapsed / boost_months
            return 1.0 + (fte_boost - 1.0) * (1.0 - t)

        b1 = _single_boost(config.indication_add_ym,   config.indication_fte_boost,   config.indication_boost_months)
        b2 = _single_boost(config.indication_add_ym_2, config.indication_fte_boost_2, config.indication_boost_months_2)
        return b1 * b2

    def _competition_boost(self, pid: str, month: str) -> float:
        """
        競合品の状況に応じた FTE ブーストを返す（2要素の積）。

        ① 発売タイミングブースト（launch_boost）:
            競合発売直後が最大、boost_months かけて線形減衰して 1.0 へ戻る。
            複数競合が重なる場合は最大値を採用。
            formula: 1 + (intensity - 1) × (1 - elapsed/boost_months)

        ② 競合品数ブースト（count_boost）:
            同効能の直接競合品数に比例して FTE を積み増す。
            優先順位: competitor_yearly（年度別指定） > competition_schedule（イベント集計）
            formula: 1 + n_direct × competition_count_factor
            例: 直接競合 2 品 × factor=0.05 → 1.10（10%増）

        最終ブースト = launch_boost × count_boost
        """
        # ① 発売タイミングブースト（competition_schedule から）
        competitors = self.competition_schedule.get(pid, [])
        launch_boost = 1.0
        n_from_schedule = 0

        for comp in competitors:
            elapsed = _months_between(comp["launch_ym"], month)
            if elapsed < 0:
                continue  # まだ未発売
            n_from_schedule += 1
            if elapsed < comp["boost_months"]:
                t = elapsed / comp["boost_months"]
                boost = 1.0 + (comp["intensity"] - 1.0) * (1.0 - t)
                launch_boost = max(launch_boost, boost)

        # ② 競合品数ブースト
        # competitor_yearly に年度別データがあればそちらを優先
        fy = month_to_fy(month)
        n_yearly = self.competitor_yearly.get((pid, fy), None)
        n_active = n_yearly if n_yearly is not None else n_from_schedule

        count_boost = 1.0 + n_active * self.competition_count_factor

        # competition_schedule も competitor_yearly も空なら 1.0 を返す
        if not competitors and not self.competitor_yearly.get((pid, fy)):
            return 1.0

        return launch_boost * count_boost

    def _supply_restriction_factor(self, pid: str, month: str) -> float:
        """
        供給制限期間中のFTE削減係数を返す（制限なし = 1.0）。

        供給制限中は処方が制限されるため、MR訪問の効果が低下する。
        FTEを factor 倍（例: 0.65）に削減してリソースを代替品に振り向ける。
        """
        for r in self.supply_restrictions.get(pid, []):
            if r["start_ym"] <= month <= r["end_ym"]:
                return float(r["factor"])
        return 1.0
    # ----------------------------------------------------------
    # メイン算出（2パス方式）
    # ----------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        全品目×指定月のFTEを算出して返す。

        FTE = MR headcount のみ（デジタルはFTEに含まない）。

        2パス方式:
          Pass1: base_fte（全活動コスト、効能追加ブースト込み）を計算
          Pass2: エリア×月別の平均base_fteを基準にMR比率を決定
                 required_fte = base_fte × mr_ratio（MR headcountのみ）

        Returns
        -------
        DataFrame: product_id, month, area, target_doctors,
                   fc_doctors, sc_doctors, visit_frequency,
                   required_calls, base_fte, required_fte,
                   mr_ratio, digital_ratio, mr_fte, digital_fte
        """
        from collections import defaultdict

        # ---- Step 1: 全品目の月別ターゲット医師数を取得 ----
        all_target: Dict[str, Dict[str, int]] = {}
        for pid, config in self.configs.items():
            ramp = self.new_product_ramp_up.get(pid)
            ref  = self.reference_products.get(pid)
            df = self.target_doctor_calc.calculate(
                pid, self.target_months, config=config if config.is_new else None,
                ramp_up_curve=ramp, reference_product_id=ref,
            )
            all_target[pid] = dict(zip(df["month"], df["target_doctors"]))

        # ---- Pass 1: base_fte（全活動コスト＋効能追加ブースト）----
        pass1_records = []
        for pid, config in self.configs.items():
            area = config.area

            for month in self.target_months:
                total_target = all_target[pid].get(month, 0)
                working_days = self.working_days_map.get(month, WORKING_DAYS_PER_MONTH)
                monthly_capacity = CALLS_PER_DAY[area] * working_days

                # ---- FC/SC 訪問コスト重み（医師の分類ではなく品目の訪問種別） ----
                # FC = 主訪問（全コスト）、SC = 他品目訪問に内包（低コスト）
                # 同一医師に対してTRIはFC・INTはSCという形で活動する
                fc_weight = self.fc_sc_allocator.get_fc_weight(pid, month=month)
                fc_ratio  = self.fc_sc_allocator.get_fc_ratio(pid, month=month)

                # ---- R/W 医師ティアによる required_calls 計算 ----
                tier = self.target_doctor_calc.get_doctor_tier(pid, month=month)
                if tier and total_target > 0:
                    # R/W比率を維持しつつ ramp-up 等の総数変動に追随
                    total_base = tier["r_doctors"] + tier["w_doctors"]
                    ratio = total_target / total_base if total_base > 0 else 1.0
                    r_docs = int(round(tier["r_doctors"] * ratio))
                    w_docs = total_target - r_docs
                    # visit_freq.csv の目標頻度×達成率で実効頻度を決定
                    r_eff, r_tgt, r_ach = self.target_doctor_calc.get_visit_freq(pid, month, "R")
                    w_eff, w_tgt, w_ach = self.target_doctor_calc.get_visit_freq(pid, month, "W")
                    # visit_freq.csv がある場合は lc_adj 不要（年度別で制御済み）
                    # ない場合（新製品フォールバック）は lc_adj を適用
                    if (pid, month_to_fy(month), "R") in self.target_doctor_calc.visit_freq_data:
                        r_freq = r_eff
                        w_freq = w_eff
                    else:
                        lc_adj = self.freq_estimator._lifecycle_adj(pid, month)
                        r_freq = r_eff * lc_adj
                        w_freq = w_eff * lc_adj
                    raw_calls = r_docs * r_freq + w_docs * w_freq
                    eff_freq  = raw_calls / total_target if total_target > 0 else 0.0
                else:
                    # ティアデータなし（新製品等）: lifecycle_adjusted頻度
                    r_docs = total_target
                    w_docs = 0
                    freq   = self.freq_estimator.get(pid, month, self.frequency_mode)
                    r_freq = w_freq = freq
                    raw_calls = total_target * freq
                    eff_freq  = freq
                    r_tgt, r_ach = freq, 1.0
                    w_tgt, w_ach = freq, 1.0

                # FC/SC重みを適用（SCが多い品目ほど必要コールを割引）
                required_calls = raw_calls * fc_weight

                base_fte = required_calls / monthly_capacity if monthly_capacity > 0 else 0.0

                # 効能追加ブースト（例: VYV 2028-07〜 30%増）
                boost = self._indication_boost(config, month)
                # 競合品ブースト（競合発売直後に活動強化が必要）
                comp_boost = self._competition_boost(pid, month)
                # 供給制限ファクター（例: GLI FY2026 供給不足 → 0.65倍）
                supply_factor = self._supply_restriction_factor(pid, month)
                base_fte *= boost * comp_boost * supply_factor

                pass1_records.append({
                    "product_id":        pid,
                    "month":             month,
                    "area":              area,
                    "working_days":      working_days,
                    "target_doctors":    total_target,
                    "r_doctors":         r_docs,
                    "w_doctors":         w_docs,
                    "r_target_freq":     round(r_tgt, 3),
                    "r_achievement":     round(r_ach, 3),
                    "w_target_freq":     round(w_tgt, 3),
                    "w_achievement":     round(w_ach, 3),
                    "fc_ratio":          round(fc_ratio, 3),
                    "fc_weight":         round(fc_weight, 3),
                    "visit_frequency":   round(eff_freq, 3),
                    "required_calls":    round(required_calls, 1),
                    "base_fte":          base_fte,
                })

        # ---- Pass 2: required_fte = base_fte（MR FTE = 全活動FTE、デジタルは独立分析）----
        records = []
        for rec in pass1_records:
            records.append({
                **rec,
                "required_fte": round(rec["base_fte"], 2),
            })

        return pd.DataFrame(records)

    def run_with_new_product(
        self,
        new_product_ids: List[str],
        donor_products: List[str],
        new_product_fte_allocator: NewProductFTEAllocator,
        min_fte_override: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        新品目FTEをドナー品目から配分し、調整後のFTE表を返す。

        Returns
        -------
        (fte_df, allocation_df)
          fte_df       : run() の結果にドナー削減後のFTEを反映したもの
          allocation_df: NewProductFTEAllocator.allocate() の結果
        """
        fte_df = self.run()

        # 新品目に必要な平均FTE（月平均）
        new_fte_needed = (
            fte_df[fte_df["product_id"].isin(new_product_ids)]["required_fte"]
            .mean()
        )

        allocation_df = new_product_fte_allocator.allocate(
            new_fte_needed, donor_products, min_fte_override
        )

        # ドナー品目の required_fte を削減後に更新
        reduction_map = dict(
            zip(allocation_df["product_id"], allocation_df["fte_reduction"])
        )
        fte_df = fte_df.copy()
        fte_df["fte_reduction"] = fte_df["product_id"].map(reduction_map).fillna(0.0)
        fte_df["adjusted_fte"]  = (fte_df["required_fte"] - fte_df["fte_reduction"]).clip(lower=0)
        # FTEはMRのみ: adjusted_mr_fte = adjusted_fte
        fte_df["adjusted_mr_fte"] = fte_df["adjusted_fte"]

        return fte_df, allocation_df

    # ----------------------------------------------------------
    # 集計ユーティリティ
    # ----------------------------------------------------------

    def summarize_fy(self, fte_df: pd.DataFrame) -> pd.DataFrame:
        """
        品目×年度で集計（各年度に対応）。
        adjusted_fte が存在する場合はそちらを使用。
        """
        fte_col = "adjusted_fte" if "adjusted_fte" in fte_df.columns else "required_fte"

        fte_df = fte_df.copy()
        # 月文字列から自動でFYを判定（マルチ年度対応）
        fte_df["fiscal_year"] = fte_df["month"].map(month_to_fy)

        summary = (
            fte_df.groupby(["product_id", "area", "fiscal_year"])
            .agg(
                avg_required_fte=(fte_col, "mean"),
                max_required_fte=(fte_col, "max"),
            )
            .reset_index()
            .round(2)
        )
        return summary

    def run_with_dynamic_new_product(
        self,
        new_product_ids: List[str],
        donor_products: List[str],
        new_product_fte_allocator: "NewProductFTEAllocator",
        min_fte_override: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        新品目の成長（ランチカーブ）に比例して月別にドナー品目からFTEを段階移動する。

        従来の run_with_new_product は平均FTEで一括削減するが、
        このメソッドは「新品目が育つほど既存品目が段階的に削られる」動きを表現する。

        アルゴリズム:
          1. 全品目の生のFTEを算出 (run())
          2. 新品目の月別FTEを計算し、ピーク時の移動量を決定
          3. 各月のドナー削減量 = ピーク削減量 × (当月新品目FTE / ピークFTE)
             → 新品目が成長するにつれて削減量が増える
          4. FY2026-2028 の既存品目は従来通り（削減なし）
             FY2029以降から段階的に適用するオプションあり

        Returns
        -------
        (fte_df, peak_allocation_df)
        """
        fte_df = self.run()
        fte_col = "required_fte"

        # 新品目の月別FTE（全新品目合算）
        new_fte_by_month = (
            fte_df[fte_df["product_id"].isin(new_product_ids)]
            .groupby("month")[fte_col]
            .sum()
        )
        new_fte_peak = new_fte_by_month.max()

        if new_fte_peak <= 0:
            fte_df["fte_reduction"] = 0.0
            fte_df["adjusted_fte"] = fte_df[fte_col]
            fte_df["adjusted_mr_fte"] = fte_df[fte_col]
            return fte_df, pd.DataFrame()

        # ピーク時の配分先を確定（限界ROI逆数按分）
        peak_allocation_df = new_product_fte_allocator.allocate(
            new_fte_peak, donor_products, min_fte_override
        )
        peak_reduction = dict(
            zip(peak_allocation_df["product_id"], peak_allocation_df["fte_reduction"])
        )

        # 月別の移行進捗率（0〜1）= 当月新品目FTE ÷ ピークFTE
        ramp_rate = (new_fte_by_month / new_fte_peak).clip(0, 1)

        # 月別ドナー削減量を適用
        fte_df = fte_df.copy()
        fte_df["fte_reduction"] = fte_df.apply(
            lambda r: peak_reduction.get(r["product_id"], 0.0)
                      * ramp_rate.get(r["month"], 0.0)
            if r["product_id"] in donor_products else 0.0,
            axis=1,
        )
        fte_df["adjusted_fte"] = (fte_df[fte_col] - fte_df["fte_reduction"]).clip(lower=0)
        # FTEはMRのみ: adjusted_mr_fte = adjusted_fte
        fte_df["adjusted_mr_fte"] = fte_df["adjusted_fte"]

        return fte_df, peak_allocation_df

    def total_fte_by_area(self, fte_df: pd.DataFrame) -> pd.DataFrame:
        """領域×月別の合計FTE（全品目合算）。現行MR数との比較用。"""
        fte_col = "adjusted_fte" if "adjusted_fte" in fte_df.columns else "required_fte"
        grouped = (
            fte_df.groupby(["area", "month"])[fte_col]
            .sum()
            .reset_index()
            .rename(columns={fte_col: "total_fte"})
        )
        grouped["fiscal_year"] = grouped["month"].map(month_to_fy)
        grouped["current_mr"] = grouped["area"].map(CURRENT_MR_COUNT)
        grouped["fte_gap"] = grouped["total_fte"] - grouped["current_mr"]
        return grouped.round(2)

    def total_fte_by_area_fy(self, fte_df: pd.DataFrame) -> pd.DataFrame:
        """領域×年度別の平均合計FTE（複数年度集計）。"""
        monthly = self.total_fte_by_area(fte_df)
        fy_agg = (
            monthly.groupby(["area", "fiscal_year"])
            .agg(
                avg_total_fte=("total_fte", "mean"),
                max_total_fte=("total_fte", "max"),
                current_mr=("current_mr", "first"),
                avg_gap=("fte_gap", "mean"),
            )
            .reset_index()
            .round(2)
        )
        return fy_agg
