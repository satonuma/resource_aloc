"""
FY2029 FTE算出モジュール
========================
売上予測を実現するために必要なMR数（FTE）と
MR/デジタル比率を品目×月別に算出する。

対象期間: FY2029（2029年4月 〜 2030年3月）
対象領域: CS（希少疾患以外, 2.5 call/day）/ PS（希少疾患, 1.5 call/day）

品目グループ（FY2029）:
  PDT : GLI, CUV, HYQ
  NS  : INT, TRI
  OVE : OVE （FY2029新発売）

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
    "CS": 2.5,
    "PS": 1.5,
}

CURRENT_MR_COUNT: Dict[str, int] = {
    "CS": 380,
    "PS": 45,
}

# SC訪問はFC訪問に内包されるが、このコスト係数を乗じてFTEを計上
SC_COEFFICIENT = 0.1

# 品目グループ（全製品）
PRODUCT_GROUPS: Dict[str, List[str]] = {
    "PDT": ["GLI", "CUV", "HYQ"],
    "NS":  ["INT", "TRI", "ENT"],
    "OVE": ["OVE"],
    "NEW": ["Zaso", "WSA"],       # CS新規発売品
    "CV":  ["LIV", "REV", "ALC"],
    "RS":  ["VYV", "VPR"],
    "PS":  ["LVM", "TKZ", "RPL", "VON"],
}

# 品目→領域
PRODUCT_AREA: Dict[str, str] = {
    # CS（希少疾患以外）
    "GLI": "CS", "CUV": "CS", "HYQ": "CS",
    "INT": "CS", "TRI": "CS", "ENT": "CS",
    "OVE": "CS",
    "Zaso": "CS", "WSA": "CS",
    "LIV": "CS", "REV": "CS", "ALC": "CS",
    "VYV": "CS", "VPR": "CS",
    # PS（希少疾患）
    "LVM": "PS", "TKZ": "PS", "RPL": "PS", "VON": "PS",
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
            parsed = datetime.strptime(value, fmt)
            # 2桁年はdatetimeで1900年代になるので2000年台に補正
            if fmt == "%y-%m":
                year = parsed.year
                if year < 1900:
                    parsed = parsed.replace(year=year + 100)
            return parsed
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


def normalize_fte_to_headcount(
    fte_df: pd.DataFrame,
    headcount_targets: Dict[str, int],
) -> pd.DataFrame:
    """
    各月の領域別合計FTEをヘッドカウント目標に正規化する。

    品目間の相対配分比率を保ちながら、毎月の領域合計が
    headcount_targets に一致するよう全品目を比例スケーリングする。

    Parameters
    ----------
    fte_df           : run() or run_with_dynamic_new_product() の出力
    headcount_targets: {"CS": 380, "PS": 45} など

    Returns
    -------
    adjusted_fte / adjusted_mr_fte を更新したDataFrame
    """
    fte_col = "adjusted_fte" if "adjusted_fte" in fte_df.columns else "required_fte"
    result = fte_df.copy()

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
) -> pd.DataFrame:
    """
    月次FTEを半期（6ヶ月）ステップ関数に変換する。

    各品目×半期について、期内の月平均FTEを計算し
    同じ値を期内全月に適用する（離散的な推移）。
    半期の定義: H1=4〜9月, H2=10〜3月（翌年）

    例: 4月〜9月は同じFTE値、10月〜3月は同じFTE値

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
    result = normalize_fte_to_headcount(result, headcount_targets)
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
                    & (decay_params_df["channel"] == "MR")
                ]
                if mr_rows.empty:
                    continue
                p = mr_rows.iloc[0]
                active.append((cfg.product_id, float(p["beta"]), float(p["ec"]), float(p["slope"])))

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
            for pid, fte_opt in pid_fte.items():
                rows.append({
                    "product_id":  pid,
                    "area":        area,
                    "fiscal_year": fy,
                    "optimal_fte": round(fte_opt * scale, 2),
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
    indication_add_ym: Optional[str] = None   # 効能追加年月 "YYYY-MM"
    indication_fte_boost: float = 1.0         # 効能追加時のFTEブースト倍率 (例: 1.30)
    indication_boost_months: int = 0          # ブースト持続月数
    # バイオシミラー浸食耐性パラメータ（省略可）
    post_loe_factor: float = 0.0  # LOE後に維持するライフサイクル係数（0.0=小分子完全停止, 0.55=バイオ品ENT等）


@dataclass
class MMMDecayParams:
    """Meridian出力: 品目×チャネル別減衰曲線パラメータ"""
    product_id: str
    channel: str    # "MR" | "Digital"
    alpha: float    # Adstock減衰率
    beta: float     # 応答曲線スケール係数（売上換算）
    ec: float       # EC50（半最大効果到達点）
    slope: float    # Hill係数


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

    入力データ（DataFrameのカラム）:
    - activity_data   : activity_ym, product_id, doctor_id, activity_type
    - doctor_attr     : doctor_id, age
    - product_info    : product_id, launch_ym, loe_months,
                        estimated_patients, num_indications
    """

    def __init__(
        self,
        activity_data: pd.DataFrame,
        doctor_attr: pd.DataFrame,
        product_info: pd.DataFrame,
    ) -> None:
        self.activity_data = activity_data
        self.doctor_attr = doctor_attr
        self.product_info = product_info
        self._base_target_cache: Dict[str, int] = {}

    # ----------------------------------------------------------
    # 既存品目用
    # ----------------------------------------------------------

    def get_base_target_doctors(self, product_id: str) -> int:
        """
        過去の活動データから品目のターゲット医師数（ユニーク数）を取得。
        直近12ヶ月の平均を使用。
        """
        if product_id in self._base_target_cache:
            return self._base_target_cache[product_id]

        prod_acts = self.activity_data[
            self.activity_data["product_id"] == product_id
        ]
        if prod_acts.empty:
            warnings.warn(f"{product_id}: 活動データなし、ターゲット医師数=0")
            return 0

        monthly_unique = (
            prod_acts.groupby("activity_ym")["doctor_id"]
            .nunique()
            .tail(12)
            .mean()
        )
        result = int(round(monthly_unique))
        self._base_target_cache[product_id] = result
        return result

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

        base = self.get_base_target_doctors(product_id)
        records = [
            {"product_id": product_id, "month": m, "target_doctors": base}
            for m in months
        ]
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
        新発売品ターゲット医師数の推計式:

          target(m) = base_ref
                      × (新品目患者数 / 参照品患者数)
                      × (新品目効能数 / 参照品効能数)^0.5
                      × ramp_up_curve[発売後月数]
        """
        if reference_product_id is None:
            # 参照品なし: 患者数から直接推定（患者の10%が対象医師と仮定）
            base = int(config.estimated_patients * 0.10)
        else:
            ref_info = self.product_info[
                self.product_info["product_id"] == reference_product_id
            ]
            new_info = self.product_info[
                self.product_info["product_id"] == config.product_id
            ]
            if ref_info.empty or new_info.empty:
                base = int(config.estimated_patients * 0.10)
            else:
                ref = ref_info.iloc[0]
                nw  = new_info.iloc[0]
                ref_base = self.get_base_target_doctors(reference_product_id)
                patient_ratio = nw["estimated_patients"] / max(ref["estimated_patients"], 1)
                indication_adj = np.sqrt(
                    nw["num_indications"] / max(ref["num_indications"], 1)
                )
                base = int(ref_base * patient_ratio * indication_adj)

        records = []
        for month in months:
            elapsed = _months_between(config.launch_ym, month)
            if elapsed < 0:
                doctors = 0
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
# Module 2: FC/SC オーバーラップ算出
# ============================================================

class FCScAllocator:
    """
    品目間のFC/SC構造を管理し、実効ターゲット医師数を分割する。

    セカンドコール（SC）の取り扱い:
      - FC訪問に内包される → 追加訪問コストなし
      - FTEコストは FC × SC_COEFFICIENT として計上

    FC比率の決定方法（優先順位順）:
      1. fc_ratios に品目IDが含まれる場合 → その値を直接使用（推奨）
         例: {"INT": 0.0, "TRI": 1.0} → INT=100%SC, TRI=100%FC
      2. fc_ratios にない場合 → activity_set_df + 医師リスト被り率で自動計算（フォールバック）

    fc_sc_ratio.csv の列: product_id, fc_ratio (0.0〜1.0), note
    activity_set_df の列: fc_product, sc_product
    """

    def __init__(
        self,
        activity_set_df: pd.DataFrame,
        target_doctor_lists: Dict[str, pd.Series],
        fc_ratios: Optional[Dict[str, float]] = None,
        # {product_id: FC比率 0.0〜1.0}。指定した品目は被り率計算をスキップ
    ) -> None:
        self.activity_set_df = activity_set_df
        self.target_doctor_lists = target_doctor_lists
        self.fc_ratios: Dict[str, float] = fc_ratios or {}
        self._overlap_cache: Dict[Tuple[str, str], float] = {}

    def overlap_rate(self, fc_product: str, sc_product: str) -> float:
        """
        被り率 = |FC医師 ∩ SC医師| / |SC医師|
        SC医師のうち何割がFC訪問時に同時にカバーできるかを示す。
        """
        key = (fc_product, sc_product)
        if key in self._overlap_cache:
            return self._overlap_cache[key]

        fc_set = set(self.target_doctor_lists.get(fc_product, pd.Series([])))
        sc_set = set(self.target_doctor_lists.get(sc_product, pd.Series([])))

        if not sc_set:
            rate = 0.0
        else:
            rate = len(fc_set & sc_set) / len(sc_set)

        self._overlap_cache[key] = rate
        return rate

    def split_fc_sc(
        self,
        product_id: str,
        total_target_doctors: int,
        all_target_doctors: Dict[str, int],
    ) -> Tuple[int, int]:
        """
        品目pの (FC医師数, SC医師数) を返す。

        fc_ratios に品目が含まれる場合:
          FC医師数 = round(total × fc_ratio)
          SC医師数 = total - FC医師数

        含まれない場合（フォールバック）:
          SC医師数 = Σ（被り率 × total）
          FC医師数 = total - SC医師数
        """
        # --- パターン1: 明示的FC比率が指定されている ---
        if product_id in self.fc_ratios:
            ratio = float(self.fc_ratios[product_id])
            ratio = max(0.0, min(1.0, ratio))   # 0〜1にクリップ
            fc_count = round(total_target_doctors * ratio)
            sc_count = total_target_doctors - fc_count
            return fc_count, sc_count

        # --- パターン2: 被り率から自動計算（フォールバック）---
        fc_partners = (
            self.activity_set_df[
                self.activity_set_df["sc_product"] == product_id
            ]["fc_product"]
            .tolist()
        )

        sc_count = 0
        for fc_prod in fc_partners:
            rate = self.overlap_rate(fc_prod, product_id)
            sc_count += int(total_target_doctors * rate)

        sc_count = min(sc_count, total_target_doctors)
        fc_count = total_target_doctors - sc_count
        return fc_count, sc_count


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
        else:
            return 0.80


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
        mr_row  = self.params[
            (self.params["product_id"] == product_id) & (self.params["channel"] == "MR")
        ]
        dig_row = self.params[
            (self.params["product_id"] == product_id) & (self.params["channel"] == "Digital")
        ]
        if not mr_row.empty and not dig_row.empty:
            mr  = mr_row.iloc[0]
            dig = dig_row.iloc[0]
            mr_val  = self._hill_value(mr_act,  float(mr["beta"]),  float(mr["ec"]),  float(mr["slope"]))
            dig_val = self._hill_value(dig_act, float(dig["beta"]), float(dig["ec"]), float(dig["slope"]))
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
        mr_row = self.params[
            (self.params["product_id"] == product_id)
            & (self.params["channel"] == "MR")
        ]
        if mr_row.empty:
            return 0.0
        p = mr_row.iloc[0]
        x = self.current_mr_activity.get(product_id, 0.0)
        if x <= 0:
            return float("inf")
        return DigitalEffectivenessScorer._hill_marginal(x, p["ec"], p["slope"]) * p["beta"]

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
        current_activities: Dict[str, Dict[str, float]],
        # {product_id: {"MR": 月次コール数, "Digital": 月次視聴数}}
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
        supply_restrictions: Optional[Dict[str, List[Dict]]] = None,
        # {product_id: [{"start_ym": "2026-02", "end_ym": "2027-03", "factor": 0.65}, ...]}
        # 供給制限スケジュール。制限期間中はFTEを factor 倍に削減する。
    ) -> None:
        self.configs = {p.product_id: p for p in product_configs}
        self.target_doctor_calc = target_doctor_calc
        self._target_months = target_months
        self.fc_sc_allocator = fc_sc_allocator
        self.freq_estimator = freq_estimator
        self.product_info = product_info
        self.current_activities = current_activities
        self.frequency_mode = frequency_mode
        self.new_product_ramp_up = new_product_ramp_up or {}
        self.reference_products = reference_products or {}
        self.competition_schedule = competition_schedule or {}
        self.supply_restrictions = supply_restrictions or {}

    @property
    def target_months(self) -> List[str]:
        """計算対象月リスト。未指定時はFY2029のみ（後方互換）"""
        return self._target_months if self._target_months else FY2029_MONTHS

    # ----------------------------------------------------------
    # ヘルパー
    # ----------------------------------------------------------

    def _indication_boost(self, config: ProductConfig, month: str) -> float:
        """
        効能追加によるFTEブースト係数を返す。

        ブースト期間中は線形に減衰:
          boost = 1 + (indication_fte_boost - 1) × (1 - elapsed/boost_months)

        効能追加前・ブースト期間外は 1.0 を返す。
        """
        if not config.indication_add_ym or config.indication_boost_months <= 0:
            return 1.0
        elapsed = _months_between(config.indication_add_ym, month)
        if elapsed < 0 or elapsed >= config.indication_boost_months:
            return 1.0
        t = elapsed / config.indication_boost_months
        return 1.0 + (config.indication_fte_boost - 1.0) * (1.0 - t)

    def _competition_boost(self, pid: str, month: str) -> float:
        """
        競合品の発売タイミングに応じたFTEブーストを返す。

        競合が発売された直後が最大ブースト（シェア防衛のため活動強化）、
        期間経過とともに線形に減衰して 1.0 に戻る。
        複数の競合が重なる場合は最大値（最も高い要求）を採用。

        formula:
          boost = 1 + (intensity - 1) × (1 - elapsed/boost_months)
        """
        competitors = self.competition_schedule.get(pid, [])
        if not competitors:
            return 1.0
        max_boost = 1.0
        for comp in competitors:
            elapsed = _months_between(comp["launch_ym"], month)
            if elapsed < 0 or elapsed >= comp["boost_months"]:
                continue
            t = elapsed / comp["boost_months"]
            boost = 1.0 + (comp["intensity"] - 1.0) * (1.0 - t)
            max_boost = max(max_boost, boost)
        return max_boost

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
            monthly_capacity = CALLS_PER_DAY[area] * WORKING_DAYS_PER_MONTH

            for month in self.target_months:
                total_target = all_target[pid].get(month, 0)
                target_this_month = {p: all_target[p].get(month, 0) for p in all_target}
                fc_docs, sc_docs = self.fc_sc_allocator.split_fc_sc(
                    pid, total_target, target_this_month
                )
                freq = self.freq_estimator.get(pid, month, self.frequency_mode)
                required_calls = fc_docs * freq + sc_docs * freq * SC_COEFFICIENT
                base_fte = required_calls / monthly_capacity if monthly_capacity > 0 else 0.0

                # 効能追加ブースト（例: VYV 2028-07〜 30%増）
                boost = self._indication_boost(config, month)
                # 競合品ブースト（競合発売直後に活動強化が必要）
                comp_boost = self._competition_boost(pid, month)
                # 供給制限ファクター（例: GLI FY2026 供給不足 → 0.65倍）
                supply_factor = self._supply_restriction_factor(pid, month)
                base_fte *= boost * comp_boost * supply_factor

                pass1_records.append({
                    "product_id":      pid,
                    "month":           month,
                    "area":            area,
                    "target_doctors":  total_target,
                    "fc_doctors":      fc_docs,
                    "sc_doctors":      sc_docs,
                    "visit_frequency": round(freq, 3),
                    "required_calls":  round(required_calls, 1),
                    "base_fte":        base_fte,  # ブースト込み全活動FTE
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
            fte_df["adjusted_mr_fte"] = fte_df["mr_fte"]
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
