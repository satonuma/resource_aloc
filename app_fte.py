"""
FTE シミュレーター — Streamlit アプリ
====================================
起動方法:
    streamlit run app_fte.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── パス設定 ──────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
INPUT_DIR = ROOT / "data" / "input"
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from fy2029_fte_calculator import (  # noqa: E402
    ActivityFrequencyEstimator,
    FCScAllocator,
    FY2029FTECalculator,
    NewProductFTEAllocator,
    ProductConfig,
    TargetDoctorCalculator,
    apply_product_fte_caps,
    discretize_fte_semiannually,
    get_fy_months,
    month_to_half,
    normalize_fte_roi_weighted,
)
from bundling_analyzer import (  # noqa: E402
    compute_bundling_supply,
    compute_bundling_by_half,
    compute_bundling_pairs,
    DEFAULT_COL_MAP as _BUNDLING_DEFAULT_COL_MAP,
)

# ── 定数 ──────────────────────────────────────────────────────────────────
ALL_MONTHS = sum([get_fy_months(fy) for fy in range(2026, 2036)], [])
ALL_FY     = [f"FY{y}" for y in range(2026, 2036)]

# run_fy2029.py と同じランチカーブ（変更不要のため定数として保持）
_OVE_RAMP = [
    1.00,0.98,0.96,0.94,0.92,0.90, 0.88,0.86,0.84,0.83,0.82,0.81,
    0.80,0.79,0.78,0.77,0.76,0.75, 0.74,0.73,0.72,0.72,0.71,0.71,
    0.70,0.70,0.69,0.69,0.69,0.68, 0.68,0.68,0.67,0.67,0.67,0.67,
    0.66,0.66,0.66,0.66,0.65,0.65, 0.64,0.63,0.63,0.62,0.61,0.61,
    0.60,0.59,0.59,0.58,0.57,0.57, 0.56,0.55,0.55,0.54,0.53,0.53,
    0.52,0.51,0.51,0.50,0.49,0.49, 0.48,0.47,0.47,0.46,0.45,0.45,
    0.44,0.43,0.43,0.42,0.41,0.41, 0.40,0.40,0.39,0.39,0.38,0.38,
    0.37,0.37,0.36,0.36,0.35,0.35, 0.35,0.35,0.35,0.35,0.35,0.35,
    0.35,0.35,0.35,0.35,0.35,0.35, 0.35,0.35,0.35,0.35,0.35,0.35,
    0.35,0.35,0.35,0.35,0.35,0.35, 0.35,0.35,
]
_ZASO_RAMP = [
    1.00,0.98,0.96,0.94,0.92,0.90, 0.88,0.86,0.84,0.83,0.82,0.81,
    0.80,0.79,0.78,0.77,0.76,0.75, 0.74,0.73,0.72,0.72,0.71,0.71,
    0.70,0.70,0.69,0.69,0.69,0.68, 0.68,0.68,0.67,0.67,0.67,0.67,
    0.66,0.66,0.66,0.66,0.65,0.65, 0.64,0.63,0.62,0.61,0.60,0.59,
    0.58,0.57,0.56,0.55,0.54,0.53, 0.52,0.51,0.50,0.49,0.48,0.47,
    0.46,0.45,0.44,0.43,0.42,0.41, 0.40,0.40,0.40,0.40,0.40,0.40,
    0.40,0.40,0.40,0.40,0.40,0.40, 0.40,0.40,0.40,0.40,0.40,0.40,
    0.40,0.40,0.40,0.40,0.40,0.40, 0.40,0.40,0.40,0.40,0.40,0.40,
    0.40,0.40,0.40,0.40,0.40,0.40, 0.40,0.40,0.40,0.40,0.40,0.40,
]
_TAK_RAMP = [
    1.00,0.98,0.96,0.94,0.92,0.90, 0.88,0.86,0.84,0.83,0.82,0.81,
    0.80,0.79,0.78,0.77,0.76,0.75, 0.74,0.73,0.72,0.72,0.71,0.71,
    0.70,0.70,0.69,0.69,0.69,0.68, 0.68,0.68,0.67,0.67,0.67,0.67,
    0.66,0.66,0.66,0.66,0.65,0.65, 0.63,0.61,0.60,0.58,0.57,0.55,
    0.54,0.52,0.51,0.50,0.50,0.50, 0.50,0.50,0.50,0.50,0.50,0.50,
    0.50,0.50,0.50,0.50,0.50,0.50, 0.50,0.50,0.50,0.50,0.50,0.50,
    0.50,0.50,0.50,0.50,0.50,0.50, 0.50,0.50,0.50,0.50,0.50,0.50,
]
_GLO_RAMP  = [1.00] * 14


# ══════════════════════════════════════════════════════════════════════════
# データ読み込み（変更しないファイルはキャッシュ）
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_static() -> dict:
    """頻繁に変えないデータを一度だけ読む。"""
    def _read(name: str) -> pd.DataFrame:
        p = INPUT_DIR / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    act_raw = _read("activity_data.csv")
    if "activity_date" in act_raw.columns and "activity_ym" not in act_raw.columns:
        act_raw["activity_ym"] = act_raw["activity_date"].str[:7]
        activity_data = (
            act_raw.groupby(["activity_ym", "product_id", "doctor_id"])
            .size().reset_index(name="activity_count")
        )
    else:
        activity_data = act_raw

    fc_ratio_df = _read("fc_sc_ratio.csv")
    if "fiscal_year" not in fc_ratio_df.columns:
        fc_ratio_df["fiscal_year"] = "default"

    fte_csv = _read("current_fte.csv")
    fy2026_apr_fte = dict(zip(fte_csv["product_id"], fte_csv["current_fte"].astype(float))) if not fte_csv.empty else {}

    wd_csv = _read("working_days.csv")
    working_days_map: Dict[str, int] = {}
    if not wd_csv.empty and "year_month" in wd_csv.columns:
        working_days_map = dict(zip(wd_csv["year_month"].astype(str), wd_csv["working_days"].astype(int)))

    yearly_csv = _read("target_doctor_yearly.csv")
    yearly_doctor_counts: Dict[Tuple[str, str], Tuple[int, int]] = {}
    if not yearly_csv.empty:
        for _, row in yearly_csv.iterrows():
            key = (str(row["product_id"]).strip(), str(row["fiscal_year"]).strip())
            yearly_doctor_counts[key] = (int(row["r_doctors"]), int(row["w_doctors"]))

    supply_csv = _read("supply_restriction.csv")
    supply_restrictions: Dict[str, list] = {}
    if not supply_csv.empty:
        for _, row in supply_csv.iterrows():
            pid = str(row["product_id"]).strip()
            supply_restrictions.setdefault(pid, []).append({
                "start_ym": str(row["restriction_start_ym"]).strip(),
                "end_ym":   str(row["restriction_end_ym"]).strip(),
                "factor":   float(row["restriction_factor"]),
            })

    mindscape_df = _read("mindscape_segments.csv")

    # 年度別直接競合数 {(pid, "FY2027"): n}
    comp_yearly_csv = _read("competitor_yearly.csv")
    competitor_yearly: Dict[Tuple[str, str], int] = {}
    if not comp_yearly_csv.empty:
        for _, row in comp_yearly_csv.iterrows():
            key = (str(row["product_id"]).strip(), str(row["fiscal_year"]).strip())
            competitor_yearly[key] = int(row["n_direct_competitors"])

    return dict(
        activity_data        = activity_data,
        fc_ratio_df          = fc_ratio_df,
        fy2026_apr_fte       = fy2026_apr_fte,
        working_days_map     = working_days_map,
        yearly_doctor_counts = yearly_doctor_counts,
        supply_restrictions  = supply_restrictions,
        mindscape_df         = mindscape_df,
        decay_params_df      = _read("mmm_decay_params.csv"),
        activity_set_df      = _read("activity_set.csv"),
        competitor_yearly    = competitor_yearly,
    )


def _load_editable_defaults() -> dict:
    """編集可能データのデフォルト値を CSV から読む（キャッシュなし）。"""
    def _read(name: str) -> pd.DataFrame:
        p = INPUT_DIR / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    return dict(
        target_doctors      = _read("target_doctors.csv"),
        products            = _read("products.csv"),
        competitors         = _read("competitor_schedule.csv"),
        visit_freq          = _read("visit_freq.csv"),
        competitor_yearly   = _read("competitor_yearly.csv"),
        activity_breakdown  = _read("activity_breakdown.csv"),
        activity_cost_coeff = _read("activity_cost_coeff.csv"),
    )


@st.cache_data(show_spinner=False)
def _compute_doctor_overlap() -> Dict[Tuple[str, str], float]:
    """
    target_doctor_list.csv（RWリスト）から品目×RWティア別の「共有医師比率」を計算。

    共有医師比率 = (他品目のターゲットリストにも載っている医師数) / (当品目の総ターゲット医師数)

    返り値: {(product_id, "R"): shared_ratio, (product_id, "W"): shared_ratio}
    """
    p = INPUT_DIR / "target_doctor_list.csv"
    if not p.exists():
        return {}

    df = pd.read_csv(p)
    if df.empty or "doctor_id" not in df.columns or "product_id" not in df.columns:
        return {}

    rw_col = "rw_flag" if "rw_flag" in df.columns else None

    # 全品目のユニーク医師セット（ティア問わず）
    all_other: Dict[str, set] = {}
    for pid in df["product_id"].unique():
        all_other[pid] = set(df[df["product_id"] != pid]["doctor_id"])

    result: Dict[Tuple[str, str], float] = {}
    tiers = ["R", "W"] if rw_col else ["R"]

    for pid in df["product_id"].unique():
        for tier in tiers:
            if rw_col:
                own_docs = set(df[(df["product_id"] == pid) & (df[rw_col] == tier)]["doctor_id"])
            else:
                own_docs = set(df[df["product_id"] == pid]["doctor_id"])

            if len(own_docs) == 0:
                result[(pid, tier)] = 0.0
                continue

            shared_count = len(own_docs & all_other[pid])
            result[(pid, tier)] = round(shared_count / len(own_docs), 4)

    # rw_flag がない場合は R の値を W にもコピー
    if not rw_col:
        for pid in df["product_id"].unique():
            result[(pid, "W")] = result.get((pid, "R"), 0.0)

    return result


@st.cache_data(show_spinner=False)
def _load_raw_activity(filename: str = "visit_records.csv") -> pd.DataFrame:
    """
    訪問レベルの生データを読み込む（抱き合わせ分析専用）。
    デフォルトは visit_records.csv。
    activity_data.csv は集計済み形式のため別ファイルを使用する。
    """
    p = INPUT_DIR / filename
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


@st.cache_data(show_spinner=False)
def _compute_bundling_from_activity(
    col_map_json: str,   # JSON 文字列でキャッシュキーに使う
    filename: str = "visit_records.csv",
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    訪問レベルデータから抱き合わせ供給率を計算する。

    Returns
    -------
    (supply_dict, detail_df, by_half_df, pairs_df)
      supply_dict : {product_code: supply_ratio}  FTE 計算に渡す
      detail_df   : 品目別集計
      by_half_df  : 半期別集計
      pairs_df    : ペア別共同訪問回数（上位20）
    """
    import json
    col_map = json.loads(col_map_json)

    raw_df = _load_raw_activity(filename)
    if raw_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        supply_dict, detail_df = compute_bundling_supply(raw_df, col_map)
    except ValueError:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 半期別・ペア別は表示用（エラーは無視）
    try:
        by_half_df = compute_bundling_by_half(raw_df, col_map)
    except Exception:
        by_half_df = pd.DataFrame()

    try:
        pairs_df = compute_bundling_pairs(raw_df, col_map, top_n=20)
    except Exception:
        pairs_df = pd.DataFrame()

    return supply_dict, detail_df, by_half_df, pairs_df


_ACTIVITY_RATE_COLS = [
    "solo_fc", "solo_sc", "solo_spc", "solo_other",
    "nonsolo_fc", "nonsolo_sc", "nonsolo_spc", "nonsolo_other",
]

# 全半期（FY2024-H1 〜 FY2035-H2）
_ALL_HALF_PERIODS = [
    f"FY{fy}-H{h}" for fy in range(2024, 2036) for h in (1, 2)
]


def _compute_activity_weights(
    breakdown_df: pd.DataFrame,
    coeff_df: pd.DataFrame,
) -> Dict[Tuple[str, str], float]:
    """
    activity_breakdown.csv × activity_cost_coeff.csv から
    品目×半期の実効FTE重みを計算する。

    実効重み = Σ(活動タイプ比率 × FTEコスト係数)

    FY2026-H1 以降は最終実績値を保持（forward-fill）して
    FY2035-H2 まで全半期をカバーする dict を返す。
    """
    if breakdown_df.empty or coeff_df.empty:
        return {}

    # コスト係数 dict
    coeff: Dict[str, float] = {}
    for _, row in coeff_df.iterrows():
        coeff[str(row["activity_type"]).strip()] = float(row["cost_coefficient"])

    # 半期別実効重みを計算（実績データ部分）
    known: Dict[str, Dict[str, float]] = {}   # {pid: {half: weight}}
    for _, row in breakdown_df.iterrows():
        pid  = str(row["product_id"]).strip()
        half = str(row["half_period"]).strip()
        w    = sum(float(row.get(c, 0)) * coeff.get(c, 0) for c in _ACTIVITY_RATE_COLS)
        known.setdefault(pid, {})[half] = round(w, 4)

    # forward-fill: 最終実績値で未来半期を埋める
    result: Dict[Tuple[str, str], float] = {}
    for pid, halfs in known.items():
        sorted_halfs = sorted(halfs.keys())
        for hp in _ALL_HALF_PERIODS:
            if hp in halfs:
                result[(pid, hp)] = halfs[hp]
            elif hp > sorted_halfs[-1]:
                result[(pid, hp)] = halfs[sorted_halfs[-1]]   # 最終実績値を保持
            # 実績より古い半期はスキップ（fc_weight にフォールバック）

    return result


# ── 訪問頻度 wide/long 変換ヘルパー ──────────────────────────────────────

def _vf_to_wide(vf_df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    """long 形式 → product 単位の wide 形式（編集用）。
    返却: fiscal_year | R_頻度/月 | R_達成率 | W_頻度/月 | W_達成率
    """
    sub = vf_df[vf_df["product_id"] == product_id].copy()
    r = (sub[sub["rw_tier"] == "R"]
         [["fiscal_year", "target_freq", "achievement_rate"]]
         .rename(columns={"target_freq": "R_頻度/月", "achievement_rate": "R_達成率"}))
    w = (sub[sub["rw_tier"] == "W"]
         [["fiscal_year", "target_freq", "achievement_rate"]]
         .rename(columns={"target_freq": "W_頻度/月", "achievement_rate": "W_達成率"}))
    merged = r.merge(w, on="fiscal_year", how="outer")
    merged = merged.sort_values("fiscal_year").reset_index(drop=True)
    return merged[["fiscal_year", "R_頻度/月", "R_達成率", "W_頻度/月", "W_達成率"]]


def _wide_to_long(wide_df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    """wide 形式 → long 形式（session_state 保存用）。"""
    rows = []
    for _, row in wide_df.iterrows():
        fy = str(row["fiscal_year"]).strip()
        rows.append({"product_id": product_id, "fiscal_year": fy, "rw_tier": "R",
                     "target_freq": float(row["R_頻度/月"]), "achievement_rate": float(row["R_達成率"])})
        rows.append({"product_id": product_id, "fiscal_year": fy, "rw_tier": "W",
                     "target_freq": float(row["W_頻度/月"]), "achievement_rate": float(row["W_達成率"])})
    return pd.DataFrame(rows)


def _vf_df_to_dict(vf_df: pd.DataFrame) -> Dict[Tuple[str, str, str], dict]:
    """DataFrame → TargetDoctorCalculator が受け取る辞書形式に変換。"""
    result: Dict[Tuple[str, str, str], dict] = {}
    for _, row in vf_df.iterrows():
        key = (str(row["product_id"]).strip(),
               str(row["fiscal_year"]).strip(),
               str(row["rw_tier"]).strip())
        result[key] = {
            "target_freq":      float(row["target_freq"]),
            "achievement_rate": float(row["achievement_rate"]),
        }
    return result


# ══════════════════════════════════════════════════════════════════════════
# 設定オブジェクト構築
# ══════════════════════════════════════════════════════════════════════════

def _build_product_configs(products_df: pd.DataFrame) -> Tuple[List[ProductConfig], Dict[str, str]]:
    configs: List[ProductConfig] = []
    ref_map: Dict[str, str] = {}

    for _, row in products_df.iterrows():
        try:
            launch = datetime.strptime(str(row["launch_ym"]).strip(), "%Y-%m")
            loe    = datetime.strptime(str(row["loe_ym"]).strip(),    "%Y-%m")
        except ValueError:
            continue
        loe_months = (loe.year - launch.year) * 12 + (loe.month - launch.month)

        def _fval(col, default):
            v = row.get(col, default)
            return default if (pd.isna(v) or str(v).strip() in ("", "nan", "None")) else float(v)

        def _ival(col, default):
            v = row.get(col, default)
            return default if (pd.isna(v) or str(v).strip() in ("", "nan", "None")) else int(float(v))

        def _sval(col):
            v = row.get(col, "")
            return None if (pd.isna(v) or str(v).strip() in ("", "nan", "None")) else str(v).strip()

        ind_ym   = _sval("indication_add_ym")
        ind_ym2  = _sval("indication_add_ym_2")
        max_fte  = None
        raw_mf   = row.get("max_fte", "")
        if not (pd.isna(raw_mf) or str(raw_mf).strip() in ("", "nan", "None")):
            max_fte = float(raw_mf)

        configs.append(ProductConfig(
            product_id                = str(row["product_id"]).strip(),
            area                      = str(row["area"]).strip(),
            is_new                    = str(row["is_new"]).strip().lower() == "true",
            launch_ym                 = str(row["launch_ym"]).strip(),
            loe_months                = loe_months,
            estimated_patients        = _ival("estimated_patients", 10000),
            num_indications           = _ival("num_indications", 1),
            indication_add_ym         = ind_ym,
            indication_fte_boost      = _fval("indication_fte_boost", 1.0),
            indication_boost_months   = _ival("indication_boost_months", 0),
            indication_add_ym_2       = ind_ym2,
            indication_fte_boost_2    = _fval("indication_fte_boost_2", 1.0),
            indication_boost_months_2 = _ival("indication_boost_months_2", 0),
            post_loe_factor           = _fval("post_loe_factor", 0.0),
            mindscape_target_pct      = _ival("mindscape_target_pct", 0),
            max_fte                   = max_fte,
        ))
        ref = _sval("reference_product")
        if ref:
            ref_map[str(row["product_id"]).strip()] = ref

    return configs, ref_map


def _build_competitor_yearly(df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """competitor_yearly.csv → {(product_id, "FY2027"): n_direct_competitors}"""
    result: Dict[Tuple[str, str], int] = {}
    if df is None or df.empty:
        return result
    for _, row in df.iterrows():
        pid = str(row["product_id"]).strip()
        fy  = str(row["fiscal_year"]).strip()
        try:
            n = int(row["n_direct_competitors"])
        except (ValueError, TypeError):
            n = 0
        result[(pid, fy)] = n
    return result


def _build_competition_schedule(comp_df: pd.DataFrame) -> Dict[str, List[Dict]]:
    schedule: Dict[str, List[Dict]] = {}
    if comp_df.empty:
        return schedule
    for _, row in comp_df.iterrows():
        pid = str(row["product_id"]).strip()
        try:
            ym = str(row["launch_ym"]).strip()
            # YYYY-MM 形式に正規化
            dt = datetime.strptime(ym, "%Y-%m")
            launch_ym = dt.strftime("%Y-%m")
        except ValueError:
            continue
        schedule.setdefault(pid, []).append({
            "launch_ym":    launch_ym,
            "intensity":    float(row["intensity"]),
            "boost_months": int(float(row["boost_months"])),
        })
    return schedule


def _build_doctor_tiers(td_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, dict]]:
    base: Dict[str, int] = {}
    tiers: Dict[str, dict] = {}
    for _, row in td_df.iterrows():
        pid = str(row["product_id"]).strip()
        r, w = int(row["r_doctors"]), int(row["w_doctors"])
        base[pid]  = r + w
        tiers[pid] = {
            "r_doctors":    r,
            "w_doctors":    w,
            "r_visit_freq": float(row["r_visit_freq"]),
            "w_visit_freq": float(row["w_visit_freq"]),
        }
    return base, tiers


# ══════════════════════════════════════════════════════════════════════════
# FTE 計算コア
# ══════════════════════════════════════════════════════════════════════════

def run_calculation(
    target_doctors_df       : pd.DataFrame,
    products_df             : pd.DataFrame,
    competitors_df          : pd.DataFrame,
    visit_freq_df           : pd.DataFrame,
    headcount_targets       : Dict[str, int],
    competition_count_factor: float,
    competitor_yearly_df    : Optional[pd.DataFrame] = None,
    overlap_discount        : float = 0.0,
    doctor_overlap          : Optional[Dict[Tuple[str, str], float]] = None,
    activity_weight_map     : Optional[Dict[Tuple[str, str], float]] = None,
    bundling_supply         : Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    (raw_summary, capped_summary, detail_df)
      raw_summary   : キャップなし（品目×年度 平均 required_fte）
      capped_summary: キャップあり（品目×年度 平均 adjusted_fte）
      detail_df     : 月次詳細
    """
    static = _load_static()

    product_configs, ref_map = _build_product_configs(products_df)
    competition_schedule     = _build_competition_schedule(competitors_df)
    base_target_doctors, doctor_tiers = _build_doctor_tiers(target_doctors_df)

    product_info_df = pd.DataFrame([
        {
            "product_id":         c.product_id,
            "launch_ym":          c.launch_ym,
            "loe_months":         c.loe_months,
            "estimated_patients": c.estimated_patients,
            "num_indications":    c.num_indications,
            "post_loe_factor":    c.post_loe_factor,
        }
        for c in product_configs
    ])

    target_doctor_calc = TargetDoctorCalculator(
        base_target_doctors   = base_target_doctors,
        product_info          = product_info_df,
        mindscape_segments_df = static["mindscape_df"],
        activity_data         = static["activity_data"],
        doctor_tiers          = doctor_tiers,
        yearly_doctor_counts  = static["yearly_doctor_counts"],
        visit_freq_data       = _vf_df_to_dict(visit_freq_df),
    )
    fc_sc_allocator = FCScAllocator(fc_ratio_df=static["fc_ratio_df"])
    freq_estimator  = ActivityFrequencyEstimator(
        activity_data         = static["activity_data"],
        product_info          = product_info_df,
        mmm_optimal_activities= None,
    )
    new_product_fte_allocator = NewProductFTEAllocator(
        decay_params_df      = static["decay_params_df"],
        current_fte          = static["fy2026_apr_fte"],
        current_mr_activity  = {},
        min_fte_ratio        = 0.5,
    )

    calculator = FY2029FTECalculator(
        product_configs          = product_configs,
        target_doctor_calc       = target_doctor_calc,
        fc_sc_allocator          = fc_sc_allocator,
        freq_estimator           = freq_estimator,
        product_info             = product_info_df,
        frequency_mode           = "lifecycle_adjusted",
        new_product_ramp_up      = {
            "OVE": _OVE_RAMP, "Zaso": _ZASO_RAMP,
            "TAK881": _TAK_RAMP, "GLO": _GLO_RAMP,
        },
        reference_products       = ref_map,
        target_months            = ALL_MONTHS,
        competition_schedule     = competition_schedule,
        competition_count_factor = competition_count_factor,
        supply_restrictions      = static["supply_restrictions"],
        working_days_map         = static["working_days_map"],
        competitor_yearly        = _build_competitor_yearly(competitor_yearly_df)
                                   if competitor_yearly_df is not None
                                   else static["competitor_yearly"],
        doctor_overlap           = doctor_overlap or {},
        overlap_discount         = overlap_discount,
        activity_weight_map      = activity_weight_map or {},
        bundling_supply          = bundling_supply or {},
    )

    fte_df, _ = calculator.run_with_dynamic_new_product(
        new_product_ids           = ["OVE", "Zaso", "TAK881"],
        donor_products            = ["GLI", "CUV", "HYQ", "INT", "TRI"],
        # ENT: Zaso競合は target_doctor_yearly で手動管理（動的削減なし）
        # REV, ALC, VYV, VPR: FTE小さすぎてdonor不可
        new_product_fte_allocator = new_product_fte_allocator,
    )

    # キャップなしサマリ（月次 required_fte ベース）
    raw_df = fte_df.drop(columns=["adjusted_fte", "adjusted_mr_fte", "fte_reduction"], errors="ignore")
    raw_summary = calculator.summarize_fy(raw_df)

    # キャップあり（ROI 正規化 → 半期離散化）
    # BU2 合計プールで正規化（CS/PS遺伝/PS血液 間のFTE移動を許容）
    fte_df = normalize_fte_roi_weighted(fte_df, headcount_targets, static["decay_params_df"], combined_mode=True)
    fte_df = discretize_fte_semiannually(fte_df, headcount_targets, static["decay_params_df"], combined_mode=True)
    fte_caps = {c.product_id: c.max_fte for c in product_configs if c.max_fte is not None}
    if fte_caps:
        fte_df = apply_product_fte_caps(fte_df, fte_caps)

    capped_summary = calculator.summarize_fy(fte_df)

    return raw_summary, capped_summary, fte_df


# ══════════════════════════════════════════════════════════════════════════
# ピボット生成ユーティリティ
# ══════════════════════════════════════════════════════════════════════════

def _make_pivot(summary_df: pd.DataFrame, value_col: str = "avg_required_fte") -> pd.DataFrame:
    """product_id × fiscal_year のピボットテーブルを返す。"""
    piv = (
        summary_df
        .pivot_table(index="product_id", columns="fiscal_year", values=value_col)
        .reindex(columns=ALL_FY)
        .fillna(0.0)
        .round(1)
    )
    piv.columns.name = None
    return piv


def _style_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """小数1桁に丸めたピボットを返す（matplotlib 不要）。"""
    return df.round(1)


# ══════════════════════════════════════════════════════════════════════════
# Plotly 棒グラフ
# ══════════════════════════════════════════════════════════════════════════

def _make_bar_chart(
    raw_piv   : pd.DataFrame,
    capped_piv: pd.DataFrame,
    fy        : str,
    area      : Optional[str],
    products_df: pd.DataFrame,
) -> go.Figure:
    area_pids = set(
        products_df[products_df["area"] == area]["product_id"].tolist()
        if area else products_df["product_id"].tolist()
    )
    pids = sorted(
        [p for p in raw_piv.index if p in area_pids],
        key=lambda p: raw_piv.loc[p, fy] if fy in raw_piv.columns else 0,
        reverse=True,
    )

    raw_vals    = [raw_piv.loc[p,    fy] if fy in raw_piv.columns    else 0.0 for p in pids]
    capped_vals = [capped_piv.loc[p, fy] if fy in capped_piv.columns else 0.0 for p in pids]

    fig = go.Figure()
    fig.add_bar(name="キャップなし（必要量）", x=pids, y=raw_vals,
                marker_color="#4C9BE8", opacity=0.7)
    fig.add_bar(name="キャップあり（ROI配分）", x=pids, y=capped_vals,
                marker_color="#E8824C", opacity=0.9)
    fig.update_layout(
        barmode="group",
        title=f"{fy}  品目別 FTE — {area or '全領域'}",
        xaxis_title="品目",
        yaxis_title="FTE (人)",
        legend=dict(orientation="h", y=1.05),
        height=420,
        margin=dict(t=60, b=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Plotly 100% 積み上げ棒グラフ
# ══════════════════════════════════════════════════════════════════════════

# 品目ごとに固定色を割り当てるカラーパレット（20色）
_PRODUCT_COLORS = [
    "#4C9BE8","#E8824C","#4CE882","#E84C82","#B84CE8",
    "#E8C44C","#4CE8D8","#E84C4C","#4C82E8","#82E84C",
    "#E8A04C","#4CD8E8","#D84CE8","#A0E84C","#E8D84C",
    "#4CE8A0","#E84CD8","#4CA0E8","#D8E84C","#E84CA0",
]

def _make_stacked_chart(
    piv         : pd.DataFrame,
    area        : Optional[str],
    products_df : pd.DataFrame,
    pct_mode    : bool = False,
    title_suffix: str = "",
) -> go.Figure:
    """
    品目 × 年度の FTE を積み上げ棒グラフで表示。
    pct_mode=True  → 100% 正規化（構成比）
    pct_mode=False → 絶対値（FTE 人数）
    """
    area_pids = set(
        products_df[products_df["area"] == area]["product_id"].tolist()
        if area else products_df["product_id"].tolist()
    )
    sub = piv.loc[[p for p in piv.index if p in area_pids], :]
    valid_fy = [fy for fy in ALL_FY if fy in sub.columns and sub[fy].sum() > 0]
    sub = sub[valid_fy]
    sort_fy = valid_fy[-1] if valid_fy else None
    pids = (
        sub[sort_fy].sort_values(ascending=False).index.tolist()
        if sort_fy else sub.index.tolist()
    )

    color_map = {pid: _PRODUCT_COLORS[i % len(_PRODUCT_COLORS)] for i, pid in enumerate(pids)}

    fig = go.Figure()
    for pid in pids:
        vals = [float(sub.loc[pid, fy]) if fy in sub.columns else 0.0 for fy in valid_fy]
        if pct_mode:
            hover = f"<b>{pid}</b><br>%{{x}}: %{{customdata:.1f}} FTE (%{{y:.1f}}%)<extra></extra>"
        else:
            hover = f"<b>{pid}</b><br>%{{x}}: %{{y:.1f}} FTE<extra></extra>"
        fig.add_bar(
            name=pid,
            x=valid_fy,
            y=vals,
            marker_color=color_map[pid],
            hovertemplate=hover,
            customdata=vals,
        )

    if pct_mode:
        title   = f"品目構成比（100% 積み上げ） — {area or '全領域'}{title_suffix}"
        yaxis   = dict(title="構成比 (%)", ticksuffix="%", range=[0, 100])
        barnorm = "percent"
    else:
        title   = f"品目別 FTE 積み上げ — {area or '全領域'}{title_suffix}"
        yaxis   = dict(title="FTE (人)")
        barnorm = ""

    fig.update_layout(
        barmode="relative",
        barnorm=barnorm,
        title=title,
        xaxis_title="年度",
        yaxis=yaxis,
        legend=dict(orientation="h", y=-0.25, x=0),
        height=460,
        margin=dict(t=60, b=120),
    )
    return fig


# 後方互換エイリアス
def _make_stacked_100_chart(piv, area, products_df, title_suffix=""):
    return _make_stacked_chart(piv, area, products_df, pct_mode=True, title_suffix=title_suffix)


# ══════════════════════════════════════════════════════════════════════════
# Streamlit メイン UI
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="FTE シミュレーター",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("💊 FTE シミュレーター")
    st.caption("パラメータを編集 → **計算実行** でリアルタイムに FTE を試算します。")

    # ── session_state 初期化 ──────────────────────────────────────────────
    if "defaults_loaded" not in st.session_state:
        defs = _load_editable_defaults()
        st.session_state.td_df           = defs["target_doctors"].copy()
        st.session_state.prod_df         = defs["products"].copy()
        st.session_state.comp_df         = defs["competitors"].copy()
        st.session_state.vf_df           = defs["visit_freq"].copy()
        st.session_state.comp_yr_df      = defs["competitor_yearly"].copy()
        st.session_state.act_brkdn_df    = defs["activity_breakdown"].copy()
        st.session_state.act_coeff_df    = defs["activity_cost_coeff"].copy()
        st.session_state.defaults_loaded = True
        st.session_state.results         = None
        # 抱き合わせ列名マッピング・ファイル名（初期値はデフォルト）
        st.session_state.bundling_col_map  = dict(_BUNDLING_DEFAULT_COL_MAP)
        st.session_state.bundling_filename = "visit_records.csv"

    # ── サイドバー: グローバル設定 ────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ グローバル設定")

        st.subheader("ヘッドカウント目標（人）")
        total_hc = st.number_input(
            "BU2 合計 MR 数",
            min_value=1, value=445, step=5,
            help="CS＋PS遺伝＋PS血液 の合計ヘッドカウント。"
                 "エリア間のFTE移動を許容し、この合計値でキャップを掛ける。",
        )
        st.caption("エリア間FTE移動あり・合計のみ固定")
        # エリア別内訳は正規化の参照値として保持（比率は自然に決まる）
        # headcount_targets のキーは合計モードでも維持（合計値のみ使用）
        headcount_targets = {"CS": 380, "PS遺伝": 25, "PS血液": 40}
        # 合計を BU2 設定値に合わせてスケール
        _default_total = 380 + 25 + 40  # 445
        _scale = total_hc / _default_total
        headcount_targets = {k: max(1, round(v * _scale)) for k, v in headcount_targets.items()}

        st.subheader("競合品数ブースト")
        cc_factor = st.slider(
            "競合品 1 品あたりの FTE 増加率",
            min_value=0.00, max_value=0.20, value=0.05, step=0.01,
            format="%.2f",
            help="発売済み競合品 1 品ごとに FTE を何 % 増やすか。\n"
                 "例: 0.05 → 2 品発売済みなら +10%（1+2×0.05=1.10）",
        )
        st.caption(f"2 品発売済み時: ×{1 + 2*cc_factor:.2f}  /  3 品: ×{1 + 3*cc_factor:.2f}")

        st.subheader("抱き合わせ活動ディスカウント")
        import json as _json
        _col_map_json   = _json.dumps(st.session_state.get("bundling_col_map", _BUNDLING_DEFAULT_COL_MAP))
        _bundling_file  = st.session_state.get("bundling_filename", "visit_records.csv")
        _bundling_supply_dict, _bundling_detail_df, _bundling_half_df, _bundling_pairs_df = \
            _compute_bundling_from_activity(_col_map_json, _bundling_file)
        bundling_available = len(_bundling_supply_dict) > 0
        if not bundling_available:
            st.caption("⚠️ activity_data.csv が見つからないか、訪問記録IDがありません。")
            bundling_enabled = False
        else:
            bundling_enabled = st.toggle(
                "抱き合わせ供給率を FTE に反映する",
                value=False,
                help="活動データから計算した「他品目PCに乗っかっている割合」を使い、\n"
                     "その分だけ必要コール数を削減してFTEを計算します。\n"
                     "供給率=30% → raw_calls × 0.70 として計算",
            )
            if bundling_enabled and _bundling_detail_df is not None and not _bundling_detail_df.empty:
                top5 = _bundling_detail_df.sort_values("供給率", ascending=False).head(5)
                lines = [f"{r['品目コード']}: {r['供給率']:.0%}" for _, r in top5.iterrows()]
                st.caption("供給率トップ5\n" + "  \n".join(lines))
        bundling_supply = _bundling_supply_dict if bundling_enabled else {}

        st.subheader("医師重複ディスカウント")
        doctor_overlap = _compute_doctor_overlap()
        overlap_available = len(doctor_overlap) > 0
        if not overlap_available:
            st.caption("⚠️ target_doctor_list.csv が見つかりません。割引は無効です。")
            overlap_discount = 0.0
        else:
            overlap_discount = st.slider(
                "共有医師の SC 転換率",
                min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                format="%.2f",
                help="他品目と共有している医師のうち、FC 訪問に乗じて SC（軽いコール）で\n"
                     "対応できる割合。\n"
                     "ユニーク医師は必ず独自 FC 訪問が必要（フルカウント）。\n"
                     "共有医師もFCは存在（新製品は投資優先）するが、\n"
                     "他品目の FC 訪問に SC タグオンできる分だけ FTE を効率化する。\n"
                     "FTE効率化 = shared_ratio × SC転換率 × (1 − SC係数0.1)\n"
                     "例: 共有80%, 転換率0.3 → FTE × (1 − 0.8×0.3×0.9) ≈ −21.6%",
            )
            if overlap_discount > 0:
                # 簡易サマリ表示（重複率が高い代表品目）
                sample_pids = sorted({pid for pid, _ in doctor_overlap.keys()})[:5]
                lines = []
                for pid in sample_pids:
                    rr = doctor_overlap.get((pid, "R"), 0.0)
                    ww = doctor_overlap.get((pid, "W"), 0.0)
                    lines.append(f"{pid}: R={rr:.0%} / W={ww:.0%}")
                st.caption("共有率サンプル（上位5品目）\n" + "  \n".join(lines))

        st.divider()

        run_btn = st.button("🔄 計算実行", type="primary", use_container_width=True)

        st.divider()
        st.subheader("💾 CSV リセット")
        if st.button("⚠️ 全パラメータをCSVに戻す", use_container_width=True):
            defs = _load_editable_defaults()
            st.session_state.td_df           = defs["target_doctors"].copy()
            st.session_state.prod_df         = defs["products"].copy()
            st.session_state.comp_df         = defs["competitors"].copy()
            st.session_state.vf_df           = defs["visit_freq"].copy()
            st.session_state.act_brkdn_df    = defs["activity_breakdown"].copy()
            st.session_state.act_coeff_df    = defs["activity_cost_coeff"].copy()
            st.session_state.results         = None
            st.rerun()

    # ── パラメータ編集タブ ────────────────────────────────────────────────
    tab_td, tab_vf, tab_comp, tab_comp_yr, tab_prod, tab_overlap, tab_act, tab_bundle = st.tabs([
        "🩺 ターゲット医師数",
        "📅 訪問頻度（年度別）",
        "⚔️ 競合品スケジュール",
        "📊 直接競合数（年度別）",
        "🔬 品目設定",
        "🔗 医師重複分析",
        "📋 活動構造（FC率/単独率）",
        "🔀 抱き合わせ分析",
    ])

    # -------- Tab 1: ターゲット医師数 ----------------------------------------
    with tab_td:
        st.markdown("### ターゲット医師数・訪問頻度")
        st.caption("R = Regular（週1以上）、W = Watcher（月1〜週1）")

        col_cfg_td = {
            "product_id":    st.column_config.TextColumn("品目 ID", disabled=True, width="small"),
            "r_doctors":     st.column_config.NumberColumn("R 医師数", min_value=0, step=10,  format="%d"),
            "w_doctors":     st.column_config.NumberColumn("W 医師数", min_value=0, step=50,  format="%d"),
            "r_visit_freq":  st.column_config.NumberColumn("R 頻度/月", min_value=0.0, max_value=10.0, step=0.25, format="%.2f"),
            "w_visit_freq":  st.column_config.NumberColumn("W 頻度/月", min_value=0.0, max_value=10.0, step=0.25, format="%.2f"),
        }
        edited_td = st.data_editor(
            st.session_state.td_df,
            column_config=col_cfg_td,
            use_container_width=True,
            hide_index=True,
            key="editor_td",
        )
        st.session_state.td_df = edited_td

        # クイック集計
        st.markdown("**現在の設定サマリ**")
        td_sum = edited_td.copy()
        td_sum["合計医師数"] = td_sum["r_doctors"] + td_sum["w_doctors"]
        td_sum["月次必要コール (目安)"] = (
            td_sum["r_doctors"] * td_sum["r_visit_freq"] +
            td_sum["w_doctors"] * td_sum["w_visit_freq"]
        ).round(0).astype(int)
        st.dataframe(
            td_sum[["product_id", "r_doctors", "w_doctors", "合計医師数", "月次必要コール (目安)"]],
            use_container_width=True, hide_index=True,
        )

    # -------- Tab 2: 訪問頻度（年度別） ----------------------------------------
    with tab_vf:
        st.markdown("### 訪問頻度（年度別）")
        st.caption(
            "R = Regular（週1以上）、W = Watcher（月1〜週1）　"
            "品目を選んで年度ごとの **目標頻度/月** と **達成率** を編集できます。"
        )

        # vf_df が空の場合はガイダンスを表示
        if st.session_state.vf_df.empty:
            st.warning("visit_freq.csv が見つかりません。data/input/ に配置してください。")
        else:
            prod_ids = sorted(st.session_state.vf_df["product_id"].unique())
            sel_pid = st.selectbox("品目を選択", prod_ids, key="vf_sel_pid")

            wide_df = _vf_to_wide(st.session_state.vf_df, sel_pid)

            col_cfg_vf = {
                "fiscal_year": st.column_config.TextColumn("年度", disabled=True, width="small"),
                "R_頻度/月":   st.column_config.NumberColumn(
                    "R 頻度/月", min_value=0.0, max_value=20.0, step=0.25, format="%.2f",
                    help="Regular ドクターへの月間訪問目標回数",
                ),
                "R_達成率":    st.column_config.NumberColumn(
                    "R 達成率",  min_value=0.0, max_value=1.5,  step=0.05, format="%.2f",
                    help="目標に対する実績達成率（1.0 = 100%）",
                ),
                "W_頻度/月":   st.column_config.NumberColumn(
                    "W 頻度/月", min_value=0.0, max_value=20.0, step=0.25, format="%.2f",
                    help="Watcher ドクターへの月間訪問目標回数",
                ),
                "W_達成率":    st.column_config.NumberColumn(
                    "W 達成率",  min_value=0.0, max_value=1.5,  step=0.05, format="%.2f",
                    help="目標に対する実績達成率（1.0 = 100%）",
                ),
            }

            edited_wide = st.data_editor(
                wide_df,
                column_config=col_cfg_vf,
                use_container_width=True,
                hide_index=True,
                key=f"editor_vf_{sel_pid}",
            )

            # 編集結果を long 形式に戻して session_state に反映
            new_long = _wide_to_long(edited_wide, sel_pid)
            # 他の品目の行はそのまま残す
            other = st.session_state.vf_df[st.session_state.vf_df["product_id"] != sel_pid]
            st.session_state.vf_df = pd.concat([other, new_long], ignore_index=True)

            # 品目全体サマリ（全品目の平均頻度）
            with st.expander("▸ 全品目 平均訪問頻度サマリ（現在設定）", expanded=False):
                summary_vf = (
                    st.session_state.vf_df
                    .groupby(["product_id", "rw_tier"])
                    .agg(avg_freq=("target_freq", "mean"), avg_ach=("achievement_rate", "mean"))
                    .reset_index()
                    .round(2)
                )
                st.dataframe(summary_vf, use_container_width=True, hide_index=True)

    # -------- Tab 3: 競合品スケジュール ----------------------------------------
    with tab_comp:
        st.markdown("### 競合品発売スケジュール")
        st.caption(
            "**intensity**: 発売直後のブースト倍率（例: 1.3 → 30%増）　"
            "**boost_months**: ブースト持続月数　"
            "**count_factor** はサイドバーで設定"
        )
        col_cfg_comp = {
            "product_id":      st.column_config.TextColumn("自社品目 ID", width="small"),
            "competitor_name": st.column_config.TextColumn("競合品名"),
            "launch_ym":       st.column_config.TextColumn(
                "発売年月 (YYYY-MM)",
                help="例: 2027-04",
                validate=r"^\d{4}-\d{2}$",
            ),
            "intensity":       st.column_config.NumberColumn("intensity", min_value=1.0, max_value=3.0, step=0.05, format="%.2f"),
            "boost_months":    st.column_config.NumberColumn("boost_months", min_value=1, max_value=60, step=6),
        }

        # 競合品追加ボタン
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("➕ 行を追加"):
                new_row = pd.DataFrame([{
                    "product_id": "", "competitor_name": "新競合品",
                    "launch_ym": "2028-04", "intensity": 1.2, "boost_months": 12,
                }])
                st.session_state.comp_df = pd.concat(
                    [st.session_state.comp_df, new_row], ignore_index=True
                )

        edited_comp = st.data_editor(
            st.session_state.comp_df,
            column_config=col_cfg_comp,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="editor_comp",
        )
        st.session_state.comp_df = edited_comp

        # 品目別の競合品数を表示
        if not edited_comp.empty:
            st.markdown("**品目別 競合品数（現在設定）**")
            cnt_df = (
                edited_comp.dropna(subset=["product_id"])
                .groupby("product_id")
                .size()
                .reset_index(name="競合品数")
                .sort_values("競合品数", ascending=False)
            )
            st.dataframe(cnt_df, use_container_width=True, hide_index=True)

    # -------- Tab 4: 年度別直接競合数 ------------------------------------------
    with tab_comp_yr:
        st.markdown("### 直接競合品数（効能効果が同じ品目数・年度別）")
        st.caption(
            "**n_direct_competitors**: その年度に市場にいる **同効能直接競合品の数**。  \n"
            "競合が存在すると `count_boost = 1 + n × count_factor` でFTEが増加します。  \n"
            "count_factor はサイドバーの「競合品数ブースト」スライダーで設定。"
        )

        col_cfg_cy = {
            "product_id":          st.column_config.TextColumn("品目 ID",       width="small"),
            "fiscal_year":         st.column_config.TextColumn("年度",          width="small"),
            "n_direct_competitors":st.column_config.NumberColumn(
                "直接競合数", min_value=0, max_value=20, step=1,
                help="同効能でシェア競合する外部品目の数（自社品目は含めない）",
            ),
            "note": st.column_config.TextColumn("備考", width="large"),
        }

        c1y, c2y = st.columns([3, 1])
        with c2y:
            if st.button("➕ 行を追加", key="add_comp_yr"):
                new_row = pd.DataFrame([{
                    "product_id": "", "fiscal_year": "FY2027",
                    "n_direct_competitors": 1, "note": "",
                }])
                st.session_state.comp_yr_df = pd.concat(
                    [st.session_state.comp_yr_df, new_row], ignore_index=True
                )

        # note 列が NaN(float) の場合は空文字列に変換してからエディタに渡す
        cy_display = st.session_state.comp_yr_df.copy()
        if "note" in cy_display.columns:
            cy_display["note"] = cy_display["note"].fillna("").astype(str)

        edited_cy = st.data_editor(
            cy_display,
            column_config=col_cfg_cy,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="editor_comp_yr",
        )
        st.session_state.comp_yr_df = edited_cy

        # 競合数 > 0 の行だけハイライト表示
        active_cy = edited_cy[edited_cy["n_direct_competitors"] > 0]
        if not active_cy.empty:
            st.markdown("**競合あり（n > 0）の品目・年度**")
            st.dataframe(
                active_cy.sort_values(["product_id", "fiscal_year"]),
                use_container_width=True, hide_index=True,
            )

    # -------- Tab 5: 品目設定 -----------------------------------------------
    with tab_prod:
        st.markdown("### 品目設定（主要パラメータ）")
        st.caption("is_new・launch_ym・loe_ym・post_loe_factor・max_fte などを直接編集できます。")

        # 表示列を絞る（全列だと多すぎる）
        display_cols = [
            "product_id", "area", "is_new", "launch_ym", "loe_ym",
            "post_loe_factor", "mindscape_target_pct", "max_fte",
        ]
        prod_disp = st.session_state.prod_df[[c for c in display_cols if c in st.session_state.prod_df.columns]].copy()

        col_cfg_prod = {
            "product_id":           st.column_config.TextColumn("品目 ID",     disabled=True, width="small"),
            "area":                 st.column_config.SelectboxColumn("エリア",  options=["CS","PS遺伝","PS血液"], width="small"),
            "is_new":               st.column_config.CheckboxColumn("新薬"),
            "launch_ym":            st.column_config.TextColumn("発売年月",     validate=r"^\d{4}-\d{2}$"),
            "loe_ym":               st.column_config.TextColumn("LOE 年月",     validate=r"^\d{4}-\d{2}$"),
            "post_loe_factor":      st.column_config.NumberColumn("LOE後係数",  min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
            "mindscape_target_pct": st.column_config.NumberColumn("Mindscape%", min_value=0,   max_value=100, step=5),
            "max_fte":              st.column_config.NumberColumn("FTE上限",    min_value=0.0,  step=5.0,     format="%.0f",
                                                                   help="空欄=上限なし"),
        }
        edited_prod_disp = st.data_editor(
            prod_disp,
            column_config=col_cfg_prod,
            use_container_width=True,
            hide_index=True,
            key="editor_prod",
        )
        # 編集した列を元の full DataFrame に反映
        prod_full = st.session_state.prod_df.copy()
        for col in edited_prod_disp.columns:
            if col != "product_id" and col in prod_full.columns:
                prod_full[col] = edited_prod_disp[col].values
        st.session_state.prod_df = prod_full

    # -------- Tab 6: 医師重複分析 -------------------------------------------
    with tab_overlap:
        st.markdown("### 医師重複分析（target_doctor_list.csv）")
        st.caption(
            "RW リスト上で他品目のターゲット医師と重複している比率を品目×ティア別に表示。  \n"
            "**共有医師**: 他品目との FC 訪問に乗じて SC（軽いコール）で対応できる医師。  \n"
            "**ユニーク医師**: 他品目と被っていないため必ず独自 FC 訪問が必要。"
        )

        if not overlap_available:
            st.warning(
                "target_doctor_list.csv が `data/input/` に見つかりません。  \n"
                "列: `product_id, rw_flag, doctor_id` を含む CSV を配置してください。",
                icon="⚠️",
            )
        else:
            # 品目×ティア の共有率テーブルを構築
            rows_ov = []
            for (pid, tier), ratio in sorted(doctor_overlap.items()):
                rows_ov.append({
                    "品目 ID": pid,
                    "ティア": tier,
                    "共有率": ratio,
                    "ユニーク率": round(1.0 - ratio, 4),
                })
            ov_df = pd.DataFrame(rows_ov)

            if not ov_df.empty:
                # R と W を横並びにして見やすく
                ov_wide = (
                    ov_df.pivot_table(index="品目 ID", columns="ティア", values="共有率")
                    .reset_index()
                    .rename(columns={"R": "R 共有率", "W": "W 共有率"})
                )
                ov_wide = ov_wide.sort_values("R 共有率", ascending=False).reset_index(drop=True)

                col_cfg_ov = {
                    "品目 ID": st.column_config.TextColumn("品目 ID", width="small"),
                    "R 共有率": st.column_config.ProgressColumn(
                        "R 共有率", min_value=0.0, max_value=1.0, format="%.0%",
                    ),
                    "W 共有率": st.column_config.ProgressColumn(
                        "W 共有率", min_value=0.0, max_value=1.0, format="%.0%",
                    ),
                }
                st.dataframe(ov_wide, column_config=col_cfg_ov,
                             use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("**現在の SC 転換率設定でのFTE効率化プレビュー**")
                st.caption(
                    f"SC 転換率 = {overlap_discount:.0%}  |  "
                    "効率化 = shared_ratio × SC転換率 × (1 − 0.1)"
                )
                if overlap_discount > 0:
                    prev_rows = []
                    for _, row in ov_wide.iterrows():
                        pid = row["品目 ID"]
                        r_s = row.get("R 共有率", 0.0) or 0.0
                        w_s = row.get("W 共有率", 0.0) or 0.0
                        # 簡易: R/W 等重み平均
                        avg_shared = (r_s + w_s) / 2.0
                        factor = 1.0 - avg_shared * overlap_discount * 0.9
                        prev_rows.append({
                            "品目 ID": pid,
                            "平均共有率": round(avg_shared, 3),
                            "FTE 係数": round(factor, 3),
                            "FTE 削減率": round(1.0 - factor, 3),
                        })
                    prev_df = pd.DataFrame(prev_rows).sort_values("FTE 削減率", ascending=False)
                    col_cfg_prev = {
                        "FTE 係数":   st.column_config.ProgressColumn(
                            "FTE 係数", min_value=0.0, max_value=1.0, format="%.3f"),
                        "FTE 削減率": st.column_config.ProgressColumn(
                            "FTE 削減率", min_value=0.0, max_value=0.5, format="%.1%"),
                    }
                    st.dataframe(prev_df, column_config=col_cfg_prev,
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("サイドバーで「共有医師の SC 転換率」を 0 より大きくするとプレビューが表示されます。")

    # -------- Tab 7: 活動構造（FC率/単独率）------------------------------
    with tab_act:
        st.markdown("### 活動構造データ（FC率 / 単独率）")
        st.caption(
            "実績活動データ（2024/4〜2026/5）から半期ごとに集計した品目別の活動タイプ構成比。  \n"
            "**実効FTE重み** = Σ(活動タイプ比率 × FTEコスト係数) として計算し、  \n"
            "fc_sc_ratio.csv の fc_weight をこの実績ベース値で**上書き**する。"
        )

        act_tab1, act_tab2 = st.tabs(["📊 活動構造データ（半期別）", "⚙️ FTEコスト係数"])

        # ---- 活動構造データ ----
        with act_tab1:
            if st.session_state.act_brkdn_df.empty:
                st.warning("activity_breakdown.csv が見つかりません。")
            else:
                # 品目選択
                ab_pids = sorted(st.session_state.act_brkdn_df["product_id"].unique())
                sel_ab_pid = st.selectbox("品目を選択", ab_pids, key="sel_ab_pid")

                sub_ab = st.session_state.act_brkdn_df[
                    st.session_state.act_brkdn_df["product_id"] == sel_ab_pid
                ].copy()

                # 実効重みを計算して表示列に追加
                coeff_now = dict(zip(
                    st.session_state.act_coeff_df["activity_type"],
                    st.session_state.act_coeff_df["cost_coefficient"].astype(float),
                ))
                sub_ab["実効FTE重み"] = (
                    sum(sub_ab[c] * coeff_now.get(c, 0) for c in _ACTIVITY_RATE_COLS)
                ).round(4)
                sub_ab["FC率(単独+非単独)"] = (sub_ab["solo_fc"] + sub_ab["nonsolo_fc"]).round(4)
                sub_ab["単独率(FC+SC+他)"] = (
                    sub_ab["solo_fc"] + sub_ab["solo_sc"] + sub_ab["solo_spc"] + sub_ab["solo_other"]
                ).round(4)

                display_cols_ab = ["half_period","FC率(単独+非単独)","単独率(FC+SC+他)",
                                   "solo_fc","solo_sc","solo_spc","solo_other",
                                   "nonsolo_fc","nonsolo_sc","nonsolo_spc","nonsolo_other",
                                   "実効FTE重み"]
                st.dataframe(
                    sub_ab[display_cols_ab],
                    use_container_width=True, hide_index=True,
                )

                # 全品目の実効重み（FY2026-H1 時点）サマリ
                with st.expander("▸ 全品目 実効FTE重みサマリ（FY2026-H1）", expanded=False):
                    latest = st.session_state.act_brkdn_df[
                        st.session_state.act_brkdn_df["half_period"] == "FY2026-H1"
                    ].copy()
                    latest["実効FTE重み"] = (
                        sum(latest[c] * coeff_now.get(c, 0) for c in _ACTIVITY_RATE_COLS)
                    ).round(4)
                    latest["FC率"] = (latest["solo_fc"] + latest["nonsolo_fc"]).round(3)
                    latest["単独率"] = (
                        latest["solo_fc"]+latest["solo_sc"]+latest["solo_spc"]+latest["solo_other"]
                    ).round(3)
                    summary_ab = latest[["product_id","FC率","単独率","実効FTE重み"]].sort_values(
                        "実効FTE重み", ascending=False
                    )
                    col_cfg_ab = {
                        "実効FTE重み": st.column_config.ProgressColumn(
                            "実効FTE重み", min_value=0.0, max_value=1.0, format="%.3f"),
                        "FC率": st.column_config.ProgressColumn(
                            "FC率", min_value=0.0, max_value=1.0, format="%.2%"),
                        "単独率": st.column_config.ProgressColumn(
                            "単独率", min_value=0.0, max_value=1.0, format="%.2%"),
                    }
                    st.dataframe(summary_ab, column_config=col_cfg_ab,
                                 use_container_width=True, hide_index=True)

        # ---- FTEコスト係数 ----
        with act_tab2:
            st.markdown("**活動タイプ別 FTEコスト係数**")
            st.caption(
                "各活動タイプが「単独FC訪問（=1.0）」に対してどの程度の FTE コストを持つかを設定。  \n"
                "変更すると次回計算実行時から反映されます。"
            )

            col_cfg_coeff = {
                "activity_type":    st.column_config.TextColumn("活動タイプ", disabled=True),
                "cost_coefficient": st.column_config.NumberColumn(
                    "FTEコスト係数", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
                "description":      st.column_config.TextColumn("説明", disabled=True),
            }
            edited_coeff = st.data_editor(
                st.session_state.act_coeff_df,
                column_config=col_cfg_coeff,
                use_container_width=True,
                hide_index=True,
                key="editor_act_coeff",
            )
            st.session_state.act_coeff_df = edited_coeff

            # 係数変更時のプレビュー
            coeff_preview = dict(zip(
                edited_coeff["activity_type"],
                edited_coeff["cost_coefficient"].astype(float),
            ))
            if not st.session_state.act_brkdn_df.empty:
                latest2 = st.session_state.act_brkdn_df[
                    st.session_state.act_brkdn_df["half_period"] == "FY2026-H1"
                ].copy()
                latest2["実効FTE重み"] = (
                    sum(latest2[c] * coeff_preview.get(c, 0) for c in _ACTIVITY_RATE_COLS)
                ).round(4)
                st.markdown("**現在の係数で算出した実効FTE重みプレビュー（FY2026-H1）**")
                pv = latest2[["product_id","実効FTE重み"]].sort_values("実効FTE重み", ascending=False)
                st.dataframe(
                    pv,
                    column_config={"実効FTE重み": st.column_config.ProgressColumn(
                        "実効FTE重み", min_value=0.0, max_value=1.0, format="%.3f")},
                    use_container_width=True, hide_index=True,
                )

    # -------- Tab 8: 抱き合わせ分析 ----------------------------------------
    with tab_bundle:
        st.markdown("### 🔀 抱き合わせ活動分析")
        st.caption(
            "activity_data.csv の **訪問記録ID** で同一訪問を識別し、  \n"
            "品目ごとに「他品目PC訪問に乗っかっている割合（供給率）」を計算します。  \n"
            "供給率が高い品目ほど、他品目の訪問スケジュールに依存して活動されています。"
        )

        # ── ファイル・列名設定 ────────────────────────────────────────────
        with st.expander("⚙️ データファイル・列名設定", expanded=not bundling_available):
            st.caption(
                "訪問レベルの生データ（訪問記録ID・活動大別を含む CSV）を指定してください。  \n"
                "`activity_data.csv` は集計済みのため**別ファイル**が必要です。"
            )

            _bundling_file_input = st.text_input(
                "ファイル名（data/input/ 以下）",
                value=st.session_state.get("bundling_filename", "visit_records.csv"),
                help="例: visit_records.csv　実データのファイル名を入力してください。",
                key="bnd_filename_input",
            )
            st.session_state["bundling_filename"] = _bundling_file_input

            st.markdown("**列名マッピング**")
            _cm = st.session_state.bundling_col_map
            col1, col2 = st.columns(2)
            with col1:
                _cm["visit_id"]       = st.text_input("訪問ID列",     value=_cm.get("visit_id",       "訪問記録ID"),  key="bnd_col_visit")
                _cm["product_code"]   = st.text_input("品目コード列", value=_cm.get("product_code",   "品目コード"),  key="bnd_col_prod")
            with col2:
                _cm["activity_class"] = st.text_input("活動大別列",   value=_cm.get("activity_class", "活動大別"),   key="bnd_col_act")
                _cm["date"]           = st.text_input("日付列",       value=_cm.get("date",           "日付"),       key="bnd_col_date")
            st.session_state.bundling_col_map = _cm

            st.markdown("**活動大別の値**")
            pc_raw  = st.text_input("PC値",  value="PC",  key="bnd_pc_vals")
            sc_raw  = st.text_input("SC値",  value="SC",  key="bnd_sc_vals")
            spc_raw = st.text_input("SPC値", value="SPC", key="bnd_spc_vals")

            st.markdown("**期待するファイル形式（例）**")
            st.code(
                "施設ID,施設名,医師ID,医師名,日付,活動種別,活動大別,品目名,品目コード,訪問記録ID\n"
                "H001,○○病院,D001,○○先生,2026-04-01,Detail,PC,ENT品,ENT,V0001\n"
                "H001,○○病院,D001,○○先生,2026-04-01,Detail,SPC,REV品,REV,V0001\n"
                "H002,△△クリニック,D002,△△先生,2026-04-02,Detail,PC,ENT品,ENT,V0002",
                language="text",
            )
            st.caption(
                "同じ **訪問記録ID** を持つ行が「1回の訪問」。  \n"
                "上例では V0001 の訪問に ENT(PC) と REV(SPC) が含まれる → REV の供給回数 +1"
            )

        if not bundling_available:
            _bfile_path = INPUT_DIR / st.session_state.get("bundling_filename", "visit_records.csv")
            if not _bfile_path.exists():
                st.warning(
                    f"`data/input/{_bfile_path.name}` が見つかりません。  \n"
                    "上の設定からファイル名を確認するか、ファイルを配置してください。",
                    icon="⚠️",
                )
            else:
                st.warning(
                    f"`{_bfile_path.name}` は存在しますが、指定した列が見つかりません。  \n"
                    "列名マッピングを確認してください。",
                    icon="⚠️",
                )
        else:
            bnd_tab1, bnd_tab2, bnd_tab3 = st.tabs([
                "📊 品目別 供給率",
                "📈 半期別トレンド",
                "🔗 品目ペア分析",
            ])

            with bnd_tab1:
                st.markdown("**品目別 抱き合わせ供給率（全期間集計）**")
                st.caption(
                    "**供給率** = (SCまたはSPCとして出現した回数) ÷ 総活動回数  \n"
                    "供給率 30% → その品目の訪問の 30% は他品目PCの訪問内で実施されている"
                )
                if _bundling_detail_df is not None and not _bundling_detail_df.empty:
                    disp = _bundling_detail_df.sort_values("供給率", ascending=False).reset_index(drop=True)
                    col_cfg_bnd = {
                        "供給率": st.column_config.ProgressColumn(
                            "供給率", min_value=0.0, max_value=1.0, format="%.1%"),
                        "PC回数":   st.column_config.NumberColumn("PC回数",  format="%d"),
                        "SC回数":   st.column_config.NumberColumn("SC回数",  format="%d"),
                        "SPC回数":  st.column_config.NumberColumn("SPC回数", format="%d"),
                        "供給回数": st.column_config.NumberColumn("供給回数（SC/SPC かつ非PC）", format="%d"),
                        "総活動回数": st.column_config.NumberColumn("総活動回数", format="%d"),
                    }
                    st.dataframe(disp, column_config=col_cfg_bnd,
                                 use_container_width=True, hide_index=True)

                    # FTE への影響プレビュー
                    if bundling_enabled:
                        st.markdown("---")
                        st.markdown("**FTE 削減プレビュー（供給率反映 ON 時）**")
                        st.caption("供給率分だけ raw_calls が削減されます。activity_weight_map と組み合わせた実効削減率の目安です。")
                        prev_rows = []
                        for _, row in disp.iterrows():
                            sr = float(row["供給率"])
                            prev_rows.append({
                                "品目コード": row["品目コード"],
                                "供給率":     sr,
                                "raw_calls 削減率": sr,
                                "FTE 係数": round(1.0 - sr, 3),
                            })
                        prev_bnd = pd.DataFrame(prev_rows)
                        st.dataframe(
                            prev_bnd,
                            column_config={
                                "供給率":         st.column_config.ProgressColumn("供給率",         min_value=0, max_value=1, format="%.1%"),
                                "raw_calls 削減率": st.column_config.ProgressColumn("raw_calls 削減率", min_value=0, max_value=1, format="%.1%"),
                                "FTE 係数":       st.column_config.ProgressColumn("FTE 係数",       min_value=0, max_value=1, format="%.3f"),
                            },
                            use_container_width=True, hide_index=True,
                        )
                    else:
                        st.info("サイドバーで「抱き合わせ供給率を FTE に反映する」をONにすると FTE 削減プレビューが表示されます。")

            with bnd_tab2:
                st.markdown("**半期別 供給率トレンド**")
                if _bundling_half_df is not None and not _bundling_half_df.empty:
                    pids_bnd = sorted(_bundling_half_df["品目コード"].unique())
                    sel_pid_bnd = st.multiselect(
                        "品目を選択", pids_bnd,
                        default=pids_bnd[:min(5, len(pids_bnd))],
                        key="bnd_pid_sel",
                    )
                    if sel_pid_bnd:
                        sub_half = _bundling_half_df[_bundling_half_df["品目コード"].isin(sel_pid_bnd)]
                        fig_half = go.Figure()
                        for pid in sel_pid_bnd:
                            s = sub_half[sub_half["品目コード"] == pid].sort_values("half_period")
                            fig_half.add_scatter(
                                x=s["half_period"], y=s["供給率"],
                                mode="lines+markers", name=pid,
                            )
                        fig_half.update_layout(
                            xaxis_title="半期", yaxis_title="供給率",
                            yaxis=dict(tickformat=".0%", range=[0, 1]),
                            height=380, margin=dict(t=20, b=40),
                            legend=dict(orientation="h", y=1.05),
                        )
                        st.plotly_chart(fig_half, use_container_width=True)

                        with st.expander("▸ 半期別テーブル", expanded=False):
                            st.dataframe(
                                sub_half.sort_values(["品目コード", "half_period"]),
                                use_container_width=True, hide_index=True,
                            )
                else:
                    st.info("日付列が有効であれば半期別トレンドが表示されます。")

            with bnd_tab3:
                st.markdown("**品目ペア別 共同訪問回数（上位20）**")
                st.caption(
                    "primary_product が PC のときに secondary_product が同時に SC/SPC として出現した回数。  \n"
                    "**secondary に占める割合** = secondary の全活動に対する共同訪問の比率"
                )
                if _bundling_pairs_df is not None and not _bundling_pairs_df.empty:
                    col_cfg_pair = {
                        "secondary に占める割合": st.column_config.ProgressColumn(
                            "secondary 側の供給率", min_value=0.0, max_value=1.0, format="%.1%"),
                        "共同訪問回数": st.column_config.NumberColumn("共同訪問回数", format="%d"),
                    }
                    st.dataframe(_bundling_pairs_df, column_config=col_cfg_pair,
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("データがありません。")

    st.divider()

    # ── 計算実行 ─────────────────────────────────────────────────────────
    if run_btn:
        with st.spinner("計算中… しばらくお待ちください"):
            try:
                # 活動構造ベースの実効FTE重みを計算（半期×品目）
                _act_weight_map = _compute_activity_weights(
                    st.session_state.act_brkdn_df,
                    st.session_state.act_coeff_df,
                )
                raw_s, cap_s, detail_df = run_calculation(
                    target_doctors_df       = st.session_state.td_df,
                    products_df             = st.session_state.prod_df,
                    competitors_df          = st.session_state.comp_df,
                    visit_freq_df           = st.session_state.vf_df,
                    headcount_targets       = headcount_targets,
                    competition_count_factor= cc_factor,
                    competitor_yearly_df    = st.session_state.comp_yr_df,
                    overlap_discount        = overlap_discount,
                    doctor_overlap          = doctor_overlap,
                    activity_weight_map     = _act_weight_map,
                    bundling_supply         = bundling_supply,
                )
                st.session_state.results = {
                    "raw":    raw_s,
                    "capped": cap_s,
                    "detail": detail_df,
                    "hc":     headcount_targets,
                }
                st.success("✅ 計算完了")
            except Exception as e:
                st.error(f"計算エラー: {e}")
                st.exception(e)

    # ── 結果表示 ──────────────────────────────────────────────────────────
    if st.session_state.results:
        res      = st.session_state.results
        raw_s    = res["raw"]
        cap_s    = res["capped"]
        detail   = res["detail"]

        raw_piv  = _make_pivot(raw_s,  "avg_required_fte")
        cap_piv  = _make_pivot(cap_s,  "avg_required_fte")

        st.header("📊 計算結果")

        # ── メトリクス ────────────────────────────────────────────────
        with st.expander("▸ 現在のヘッドカウント目標", expanded=False):
            hc = res["hc"]
            bu2_total = sum(hc.values())
            mc = st.columns(4)
            mc[0].metric("BU2 合計（固定）", bu2_total)
            mc[1].metric("CS（参考）",    hc.get("CS", 380))
            mc[2].metric("PS遺伝（参考）", hc.get("PS遺伝", 25))
            mc[3].metric("PS血液（参考）", hc.get("PS血液", 40))
            st.caption("エリア間FTE移動あり。合計のみ固定。")

        # ── FTE テーブル ─────────────────────────────────────────────
        res_tab1, res_tab2 = st.tabs(["📋 キャップなし（必要量）", "🎯 キャップあり（ROI 配分）"])

        # 数値列を "%.1f" 表示にする column_config を動的生成
        def _fy_col_cfg(df: pd.DataFrame) -> dict:
            return {
                col: st.column_config.NumberColumn(col, format="%.1f")
                for col in df.columns if col != "product_id"
            }

        with res_tab1:
            st.caption("活動量から積み上げた純粋な必要 FTE（ヘッドカウント制約なし）")
            piv = _style_pivot(raw_piv).reset_index()
            st.dataframe(piv, column_config=_fy_col_cfg(piv),
                         use_container_width=True, hide_index=True)

        with res_tab2:
            st.caption("ヘッドカウント目標に正規化し、ROI 等限界収益で最適配分した FTE")
            piv = _style_pivot(cap_piv).reset_index()
            st.dataframe(piv, column_config=_fy_col_cfg(piv),
                         use_container_width=True, hide_index=True)

        # ── 棒グラフ ─────────────────────────────────────────────────
        st.subheader("📈 年度 × エリア別 比較グラフ")
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            sel_fy   = st.selectbox("年度", ALL_FY, index=0, key="sel_fy")
        with gcol2:
            areas    = ["CS", "PS遺伝", "PS血液"]
            sel_area = st.selectbox("エリア", areas, key="sel_area")

        fig = _make_bar_chart(
            raw_piv, cap_piv, sel_fy, sel_area, st.session_state.prod_df
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── 100% 積み上げ構成比グラフ ────────────────────────────────
        st.subheader("📊 品目構成比（100% 積み上げ）")
        s100_col1, s100_col2, s100_col3 = st.columns(3)
        with s100_col1:
            sel_area_s100 = st.selectbox(
                "エリア", ["全領域", "CS", "PS遺伝", "PS血液"],
                key="sel_area_s100",
            )
        with s100_col2:
            sel_data_s100 = st.radio(
                "データ種別",
                ["キャップあり（ROI配分）", "キャップなし（必要量）"],
                horizontal=True,
                key="sel_data_s100",
            )
        piv_s100  = cap_piv if "キャップあり" in sel_data_s100 else raw_piv
        area_s100 = None if sel_area_s100 == "全領域" else sel_area_s100
        fig_s100  = _make_stacked_chart(piv_s100, area_s100, st.session_state.prod_df, pct_mode=True)
        st.plotly_chart(fig_s100, use_container_width=True)

        # ── 絶対値 積み上げグラフ ────────────────────────────────────
        st.subheader("📊 品目別 FTE 積み上げ（絶対値）")
        sabs_col1, sabs_col2, _ = st.columns(3)
        with sabs_col1:
            sel_area_sabs = st.selectbox(
                "エリア", ["全領域", "CS", "PS遺伝", "PS血液"],
                key="sel_area_sabs",
            )
        with sabs_col2:
            sel_data_sabs = st.radio(
                "データ種別",
                ["キャップあり（ROI配分）", "キャップなし（必要量）"],
                horizontal=True,
                key="sel_data_sabs",
            )
        piv_sabs  = cap_piv if "キャップあり" in sel_data_sabs else raw_piv
        area_sabs = None if sel_area_sabs == "全領域" else sel_area_sabs
        fig_sabs  = _make_stacked_chart(piv_sabs, area_sabs, st.session_state.prod_df, pct_mode=False)
        st.plotly_chart(fig_sabs, use_container_width=True)

        # ── 月次折れ線グラフ ─────────────────────────────────────────
        st.subheader("📉 品目別 月次推移")
        all_pids = sorted(detail["product_id"].unique())
        sel_pids = st.multiselect(
            "品目を選択（複数可）", all_pids,
            default=["ENT", "TRI", "OVE", "REV"] if all(p in all_pids for p in ["ENT","TRI","OVE","REV"]) else all_pids[:4],
            key="sel_pids",
        )
        if sel_pids:
            view_col = st.radio(
                "表示値", ["adjusted_fte（キャップあり）", "required_fte（キャップなし）"],
                horizontal=True,
            )
            val_col = "adjusted_fte" if "キャップあり" in view_col else "required_fte"

            fig2 = go.Figure()
            for pid in sel_pids:
                sub = detail[detail["product_id"] == pid].sort_values("month")
                fig2.add_scatter(
                    x=sub["month"], y=sub[val_col],
                    mode="lines", name=pid,
                )
            fig2.update_layout(
                xaxis_title="月", yaxis_title="FTE (人)",
                height=380, margin=dict(t=20, b=40),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── CSV ダウンロード ─────────────────────────────────────────
        st.subheader("💾 結果ダウンロード")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.download_button(
                "⬇️ キャップなし CSV",
                data=raw_piv.reset_index().to_csv(index=False),
                file_name="fte_raw.csv",
                mime="text/csv",
            )
        with dc2:
            st.download_button(
                "⬇️ キャップあり CSV",
                data=cap_piv.reset_index().to_csv(index=False),
                file_name="fte_capped.csv",
                mime="text/csv",
            )
        st.download_button(
            "⬇️ 月次詳細 CSV（全品目）",
            data=detail.to_csv(index=False),
            file_name="fte_detail.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("👈 パラメータを設定して **計算実行** ボタンを押してください。")


if __name__ == "__main__":
    main()
