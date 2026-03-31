
# %%
"""
FTE算出 メイン実行スクリプト（全品目・FY2026〜2035対応）
=================================================================
対象品目（13品目）:
  CS: GLI, CUV, HYQ, INT, TRI, OVE, ENT, LIV, REV, ALC, VYV, VPR
  PS: LVM（希少疾患）

対象期間: FY2026〜FY2035（120ヶ月）
OVE発売: FY2026（2026年7月想定）
TRI LOE: FY2031（2031-07）
ENT LOE: FY2032（2032-10）

出力ファイル:
  output/fy2029_fte_report.html   - メインレポート（FYトレンド含む）
  output/fy2029_sensitivity.html  - 感度分析
  output/fy2029_fte_detail.csv    - 詳細データ
  output/fy2029_summary.csv       - 年度サマリー
  output/fy2029_allocation.csv    - OVE FTE配分
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fy2029_fte_calculator import (
    ProductConfig,
    TargetDoctorCalculator,
    FCScAllocator,
    ActivityFrequencyEstimator,
    DigitalEffectivenessScorer,
    NewProductFTEAllocator,
    FY2029FTECalculator,
    _months_between,
    get_fy_months,
    normalize_fte_to_headcount,
    discretize_fte_semiannually,
    calculate_roi_optimal_fte,
    CURRENT_MR_COUNT,
)
from fy2029_html_reporter import FY2029HTMLReporter, generate_logic_document
from fy2029_sensitivity import FY2029SensitivityAnalyzer

OUTPUT_DIR = Path(__file__).parent / "output"
PRODUCTS_CSV = Path(__file__).parent / "products.csv"

# FY2026〜FY2035 の120ヶ月
ALL_MONTHS = sum([get_fy_months(fy) for fy in range(2026, 2036)], [])


def normalize_ym(value: str) -> str:
    clean = str(value).strip()
    if not clean or clean.lower() in ("nan", "none"):
        raise ValueError(f"Invalid year-month value: '{value}'")
    for fmt in ("%Y-%m", "%y-%m", "%b-%y", "%b-%Y", "%Y/%m", "%Y.%m"):
        try:
            dt = datetime.strptime(clean, fmt)
            if fmt in ("%y-%m",):
                if dt.year < 1900:
                    dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m")
        except ValueError:
            continue
    raise ValueError(f"Invalid year-month format: '{value}'")


# ============================================================
# 品目設定CSV読み込み
# ============================================================

def load_product_configs(
    csv_path: Path = PRODUCTS_CSV,
) -> Tuple[List[ProductConfig], Dict[str, str]]:
    """
    products.csv から品目設定を読み込む。

    CSVカラム:
      product_id, area, is_new, launch_ym, loe_ym,
      estimated_patients, num_indications, reference_product

    loe_months は loe_ym と launch_ym の差分から自動計算する。
    reference_product が空でない品目は新発売品として参照先を登録。

    Returns
    -------
    (configs, reference_products)
    """
    df = pd.read_csv(csv_path, dtype=str)
    configs: List[ProductConfig] = []
    reference_products: Dict[str, str] = {}

    for _, row in df.iterrows():
        launch = datetime.strptime(row["launch_ym"].strip(), "%Y-%m")
        loe    = datetime.strptime(row["loe_ym"].strip(),    "%Y-%m")
        loe_months = (loe.year - launch.year) * 12 + (loe.month - launch.month)

        # 効能追加パラメータ（CSVに列がない場合 or NaN の場合はデフォルト）
        ind_ym_raw = row.get("indication_add_ym", "")
        ind_ym = "" if pd.isna(ind_ym_raw) else str(ind_ym_raw).strip()
        ind_boost_raw = row.get("indication_fte_boost", 1.0)
        ind_boost = 1.0 if (pd.isna(ind_boost_raw) or str(ind_boost_raw).strip() == "") else float(ind_boost_raw)
        ind_months_raw = row.get("indication_boost_months", 0)
        ind_months = 0 if (pd.isna(ind_months_raw) or str(ind_months_raw).strip() == "") else int(float(ind_months_raw))

        # バイオシミラー耐性パラメータ（CSVに列がない場合はデフォルト 0.0）
        post_loe_raw = row.get("post_loe_factor", 0.0)
        post_loe = 0.0 if (pd.isna(post_loe_raw) or str(post_loe_raw).strip() == "") else float(post_loe_raw)

        mindscape_pct_raw = row.get("mindscape_target_pct", 0)
        mindscape_pct = 0 if (pd.isna(mindscape_pct_raw) or str(mindscape_pct_raw).strip() == "") else int(float(mindscape_pct_raw))

        configs.append(ProductConfig(
            product_id            = row["product_id"].strip(),
            area                  = row["area"].strip(),
            is_new                = row["is_new"].strip().lower() == "true",
            launch_ym             = row["launch_ym"].strip(),
            loe_months            = loe_months,
            estimated_patients    = int(row["estimated_patients"]),
            num_indications       = int(row["num_indications"]),
            indication_add_ym     = ind_ym if ind_ym and ind_ym.lower() not in ("", "nan", "none") else None,
            indication_fte_boost  = ind_boost,
            indication_boost_months = ind_months,
            post_loe_factor       = post_loe,
            mindscape_target_pct  = mindscape_pct,
        ))

        ref = str(row.get("reference_product", "")).strip()
        if ref and ref.lower() not in ("", "nan", "none"):
            reference_products[row["product_id"].strip()] = ref

    return configs, reference_products


# ============================================================
# データ読み込み
# ============================================================

def load_data():
    """
    全データをCSVから読み込んで返す。

    読み込むCSVファイル（同ディレクトリ）:
      activity_data.csv        - 活動データ（generate_data_csv.py で生成）
      doctor_attr.csv          - 医師属性（generate_data_csv.py で生成）
      delivery_data.csv        - 納入データ（generate_data_csv.py で生成）
      sales_forecast.csv       - 売上予測（手動編集可）
      mmm_decay_params.csv     - MMM減衰パラメータ（手動編集可）
      activity_set.csv         - FC/SC品目セット（手動編集可）
      target_doctors.csv       - 品目別ターゲット医師数（手動編集可）
      target_doctor_list.csv   - 品目別ターゲット医師リスト（被り率計算用）
      current_fte.csv          - 現在のFTE（手動編集可）

    Databricks本番環境に切り替える場合:
      pd.read_csv(...) を spark.read.table(...).toPandas() に置き換える
    """
    BASE = Path(__file__).parent

    # ---- 活動データ ----
    # 新スキーマ: activity_date, facility_id, doctor_id, product_id, activity_type（1行=1活動）
    # FTE計算モジュールが期待する旧スキーマ（activity_ym, doctor_id, product_id, activity_count）に変換
    _act_raw = pd.read_csv(BASE / "activity_data.csv")
    if "activity_date" in _act_raw.columns and "activity_ym" not in _act_raw.columns:
        _act_raw["activity_ym"] = _act_raw["activity_date"].str[:7]
        activity_data = (
            _act_raw.groupby(["activity_ym", "product_id", "doctor_id"])
            .size()
            .reset_index(name="activity_count")
        )
    else:
        activity_data = _act_raw

    # ---- 医師属性 ----
    doctor_attr = pd.read_csv(BASE / "doctor_attr.csv")

    # ---- MMM減衰パラメータ ----
    decay_params_df = pd.read_csv(BASE / "mmm_decay_params.csv")

    # ---- FC/SC活動セット ----
    activity_set_df = pd.read_csv(BASE / "activity_set.csv")

    # ---- ターゲット医師IDリスト（被り率計算用: target_doctor_list.csv）----
    tdl_path = BASE / "target_doctor_list.csv"
    if tdl_path.exists():
        tdl_df = pd.read_csv(tdl_path, dtype=str)
        target_doctor_lists: Dict[str, pd.Series] = {
            pid: pd.Series(grp["doctor_id"].unique())
            for pid, grp in tdl_df.groupby("product_id")
        }
        print(f"       → target_doctor_list.csv: {len(target_doctor_lists)} 品目 / "
              f"{sum(len(v) for v in target_doctor_lists.values())} 医師-品目ペア読み込み")
    else:
        target_doctor_lists = {}
        print("       → target_doctor_list.csv なし → 被り率計算スキップ")

    # ---- FC/SC 比率（年度別対応）----
    fc_ratio_df = pd.read_csv(BASE / "fc_sc_ratio.csv")
    # 後方互換: fiscal_year 列がない旧フォーマットの場合は default として扱う
    if "fiscal_year" not in fc_ratio_df.columns:
        fc_ratio_df["fiscal_year"] = "default"
    fc_ratios = dict(zip(fc_ratio_df[fc_ratio_df["fiscal_year"] == "default"]["product_id"],
                         fc_ratio_df[fc_ratio_df["fiscal_year"] == "default"]["fc_ratio"].astype(float)))

    # ---- 月次MR活動量（activity_data.csv から算出） ----
    # current_activities.csv は不要: activity_data から品目別月次平均コール数を直接算出
    _act_for_mr = activity_data.copy()
    if "activity_ym" not in _act_for_mr.columns and "activity_date" in _act_for_mr.columns:
        _act_for_mr["activity_ym"] = _act_for_mr["activity_date"].str[:7]
    _mr_monthly_counts = (
        _act_for_mr.groupby(["product_id", "activity_ym"])
        .size()
        .reset_index(name="count")
        .groupby("product_id")["count"]
        .mean()
    )
    current_mr_activity: Dict[str, float] = _mr_monthly_counts.to_dict()

    # ---- FY2026/4時点のFTE（シミュレーション開始時点の実績値） ----
    fte_df = pd.read_csv(BASE / "current_fte.csv")
    fy2026_apr_fte = dict(zip(fte_df["product_id"], fte_df["current_fte"].astype(float)))

    # ---- ターゲット医師数（target_doctors.csv: 既存品目の正値） ----
    td_df = pd.read_csv(BASE / "target_doctors.csv")
    if "r_doctors" in td_df.columns and "w_doctors" in td_df.columns:
        # 新形式: R/W ティア別に医師数・訪問頻度を保持
        base_target_doctors: Dict[str, int] = {
            row["product_id"]: int(row["r_doctors"]) + int(row["w_doctors"])
            for _, row in td_df.iterrows()
        }
        doctor_tiers: Dict[str, dict] = {
            row["product_id"]: {
                "r_doctors":    int(row["r_doctors"]),
                "w_doctors":    int(row["w_doctors"]),
                "r_visit_freq": float(row["r_visit_freq"]),
                "w_visit_freq": float(row["w_visit_freq"]),
            }
            for _, row in td_df.iterrows()
        }
    else:
        # 旧形式: 合計のみ（後方互換）
        base_target_doctors = dict(zip(td_df["product_id"], td_df["target_doctors"].astype(int)))
        doctor_tiers = {}

    # ---- Doctor Mindscapeセグメントデータ（新製品ターゲット医師数算出用） ----
    mindscape_path = BASE / "mindscape_segments.csv"
    mindscape_segments_df = pd.read_csv(mindscape_path) if mindscape_path.exists() else pd.DataFrame()
    if not mindscape_segments_df.empty:
        print(f"       → mindscape_segments.csv: {len(mindscape_segments_df['product_id'].unique())} 品目のセグメントデータ読み込み")

    # ---- 年度別ターゲット医師数（target_doctor_yearly.csv）----
    yearly_path = BASE / "target_doctor_yearly.csv"
    yearly_doctor_counts: Dict[Tuple[str, str], Tuple[int, int]] = {}
    if yearly_path.exists():
        yr_df = pd.read_csv(yearly_path)
        for _, row in yr_df.iterrows():
            key = (str(row["product_id"]).strip(), str(row["fiscal_year"]).strip())
            yearly_doctor_counts[key] = (int(row["r_doctors"]), int(row["w_doctors"]))
        print(f"       → target_doctor_yearly.csv: {len(yearly_doctor_counts)} 件の年度別データ読み込み")

    # ---- 訪問頻度CSV（visit_freq.csv: 品目×年度×ティア別 目標頻度・達成率）----
    vf_path = BASE / "visit_freq.csv"
    visit_freq_data: Dict[Tuple[str, str, str], dict] = {}
    if vf_path.exists():
        vf_df = pd.read_csv(vf_path)
        for _, row in vf_df.iterrows():
            key = (str(row["product_id"]).strip(),
                   str(row["fiscal_year"]).strip(),
                   str(row["rw_tier"]).strip())
            visit_freq_data[key] = {
                "target_freq":     float(row["target_freq"]),
                "achievement_rate": float(row["achievement_rate"]),
            }
        print(f"       → visit_freq.csv: {len(visit_freq_data)} 件の訪問頻度データ読み込み")

    # ---- 実稼働日数CSV（working_days.csv: 年月別 土日祝除外日数）----
    wd_path = BASE / "working_days.csv"
    working_days_map: Dict[str, int] = {}
    if wd_path.exists():
        wd_df = pd.read_csv(wd_path, dtype={"year_month": str})
        working_days_map = dict(zip(wd_df["year_month"], wd_df["working_days"].astype(int)))
        print(f"       → working_days.csv: {len(working_days_map)} ヶ月分の実稼働日数読み込み")

    # ---- 売上予測・納入データ（将来分析用として保持）----
    sales_forecast = pd.read_csv(BASE / "sales_forecast.csv")
    delivery_data  = pd.read_csv(BASE / "delivery_data.csv")

    # ---- デジタル活動データ（digital_act.csv）----
    digital_path = BASE / "digital_act.csv"
    digital_act_df: pd.DataFrame = pd.DataFrame()
    if digital_path.exists():
        digital_act_df = pd.read_csv(digital_path, dtype=str)
        print(f"       → digital_act.csv: {len(digital_act_df):,} 行読み込み（全行=視聴イベント）")
    else:
        print("       → digital_act.csv なし → デジタル活動なし")

    # ---- 供給制限スケジュール（supply_restriction.csv）----
    supply_path = BASE / "supply_restriction.csv"
    supply_restrictions: Dict[str, list] = {}
    if supply_path.exists():
        sr_df = pd.read_csv(supply_path, dtype=str)
        for _, row in sr_df.iterrows():
            pid = row["product_id"].strip()
            supply_restrictions.setdefault(pid, []).append({
                "start_ym": normalize_ym(row["restriction_start_ym"]),
                "end_ym":   normalize_ym(row["restriction_end_ym"]),
                "factor":   float(row["restriction_factor"]),
            })
        n = sum(len(v) for v in supply_restrictions.values())
        print(f"       → 供給制限: {n} 件読み込み ({', '.join(supply_restrictions.keys())})")

    return dict(
        activity_data        = activity_data,
        doctor_attr          = doctor_attr,
        digital_act          = digital_act_df,
        supply_restrictions  = supply_restrictions,
        product_info       = None,          # products.csv で管理（main()で上書き）
        decay_params_df    = decay_params_df,
        activity_set_df    = activity_set_df,
        target_doctor_lists= target_doctor_lists,
        current_mr_activity  = current_mr_activity,
        fy2026_apr_fte       = fy2026_apr_fte,
        base_target_doctors    = base_target_doctors,
        doctor_tiers           = doctor_tiers,
        yearly_doctor_counts   = yearly_doctor_counts,
        mindscape_segments_df  = mindscape_segments_df,
        sales_forecast     = sales_forecast,
        delivery_data      = delivery_data,
        fc_ratios          = fc_ratios,
        fc_ratio_df        = fc_ratio_df,
        mmm_optimal_df     = None,
        visit_freq_data      = visit_freq_data,
        working_days_map     = working_days_map,
    )


# ============================================================
# メイン処理
# ============================================================

def main():
    print("=" * 65)
    print("FY2029 FTE算出（全品目・FY2026〜FY2035）開始")
    print("=" * 65)

    # ---- 1. データ読み込み ----
    print("\n[1/6] データ読み込み...")
    data = load_data()

    # ---- 2. 品目設定（products.csv から読み込み）+ MR比率パラメータ ----
    print(f"[2/6] 品目設定（{PRODUCTS_CSV.name} から読み込み）...")
    product_configs, csv_reference_products = load_product_configs(PRODUCTS_CSV)

    # competitor_schedule.csv（競合品発売スケジュール）
    comp_path = Path(__file__).parent / "competitor_schedule.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        competition_schedule: Dict[str, List[Dict]] = {}
        for _, row in comp_df.iterrows():
            pid = str(row["product_id"]).strip()
            comp_ym_raw = row["launch_ym"]
            comp_launch_ym = normalize_ym(comp_ym_raw)
            competition_schedule.setdefault(pid, []).append({
                "launch_ym":    comp_launch_ym,
                "intensity":    float(row["intensity"]),
                "boost_months": int(float(row["boost_months"])),
            })
        print(f"       → 競合スケジュール: {sum(len(v) for v in competition_schedule.values())} 件読み込み")
    else:
        competition_schedule = {}
        print("       → competitor_schedule.csv なし → 競合ブーストなし")
    print(f"       → {len(product_configs)} 品目をロード")
    for cfg in product_configs:
        print(f"          {cfg.product_id}: {cfg.area}, LOE={cfg.loe_months}ヶ月 ({cfg.launch_ym}+{cfg.loe_months}m)")

    # product_info を CSV由来のデータで上書き（loe_monthsをCSVの値に統一）
    data["product_info"] = pd.DataFrame([
        {
            "product_id":          cfg.product_id,
            "launch_ym":           cfg.launch_ym,
            "loe_months":          cfg.loe_months,
            "estimated_patients":  cfg.estimated_patients,
            "num_indications":     cfg.num_indications,
            "post_loe_factor":     cfg.post_loe_factor,
        }
        for cfg in product_configs
    ])

    # ---- 3. モジュール初期化 ----
    print("[3/6] モジュール初期化...")

    # 既存品目: target_doctors.csv の値を直接使用
    # 新製品: mindscape_segments.csv + products.csv の mindscape_target_pct から算出
    target_doctor_calc = TargetDoctorCalculator(
        base_target_doctors   = data["base_target_doctors"],
        product_info          = data["product_info"],
        mindscape_segments_df = data["mindscape_segments_df"],
        activity_data         = data["activity_data"],   # フォールバック用
        doctor_tiers          = data["doctor_tiers"],    # R/W ティア
        yearly_doctor_counts  = data["yearly_doctor_counts"],  # 年度別医師数
        visit_freq_data       = data["visit_freq_data"],
    )

    # FC/SC = 医師の分類ではなく品目の訪問種別（fc_sc_ratio.csv で年度別指定）
    fc_sc_allocator = FCScAllocator(
        fc_ratio_df=data["fc_ratio_df"],  # 年度別対応版（fiscal_year列あり）
    )

    freq_estimator = ActivityFrequencyEstimator(
        activity_data=data["activity_data"],
        product_info=data["product_info"],
        mmm_optimal_activities=data["mmm_optimal_df"],
    )

    # ---- SOC（Share of Channel）実績データ集計 ----
    # MR活動数: activity_data の月次平均（product_id 別）
    # デジタル活動数: digital_act の月次平均（全行=視聴イベント）
    _act = data["activity_data"]
    _digi = data["digital_act"]

    _mr_monthly = pd.DataFrame()
    if not _act.empty and "product_id" in _act.columns:
        _act_ym = _act.copy()
        if "activity_ym" not in _act_ym.columns and "activity_date" in _act_ym.columns:
            _act_ym["activity_ym"] = _act_ym["activity_date"].str[:7]
        if "activity_ym" in _act_ym.columns:
            _mr_monthly = (
                _act_ym.groupby(["product_id", "activity_ym"])
                .size()
                .reset_index(name="count")
                .groupby("product_id")["count"]
                .mean()
            )

    _dig_monthly = pd.Series(dtype=float)
    if not _digi.empty and "product_id" in _digi.columns:
        _digi_ym = _digi.copy()
        if "activity_ym" not in _digi_ym.columns and "activity_date" in _digi_ym.columns:
            _digi_ym["activity_ym"] = _digi_ym["activity_date"].str[:7]
        if "activity_ym" in _digi_ym.columns:
            _dig_monthly = (
                _digi_ym.groupby(["product_id", "activity_ym"])
                .size()
                .reset_index(name="count")
                .groupby("product_id")["count"]
                .mean()
            )

    # SOCデータを辞書化
    all_pids = set(_mr_monthly.index.tolist()) | set(_dig_monthly.index.tolist())
    soc_activity = {
        pid: {
            "mr":      float(_mr_monthly.get(pid, 0.0)),
            "digital": float(_dig_monthly.get(pid, 0.0)),
        }
        for pid in all_pids
    }
    # ---- SOC想起率（soc_params.csv）----
    soc_rates: Dict[str, Dict[str, float]] = {}
    soc_params_path = Path(__file__).parent / "soc_params.csv"
    if soc_params_path.exists():
        soc_params_df = pd.read_csv(soc_params_path)
        for _, row in soc_params_df.iterrows():
            soc_rates[str(row["product_id"]).strip()] = {
                "mr":      float(row["mr_soc_rate"]),
                "digital": float(row["digital_soc_rate"]),
            }
        print(f"       → soc_params.csv: {len(soc_rates)} 品目の想起率を読み込み")
    else:
        print("       → soc_params.csv なし → デフォルト想起率使用")

    print(f"       → SOCデータ集計: {len(soc_activity)} 品目")
    for pid in sorted(soc_activity.keys()):
        v = soc_activity[pid]
        r = soc_rates.get(pid, {})
        mr_eff  = v['mr']  * r.get('mr', 0.50)
        dig_eff = v['digital'] * r.get('digital', 0.25)
        total   = mr_eff + dig_eff
        soc_share = mr_eff / total if total > 0 else 0.5
        print(f"          {pid}: MR有効接触={mr_eff:.0f} / デジタル有効接触={dig_eff:.0f} → SOC MR比={soc_share:.2f}")

    digital_scorer = DigitalEffectivenessScorer(
        decay_params_df=data["decay_params_df"],
        soc_activity=soc_activity,
        soc_rates=soc_rates,
    )

    # ----------------------------------------------------------
    # OVEランチカーブ（発売後月次浸透率）
    # FY2026-07発売 → FY2029末で目標FTE ~40 を達成する設定
    # 参照品: TRI（同CS、類似規模）
    # ----------------------------------------------------------
    # OVEランチカーブ（発売時が最大 → 徐々に減少）
    # 新製品は発売直後が最も活動量が多い（医師教育・処方獲得に集中投資）
    # 普及が進むにつれてメンテナンスモードへ移行し頻度を低下させる
    # OVEランチカーブ（FY2026-07発売, 116ヶ月 = FY2035末まで）
    # 発売直後が最大投資、以降は段階的に逓減（立ち上げ→浸透→維持モード）
    ove_ramp_up = [
        # --- 立ち上げ期（FY2026〜FY2028: 1〜30ヶ月）---
        1.00, 0.98, 0.96, 0.94, 0.92, 0.90,   # 1〜6ヶ月  (FY2026発売)
        0.88, 0.86, 0.84, 0.83, 0.82, 0.81,   # 7〜12ヶ月
        0.80, 0.79, 0.78, 0.77, 0.76, 0.75,   # 13〜18ヶ月 (FY2027)
        0.74, 0.73, 0.72, 0.72, 0.71, 0.71,   # 19〜24ヶ月
        0.70, 0.70, 0.69, 0.69, 0.69, 0.68,   # 25〜30ヶ月 (FY2028)
        # --- 浸透期（FY2028〜FY2030: 31〜54ヶ月）---
        0.68, 0.68, 0.67, 0.67, 0.67, 0.67,   # 31〜36ヶ月
        0.66, 0.66, 0.66, 0.66, 0.65, 0.65,   # 37〜42ヶ月 (FY2029)
        0.64, 0.63, 0.63, 0.62, 0.61, 0.61,   # 43〜48ヶ月 (FY2030)
        0.60, 0.59, 0.59, 0.58, 0.57, 0.57,   # 49〜54ヶ月
        # --- 維持期（FY2031〜FY2035: 55〜116ヶ月）--- 緩やかに逓減
        0.56, 0.55, 0.55, 0.54, 0.53, 0.53,   # 55〜60ヶ月 (FY2031)
        0.52, 0.51, 0.51, 0.50, 0.49, 0.49,   # 61〜66ヶ月
        0.48, 0.47, 0.47, 0.46, 0.45, 0.45,   # 67〜72ヶ月 (FY2032)
        0.44, 0.43, 0.43, 0.42, 0.41, 0.41,   # 73〜78ヶ月
        0.40, 0.40, 0.39, 0.39, 0.38, 0.38,   # 79〜84ヶ月 (FY2033)
        0.37, 0.37, 0.36, 0.36, 0.35, 0.35,   # 85〜90ヶ月
        0.35, 0.35, 0.35, 0.35, 0.35, 0.35,   # 91〜96ヶ月 (FY2034)
        0.35, 0.35, 0.35, 0.35, 0.35, 0.35,   # 97〜102ヶ月
        0.35, 0.35, 0.35, 0.35, 0.35, 0.35,   # 103〜108ヶ月 (FY2035)
        0.35, 0.35, 0.35, 0.35, 0.35, 0.35,   # 109〜114ヶ月
        0.35, 0.35,                             # 115〜116ヶ月
    ]

    # Zasoランチカーブ（FY2027-04発売, 108ヶ月 = FY2035末まで、参照品: ENT）
    zaso_ramp_up = [
        # --- 立ち上げ期（FY2027〜FY2029: 1〜30ヶ月）---
        1.00, 0.98, 0.96, 0.94, 0.92, 0.90,   # 1〜6ヶ月  (FY2027発売)
        0.88, 0.86, 0.84, 0.83, 0.82, 0.81,   # 7〜12ヶ月
        0.80, 0.79, 0.78, 0.77, 0.76, 0.75,   # 13〜18ヶ月 (FY2028)
        0.74, 0.73, 0.72, 0.72, 0.71, 0.71,   # 19〜24ヶ月
        0.70, 0.70, 0.69, 0.69, 0.69, 0.68,   # 25〜30ヶ月 (FY2029)
        # --- 浸透期（FY2030〜FY2031: 31〜54ヶ月）---
        0.68, 0.68, 0.67, 0.67, 0.67, 0.67,   # 31〜36ヶ月
        0.66, 0.66, 0.66, 0.66, 0.65, 0.65,   # 37〜42ヶ月 (FY2030)
        0.64, 0.63, 0.62, 0.61, 0.60, 0.59,   # 43〜48ヶ月 (FY2031)
        0.58, 0.57, 0.56, 0.55, 0.54, 0.53,   # 49〜54ヶ月
        # --- 維持期（FY2032〜FY2035: 55〜108ヶ月）---
        0.52, 0.51, 0.50, 0.49, 0.48, 0.47,   # 55〜60ヶ月 (FY2032)
        0.46, 0.45, 0.44, 0.43, 0.42, 0.41,   # 61〜66ヶ月
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 67〜72ヶ月 (FY2033)
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 73〜78ヶ月
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 79〜84ヶ月 (FY2034)
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 85〜90ヶ月
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 91〜96ヶ月 (FY2035)
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 97〜102ヶ月
        0.40, 0.40, 0.40, 0.40, 0.40, 0.40,   # 103〜108ヶ月
    ]

    # WSAランチカーブ（FY2029-04発売, 84ヶ月 = FY2035末まで、参照品: LIV）
    wsa_ramp_up = [
        # --- 立ち上げ期（FY2029〜FY2031: 1〜30ヶ月）---
        1.00, 0.98, 0.96, 0.94, 0.92, 0.90,   # 1〜6ヶ月  (FY2029発売)
        0.88, 0.86, 0.84, 0.83, 0.82, 0.81,   # 7〜12ヶ月
        0.80, 0.79, 0.78, 0.77, 0.76, 0.75,   # 13〜18ヶ月 (FY2030)
        0.74, 0.73, 0.72, 0.72, 0.71, 0.71,   # 19〜24ヶ月
        0.70, 0.70, 0.69, 0.69, 0.69, 0.68,   # 25〜30ヶ月 (FY2031)
        # --- 浸透期（FY2032〜FY2033: 31〜54ヶ月）---
        0.68, 0.68, 0.67, 0.67, 0.67, 0.67,   # 31〜36ヶ月
        0.66, 0.66, 0.66, 0.66, 0.65, 0.65,   # 37〜42ヶ月 (FY2032)
        0.63, 0.61, 0.60, 0.58, 0.57, 0.55,   # 43〜48ヶ月 (FY2033)
        0.54, 0.52, 0.51, 0.50, 0.50, 0.50,   # 49〜54ヶ月
        # --- 維持期（FY2034〜FY2035: 55〜84ヶ月）---
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50,   # 55〜60ヶ月 (FY2034)
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50,   # 61〜66ヶ月
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50,   # 67〜72ヶ月 (FY2035)
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50,   # 73〜78ヶ月
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50,   # 79〜84ヶ月
    ]

    # GLOランチカーブ（FY2026-02発売, 13ヶ月 = FY2026末まで）
    # 既知の海外製品を代替品として導入するため、立ち上げ教育は不要 → フラット
    glo_ramp_up = [1.00] * 14  # 余裕をもって14ヶ月分（2026-02〜2027-03）

    calculator = FY2029FTECalculator(
        product_configs=product_configs,
        target_doctor_calc=target_doctor_calc,
        fc_sc_allocator=fc_sc_allocator,
        freq_estimator=freq_estimator,
        product_info=data["product_info"],
        frequency_mode="lifecycle_adjusted",
        new_product_ramp_up={
            "OVE": ove_ramp_up, "Zaso": zaso_ramp_up, "WSA": wsa_ramp_up,
            "GLO": glo_ramp_up,   # 供給制限代替品: FY2026のみ活動
        },
        reference_products=csv_reference_products,  # products.csvのreference_product列
        target_months=ALL_MONTHS,  # FY2026〜2035の120ヶ月
        competition_schedule=competition_schedule,  # competitor_schedule.csv から読み込み
        supply_restrictions=data["supply_restrictions"],  # 供給制限（GLI FY2026等）
        working_days_map      = data["working_days_map"],
    )

    # ---- 4. FTE 算出（FY2026〜FY2029）----
    print("[4/6] FY2026〜FY2035 FTE算出...")

    # 新品目: OVE（FY2026発売）、Zaso（FY2027発売）、WSA（FY2029発売）
    # FTE不足をドナー品目から補う
    # fy2026_apr_fte = FY2026/4時点の実績FTE（ドナー品目の削減上限として使用）
    new_product_fte_allocator = NewProductFTEAllocator(
        decay_params_df=data["decay_params_df"],
        current_fte=data["fy2026_apr_fte"],
        current_mr_activity=data["current_mr_activity"],
        min_fte_ratio=0.5,
    )

    # 動的FTE移動: 新品目（OVE/Zaso/WSA）の成長（ランチカーブ）に比例してドナー品目から段階的に削減
    fte_df, allocation_df = calculator.run_with_dynamic_new_product(
        new_product_ids=["OVE", "Zaso", "WSA"],
        donor_products=["GLI", "CUV", "HYQ", "INT", "TRI", "ENT",
                        "LIV", "REV", "ALC", "VYV", "VPR"],
        new_product_fte_allocator=new_product_fte_allocator,
    )

    # 正規化前の「本来必要FTE」を保存（制約なし、活動量から積み上げた真の必要値）
    raw_fte_df = fte_df.copy()
    raw_summary_fy = calculator.total_fte_by_area_fy(raw_fte_df)

    # ⑥ 新発売品ごとの発売時点FTE配分（どの品目から何FTE取るか）
    fte_col_raw = "adjusted_fte" if "adjusted_fte" in raw_fte_df.columns else "required_fte"
    new_product_launch_months = {"OVE": "2026-07", "Zaso": "2027-04", "WSA": "2029-04"}
    all_cs_donors = ["GLI", "CUV", "HYQ", "INT", "TRI", "ENT", "LIV", "REV", "ALC", "VYV", "VPR"]
    per_launch_allocations: Dict[str, pd.DataFrame] = {}
    for new_pid, launch_month in new_product_launch_months.items():
        month_df = raw_fte_df[raw_fte_df["month"] == launch_month]
        new_row = month_df[month_df["product_id"] == new_pid]
        if new_row.empty:
            continue
        launch_fte = float(new_row[fte_col_raw].iloc[0])
        # 発売時点で FTE > 0 のドナーのみ対象
        active_donors = [
            p for p in all_cs_donors
            if not month_df[month_df["product_id"] == p].empty
            and float(month_df[month_df["product_id"] == p][fte_col_raw].sum()) > 0
        ]
        if launch_fte > 0 and active_donors:
            per_launch_allocations[new_pid] = new_product_fte_allocator.allocate(
                launch_fte, active_donors
            )

    # 年度ヘッドカウント目標に正規化（CS=380, PS=45）
    # 品目間の相対比を保ちながら毎月の領域合計を目標値にスケーリング
    fte_df = normalize_fte_to_headcount(fte_df, CURRENT_MR_COUNT)

    # 月次FTEを半期（6ヶ月）ステップ関数に変換（離散的な計画値）
    fte_df = discretize_fte_semiannually(fte_df, CURRENT_MR_COUNT)

    # ROI最大化最適FTE算出（等限界収益配分）
    print("  ROI最適FTE算出中...")
    optimal_fte_df = calculate_roi_optimal_fte(
        decay_params_df=data["decay_params_df"],
        product_configs=product_configs,
        headcount_targets=CURRENT_MR_COUNT,
    )

    summary_df    = calculator.summarize_fy(fte_df)
    total_fte_df  = calculator.total_fte_by_area(fte_df)
    total_fte_fy  = calculator.total_fte_by_area_fy(fte_df)

    # ---- デジタル有効性スコア算出（FTE算出とは独立）----
    print("  デジタル有効性スコア算出中...")
    _score_rows = []
    for config in product_configs:
        pid = config.product_id
        # FY2029時点（発売後経過月数・LOE残月数を代表値として使用）
        _elapsed_repr = max(0, _months_between(config.launch_ym, "2029-10"))
        _loe_repr = max(0, config.loe_months - _elapsed_repr)
        s = digital_scorer.score(pid, _elapsed_repr, _loe_repr)
        _score_rows.append({
            "product_id":           pid,
            "digital_score":        s["score"],
            "digital_level":        s["level"],
            "mmm_digital_fraction": s["mmm_digital_fraction"],
            "digital_soc_rate":     s["digital_soc_rate"],
            "lifecycle_adj":        s["lifecycle_adj"],
        })
    digital_score_df = pd.DataFrame(_score_rows).sort_values("digital_score", ascending=False)
    print("  品目別デジタル有効性スコア（上位順）:")
    for _, row in digital_score_df.iterrows():
        print(f"    {row['product_id']:6s}: score={row['digital_score']:.3f} [{row['digital_level']}]"
              f"  MMM={row['mmm_digital_fraction']:.3f}  SOC={row['digital_soc_rate']:.3f}"
              f"  LC={row['lifecycle_adj']:+.2f}")

    # 結果サマリー表示
    print("\n  --- 品目×年度別 平均FTE ---")
    pivot = summary_df.pivot_table(
        index="product_id", columns="fiscal_year",
        values="avg_required_fte",
    ).round(1)
    print(pivot.to_string())

    print("\n  --- 新品目（OVE/Zaso/WSA）FTE配分先 ---")
    print(allocation_df.to_string(index=False))

    print("\n  --- 領域×年度別 合計FTE vs 現行MR数 ---")
    print(total_fte_fy.to_string(index=False))

    # ---- 5. 感度分析 ----
    print("\n[5/6] 感度分析...")
    analyzer = FY2029SensitivityAnalyzer(
        base_calculator=calculator,
        base_target_calc=target_doctor_calc,
        mmm_freq_estimator=None,
    )
    sensitivity_results = analyzer.run_all_scenarios(
        new_product_ids=["OVE", "Zaso", "WSA"],
        donor_products=["GLI", "CUV", "HYQ", "INT", "TRI", "ENT",
                        "LIV", "REV", "ALC", "VYV", "VPR"],
        new_product_fte_allocator=new_product_fte_allocator,
    )

    # ---- 6. HTML出力 ----
    print(f"\n[6/6] HTMLレポート出力 → {OUTPUT_DIR}/")

    # LOE / 新製品発売スケジュールを products.csv から構築
    _SIM_END = 2036  # シミュレーション終了年
    loe_schedule: Dict[str, str] = {}
    launch_schedule_mmm: Dict[str, str] = {}
    for cfg in product_configs:
        # LOE が 2099-01 = 実質なし → シミュレーション範囲外なら除外
        loe_dt_str = next(
            (row["loe_ym"] for _, row in pd.read_csv(PRODUCTS_CSV).iterrows()
             if row["product_id"] == cfg.product_id), None
        )
        if loe_dt_str and not loe_dt_str.startswith("2099"):
            loe_yr = int(loe_dt_str[:4])
            if loe_yr < _SIM_END:
                loe_schedule[cfg.product_id] = loe_dt_str
        if cfg.is_new:
            launch_schedule_mmm[cfg.product_id] = cfg.launch_ym

    reporter = FY2029HTMLReporter(output_dir=str(OUTPUT_DIR))
    reporter.generate(
        fte_df=fte_df,
        summary_df=summary_df,
        allocation_df=allocation_df,
        total_fte_df=total_fte_df,
        total_fte_fy_df=total_fte_fy,
        optimal_fte_df=optimal_fte_df,
        raw_summary_fy=raw_summary_fy,
        per_launch_allocations=per_launch_allocations,
        digital_act_df=data["digital_act"],
        soc_rates=soc_rates,
        digital_score_df=digital_score_df,
        decay_params_df=data["decay_params_df"],
        loe_schedule=loe_schedule,
        launch_schedule=launch_schedule_mmm,
        frequency_mode="lifecycle_adjusted",
        filename="fy2029_fte_report.html",
    )
    generate_logic_document(str(OUTPUT_DIR))

    analyzer.export_sensitivity_html(
        results=sensitivity_results,
        output_dir=str(OUTPUT_DIR),
        filename="fy2029_sensitivity.html",
    )

    # CSV出力
    fte_df.to_csv(OUTPUT_DIR / "fy2029_fte_detail.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "fy2029_summary.csv", index=False, encoding="utf-8-sig")
    allocation_df.to_csv(OUTPUT_DIR / "fy2029_allocation.csv", index=False, encoding="utf-8-sig")
    total_fte_fy.to_csv(OUTPUT_DIR / "fy2029_area_fy.csv", index=False, encoding="utf-8-sig")
    optimal_fte_df.to_csv(OUTPUT_DIR / "fy2029_optimal_fte.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] CSV出力完了")

    print("\n" + "=" * 65)
    print("FY2029 FTE算出 完了")
    print(f"レポート: {OUTPUT_DIR.resolve()}")
    print("=" * 65)

    return fte_df, summary_df, allocation_df, total_fte_df, total_fte_fy, sensitivity_results


if __name__ == "__main__":
    main()

# %%
