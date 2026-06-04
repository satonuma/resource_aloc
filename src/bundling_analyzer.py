"""
抱き合わせ活動分析モジュール
==============================
訪問記録IDでグループ化し、品目ごとの
「他品目PCの訪問に乗っかっている率（供給率）」を計算する。

供給率 (supply_ratio) = 品目XのSC/SPC回数（非PCとして） ÷ 品目Xの総活動回数

  supply_ratio が高い
    → その品目の訪問の多くが他品目PCに乗っかって発生している
    → B 自身が独自にスケジュールすべき訪問は少ない

FTE 補正式 (呼び元の fy2029_fte_calculator.py で適用):
  effective_raw_calls = required_calls × (1 − supply_ratio)
  required_fte        = effective_raw_calls × fc_weight / capacity

想定データ形式:
  列名はデフォルト DEFAULT_COL_MAP の通り (後述)。
  同一 訪問記録ID を持つ行が「1回の訪問」を構成する。
  例:
    訪問記録ID | 品目コード | 活動大別
    V001       | ENT        | PC       ← その訪問のメイン品目
    V001       | REV        | SPC      ← ついでに添え物で説明
    V001       | (空)       | 面談     ← 品目なし活動（除外される）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


# ─────────────────────────────────────────────────────────────
# デフォルト列名マッピング  ← ユーザーの実データ列名に合わせる
# ─────────────────────────────────────────────────────────────
DEFAULT_COL_MAP: Dict[str, str] = {
    "visit_id":       "訪問記録ID",    # 同一訪問を紐付けるキー
    "product_code":   "品目コード",    # product_id と対応させる値
    "activity_class": "活動大別",      # PC / SC / SPC / Other の値が入る列
    "date":           "日付",          # 半期別集計に使用
}

# 各活動大別のデフォルト値セット（複数表記がある場合は追加）
DEFAULT_PC_VALUES:  set = {"PC"}
DEFAULT_SC_VALUES:  set = {"SC"}
DEFAULT_SPC_VALUES: set = {"SPC"}


# ─────────────────────────────────────────────────────────────
# メイン集計関数
# ─────────────────────────────────────────────────────────────

def compute_bundling_supply(
    activity_df : pd.DataFrame,
    col_map     : Optional[Dict[str, str]] = None,
    pc_values   : Optional[set] = None,
    sc_values   : Optional[set] = None,
    spc_values  : Optional[set] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    訪問記録から品目ごとの抱き合わせ供給率を計算する。

    Parameters
    ----------
    activity_df : 訪問記録 DataFrame (1行=1品目×1活動)
    col_map     : 列名マッピング。未指定は DEFAULT_COL_MAP を使用。
                  例: {"visit_id": "call_id", "product_code": "prod_cd"}
    pc_values   : PC と見なす 活動大別 の値セット
    sc_values   : SC と見なす 活動大別 の値セット
    spc_values  : SPC と見なす 活動大別 の値セット

    Returns
    -------
    (supply_dict, detail_df)
      supply_dict : {product_code: supply_ratio}
                    FTE 計算で (1 − supply_ratio) を raw_calls に乗算する
      detail_df   : 品目別の集計詳細 DataFrame
                    列: 品目コード, PC回数, SC回数, SPC回数,
                        供給回数, 総活動回数, 供給率

    Notes
    -----
    同一 visit_id 内で PC と SC/SPC 両方に現れた品目は PC 扱いとし
    供給回数にカウントしない。（その訪問では自分がメインのため）
    """
    cm = {**DEFAULT_COL_MAP, **(col_map or {})}

    _pc  = pc_values  or DEFAULT_PC_VALUES
    _sc  = sc_values  or DEFAULT_SC_VALUES
    _spc = spc_values or DEFAULT_SPC_VALUES

    visit_col  = cm["visit_id"]
    prod_col   = cm["product_code"]
    act_col    = cm["activity_class"]

    # ── 必要列チェック ──────────────────────────────────────────
    missing = [c for c in [visit_col, prod_col, act_col] if c not in activity_df.columns]
    if missing:
        raise ValueError(
            f"列が見つかりません: {missing}\n"
            f"実際の列名: {list(activity_df.columns)}\n"
            "col_map で列名を指定してください。"
        )

    df = activity_df[[visit_col, prod_col, act_col]].copy()
    df[prod_col] = df[prod_col].astype(str).str.strip()
    df[act_col]  = df[act_col].astype(str).str.strip()

    # 品目コードが空 / NaN の行（面談など品目なし活動）を除外
    df = df[df[prod_col].notna() & ~df[prod_col].isin(["", "nan", "None"])]
    if df.empty:
        return {}, pd.DataFrame()

    # ── 活動フラグ付与 ──────────────────────────────────────────
    df["_is_pc"]  = df[act_col].isin(_pc)
    df["_is_sc"]  = df[act_col].isin(_sc)
    df["_is_spc"] = df[act_col].isin(_spc)

    # 同一訪問内でPC品目のセットを取得
    pc_sets = (
        df[df["_is_pc"]]
        .groupby(visit_col)[prod_col]
        .apply(frozenset)
        .rename("_pc_set")
    )
    df = df.merge(pc_sets, on=visit_col, how="left")
    df["_pc_set"] = df["_pc_set"].apply(
        lambda x: x if isinstance(x, frozenset) else frozenset()
    )

    # 「供給」= SC/SPC として出現 かつ 同訪問で自分がPCではない
    df["_is_supplied"] = (
        (df["_is_sc"] | df["_is_spc"])
        & df.apply(lambda r: r[prod_col] not in r["_pc_set"], axis=1)
    )

    # ── 品目別集計 ──────────────────────────────────────────────
    agg = df.groupby(prod_col, as_index=False).agg(
        PC回数    = ("_is_pc",       "sum"),
        SC回数    = ("_is_sc",       "sum"),
        SPC回数   = ("_is_spc",      "sum"),
        供給回数  = ("_is_supplied", "sum"),
    )
    agg["総活動回数"] = agg["PC回数"] + agg["SC回数"] + agg["SPC回数"]
    agg["供給率"]     = (
        (agg["供給回数"] / agg["総活動回数"]).fillna(0.0).clip(0.0, 1.0).round(4)
    )

    supply_dict: Dict[str, float] = dict(zip(agg[prod_col], agg["供給率"]))

    detail_df = agg.rename(columns={prod_col: "品目コード"})

    return supply_dict, detail_df


# ─────────────────────────────────────────────────────────────
# 半期別集計（時系列トレンド用）
# ─────────────────────────────────────────────────────────────

def compute_bundling_by_half(
    activity_df : pd.DataFrame,
    col_map     : Optional[Dict[str, str]] = None,
    pc_values   : Optional[set] = None,
    sc_values   : Optional[set] = None,
    spc_values  : Optional[set] = None,
) -> pd.DataFrame:
    """
    半期（FY2024-H1, FY2024-H2 …）ごとに compute_bundling_supply を実行し、
    供給率の時系列トレンドを返す。

    Parameters
    ----------
    activity_df : 日付列（デフォルト: "日付"）を含む訪問記録 DataFrame

    Returns
    -------
    DataFrame: 品目コード, half_period, PC回数, SC回数, SPC回数, 供給率
    """
    # 遅延インポート（循環を避ける）
    import sys, os
    _src = os.path.dirname(__file__)
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from fy2029_fte_calculator import month_to_half  # type: ignore

    cm       = {**DEFAULT_COL_MAP, **(col_map or {})}
    date_col = cm["date"]

    if date_col not in activity_df.columns:
        raise ValueError(
            f"日付列 '{date_col}' が見つかりません。\n"
            f"実際の列名: {list(activity_df.columns)}"
        )

    df = activity_df.copy()
    try:
        df["_ym"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m")
    except Exception:
        df["_ym"] = df[date_col].astype(str).str[:7]

    df["_half"] = df["_ym"].apply(
        lambda x: month_to_half(x) if (isinstance(x, str) and len(x) == 7) else None
    )
    df = df.dropna(subset=["_half"])

    rows: List[pd.DataFrame] = []
    for half, grp in df.groupby("_half"):
        try:
            _, detail = compute_bundling_supply(
                grp.drop(columns=["_ym", "_half"], errors="ignore"),
                col_map, pc_values, sc_values, spc_values,
            )
            if not detail.empty:
                detail["half_period"] = half
                rows.append(detail)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# 抱き合わせペア分析（どの品目の組み合わせが多いか）
# ─────────────────────────────────────────────────────────────

def compute_bundling_pairs(
    activity_df : pd.DataFrame,
    col_map     : Optional[Dict[str, str]] = None,
    pc_values   : Optional[set] = None,
    sc_values   : Optional[set] = None,
    spc_values  : Optional[set] = None,
    top_n       : int = 20,
) -> pd.DataFrame:
    """
    「PC品目 → SC/SPC品目」のペア別出現回数を集計する。

    Returns
    -------
    DataFrame: primary_product, secondary_product, 共同訪問回数, 供給回数, 比率
      primary_product   : PCとして出現した品目
      secondary_product : SC/SPCとして同時に出現した品目（≠ primary）
      共同訪問回数      : 両品目が同一訪問に出現した回数
      比率              : secondary_product の全活動に占める割合
    """
    cm = {**DEFAULT_COL_MAP, **(col_map or {})}
    _pc  = pc_values  or DEFAULT_PC_VALUES
    _sc  = sc_values  or DEFAULT_SC_VALUES
    _spc = spc_values or DEFAULT_SPC_VALUES

    visit_col = cm["visit_id"]
    prod_col  = cm["product_code"]
    act_col   = cm["activity_class"]

    missing = [c for c in [visit_col, prod_col, act_col] if c not in activity_df.columns]
    if missing:
        raise ValueError(f"列が見つかりません: {missing}")

    df = activity_df[[visit_col, prod_col, act_col]].copy()
    df[prod_col] = df[prod_col].astype(str).str.strip()
    df[act_col]  = df[act_col].astype(str).str.strip()
    df = df[df[prod_col].notna() & ~df[prod_col].isin(["", "nan", "None"])]

    # 訪問ごとの PC 品目セットと SC/SPC 品目セット
    pc_df  = df[df[act_col].isin(_pc)].groupby(visit_col)[prod_col].apply(list).rename("pc_prods")
    sec_df = df[df[act_col].isin(_sc | _spc)].groupby(visit_col)[prod_col].apply(list).rename("sec_prods")

    visits = pd.concat([pc_df, sec_df], axis=1).dropna(how="all")
    visits["pc_prods"]  = visits["pc_prods"].apply(lambda x: x if isinstance(x, list) else [])
    visits["sec_prods"] = visits["sec_prods"].apply(lambda x: x if isinstance(x, list) else [])

    pairs: Dict[Tuple[str, str], int] = {}
    for _, row in visits.iterrows():
        pcs  = set(row["pc_prods"])
        secs = set(row["sec_prods"]) - pcs  # PCでもある品目は除外
        for pc in pcs:
            for sec in secs:
                pairs[(pc, sec)] = pairs.get((pc, sec), 0) + 1

    if not pairs:
        return pd.DataFrame()

    pair_df = pd.DataFrame(
        [(pc, sec, cnt) for (pc, sec), cnt in pairs.items()],
        columns=["primary_product", "secondary_product", "共同訪問回数"],
    ).sort_values("共同訪問回数", ascending=False).head(top_n).reset_index(drop=True)

    # secondary の総活動回数を付与して比率計算
    total_counts = df.groupby(prod_col).size().rename("_total")
    pair_df = pair_df.merge(
        total_counts.rename_axis("secondary_product").reset_index(),
        on="secondary_product", how="left",
    )
    pair_df["secondary に占める割合"] = (
        (pair_df["共同訪問回数"] / pair_df["_total"]).round(3)
    )
    pair_df = pair_df.drop(columns=["_total"])

    return pair_df
