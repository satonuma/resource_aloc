"""
FTE シミュレーター ─ ロジック説明ページ
"""

import streamlit as st

st.set_page_config(
    page_title="ロジック説明 | FTE シミュレーター",
    page_icon="📖",
    layout="wide",
)

st.title("📖 FTE シミュレーター ロジック説明")
st.caption("このページでは FTE 算出のアルゴリズムとパラメータを解説します。")

# ─────────────────────────────────────────────
# 0. 全体フロー
# ─────────────────────────────────────────────
st.header("0. 全体処理フロー")

st.markdown("""
```
[入力 CSV]
  ├─ products.csv          品目マスタ（発売日・LOE・ブーストパラメータ等）
  ├─ target_doctors.csv    ターゲット医師数（R/W ティア、訪問頻度デフォルト）
  ├─ target_doctor_yearly  年度別ターゲット医師数（デフォルト上書き）
  ├─ visit_freq.csv        訪問頻度・達成率（品目×年度×R/W）
  ├─ fc_sc_ratio.csv       FC/SC 比率（品目×年度）
  ├─ competitor_yearly.csv 年度別直接競合品数
  └─ competitor_schedule   競合品発売スケジュール（発売ブースト用）
          ↓
[Step 1]  月別 required_fte を算出（Pass1: base_fte → Pass2: required_fte）
          ↓
[Step 2]  新製品の成長に合わせてドナー品目の FTE を段階削減
          （run_with_dynamic_new_product）
          ↓
[Step 3]  ROI 最適配分で BU2 合計ヘッドカウントに正規化
          （normalize_fte_roi_weighted, combined_mode=True）
          ↓
[Step 4]  半期ステップ関数に離散化 + 再正規化
          （discretize_fte_semiannually）
          ↓
[Step 5]  品目別 max_fte キャップを適用
          ↓
[出力]
  ├─ キャップなし FTE サマリ（品目×年度）
  ├─ キャップあり FTE サマリ（品目×年度）
  └─ 月次詳細 DataFrame
```
""")

# ─────────────────────────────────────────────
# 1. FTE 算出式
# ─────────────────────────────────────────────
st.header("1. FTE 算出式（Step 1）")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("基本式")
    st.latex(r"""
    \text{base\_fte} = \frac{\text{required\_calls}}{\text{calls\_per\_day} \times \text{working\_days}}
    """)
    st.latex(r"""
    \text{required\_calls} = (N_R \cdot f_R \cdot a_R \;+\; N_W \cdot f_W \cdot a_W)
    \;\times\; w_{FC}
    """)

    st.markdown("""
    | 変数 | 意味 |
    |------|------|
    | $N_R, N_W$ | R/W ティアのターゲット医師数 |
    | $f_R, f_W$ | 目標訪問頻度（回/月） |
    | $a_R, a_W$ | 達成率（0〜1） |
    | $w_{FC}$ | FC 重み係数（後述） |
    """)

with col2:
    st.subheader("エリア別コール数・稼働日数")

    st.markdown("""
    | エリア | calls/day | calls/month(20日) |
    |--------|-----------|-------------------|
    | CS     | 2.5       | 50                |
    | PS遺伝 | 1.5       | 30                |
    | PS血液 | 1.5       | 30                |

    > 稼働日数は `working_days.csv` で月別に調整可能。
    """)

    st.subheader("FC 重み係数")
    st.latex(r"""
    w_{FC} = r_{FC} + (1 - r_{FC}) \times 0.1
    """)
    st.markdown("""
    - $r_{FC}$ : FC（ファーストコール）比率（0〜1）
    - SC 訪問は FC 訪問に内包されるため **×0.1** のコストで計上
    - 例: $r_{FC}=0.3$ → $w_{FC} = 0.3 + 0.7 \\times 0.1 = 0.37$
    """)

# ─────────────────────────────────────────────
# 2. ブースト係数
# ─────────────────────────────────────────────
st.header("2. ブースト係数")

tab_ind, tab_comp, tab_supply = st.tabs(["効能追加ブースト", "競合品ブースト", "供給制限"])

with tab_ind:
    st.subheader("効能追加ブースト（indication_boost）")
    st.markdown("効能追加時に一時的にFTEを増やし、その後線形に減衰して元に戻ります（最大2効能まで積算）。")
    st.latex(r"""
    b_i = 1 + (B_i - 1)\left(1 - \frac{t_i}{T_i}\right), \quad 0 \le t_i < T_i
    """)
    st.markdown("""
    | パラメータ | 説明 |
    |------------|------|
    | $B_i$ | ブースト倍率（例: 1.3 = +30%） |
    | $T_i$ | ブースト持続月数 |
    | $t_i$ | 効能追加からの経過月数 |

    **最終ブースト = $b_1 \\times b_2$**（第1・第2効能の積）

    設定場所: `products.csv` の `indication_add_ym`, `indication_fte_boost`, `indication_boost_months`
    """)

with tab_comp:
    st.subheader("競合品ブースト（competition_boost）")
    st.markdown("競合品の発売や市場競合に応じて FTE を増やします（2要素の積）。")

    st.markdown("#### ① 発売タイミングブースト（launch_boost）")
    st.latex(r"""
    \text{launch\_boost} = \max_k \left[ 1 + (I_k - 1)\left(1 - \frac{e_k}{M_k}\right) \right]
    """)
    st.markdown("""
    - $I_k$: 競合品 k の intensity（発売直後の倍率）
    - $e_k$: 発売から経過月数
    - $M_k$: boost_months（持続期間）
    - 複数競合が重なる場合は**最大値**を採用

    設定場所: `competitor_schedule.csv`（競合品スケジュール タブ）
    """)

    st.markdown("#### ② 直接競合数ブースト（count_boost）")
    st.latex(r"""
    \text{count\_boost} = 1 + n_{\text{direct}} \times f_{\text{count}}
    """)
    st.markdown("""
    - $n_{\\text{direct}}$: その年度の同効能直接競合品数
    - $f_{\\text{count}}$: count_factor（サイドバーで設定、デフォルト 0.05）
    - 例: 競合 2 品 × 0.05 = +10%

    設定場所: `competitor_yearly.csv`（直接競合数 タブ）

    **最終競合ブースト = launch_boost × count_boost**
    """)

with tab_supply:
    st.subheader("供給制限ファクター（supply_restriction）")
    st.markdown("供給不足期間中はFTEを削減します。")
    st.latex(r"""
    \text{base\_fte} \;\times= f_{\text{supply}}
    """)
    st.markdown("""
    - $f_{\\text{supply}}$: 制限係数（例: 0.65 = 35%削減）
    - 設定場所: `supply_restriction.csv`
    """)

st.markdown("""
**3 要素の適用順序:**
```
base_fte × indication_boost × competition_boost × supply_factor = required_fte
```
""")

# ─────────────────────────────────────────────
# 3. ライフサイクル調整
# ─────────────────────────────────────────────
st.header("3. ライフサイクル調整")

col_lc1, col_lc2 = st.columns(2)

with col_lc1:
    st.subheader("発売前後のランプアップ")
    st.markdown("""
    新製品は発売直後から徐々に医師へのアクセスが増えます。
    各月の浸透率（ランプ率）を設定し、ターゲット医師数に掛けて実効数を決定します。

    ```
    effective_doctors = target_doctors × ramp_rate(month)
    ```

    設定: `products.csv` → `is_new=True` の品目に適用
    ランプ曲線: 参照品目（`reference_product`）の普及パターンを参照
    """)

with col_lc2:
    st.subheader("LOE（特許切れ）後の減少")
    st.markdown("""
    LOE 後は後発品参入により活動価値が低下します。

    ```
    fte × post_loe_factor  （LOE月の翌月から適用）
    ```

    | post_loe_factor | 意味 |
    |-----------------|------|
    | 0.0 | LOE と同時に撤退（FTE=0） |
    | 0.55 | LOE 後も 55% の FTE を継続 |
    | 0.9 | 実質ほぼ維持（希少疾患等） |

    設定: `products.csv` → `post_loe_factor` 列
    """)

# ─────────────────────────────────────────────
# 4. 新製品 FTE のドナー配分
# ─────────────────────────────────────────────
st.header("4. 新製品 FTE のドナー配分（Step 2）")

st.markdown("""
新製品（OVE・Zaso・TAK881）の成長に比例して、既存のドナー品目から FTE を段階的に削減します。

```
ドナー削減量(月) = ピーク削減量 × (当月新製品FTE合計 / ピーク月新製品FTE)
adjusted_fte = required_fte − fte_reduction    （下限 0）
```

### ドナー品目の割り当てロジック

$$\\text{削減量}_d = \\text{peak\\_fte} \\times \\frac{w_d}{\\sum_d w_d}$$

$$w_d = \\frac{1}{\\text{marginal ROI}_d} \\times \\text{slack}_d, \\quad \\text{slack}_d = \\text{FTE}_d - 0.5 \\times \\text{FTE}_d$$

- **限界ROI が低い品目**（成熟・飽和）ほど多く削減される
- **slack** = 現在FTEの 50% 以上は守られる（min_fte_ratio = 0.5）

現在のドナー品目: **GLI, CUV, HYQ, INT, TRI**
（ENT は target_doctor_yearly で手動管理のため除外）
""")

# ─────────────────────────────────────────────
# 5. 正規化ロジック
# ─────────────────────────────────────────────
st.header("5. 正規化ロジック（Step 3・4）")

st.markdown("""
### BU2 合計プール正規化（combined_mode）

**方針:** CS / PS遺伝 / PS血液 のエリア間でFTEを自由に移動させながら、
BU2 全体の合計ヘッドカウントを固定します。

$$\\text{scale} = \\frac{\\text{BU2合計目標}}{\\sum_i \\text{adjusted\\_fte}_i}$$

$$\\text{normalized\\_fte}_i = \\text{adjusted\\_fte}_i \\times \\text{scale}$$

- 需要 > 目標: スケールダウン（削減）→ ROI 最適化で各品目の配分を決定
- 需要 < 目標: スケールアップ（全員フル稼働）→ 比例スケール

### ROI 最適配分（MMM パラメータがある品目）

Hill 関数型 Response curve に基づく限界ROI均等化：

$$\\text{Marginal ROI}_i(\\lambda) = \\beta_i \\cdot \\text{Hill}'(x_i; EC_{m,i}, s_i)$$

制約: $\\sum_i \\text{FTE}_i = \\text{BU2合計目標}$
最適解: 全品目の限界ROI が等しくなる $\\lambda$ を二分探索で求める。

### 半期離散化（Step 4）

月次FTEを半期（H1: 4〜9月、H2: 10〜3月）の平均値に平滑化し、
人員計画の実運用に合わせた「ステップ関数」型のFTE推移に変換します。
""")

# ─────────────────────────────────────────────
# 6. パラメータ一覧
# ─────────────────────────────────────────────
st.header("6. 主要パラメータ一覧")

st.markdown("""
### サイドバーパラメータ

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| BU2 合計 MR 数 | 445 | CS＋PS遺伝＋PS血液 の合計ヘッドカウント上限 |
| 競合品数 count_factor | 0.05 | 直接競合 1 品あたりの FTE 増加率（5%/品目） |

### CSV パラメータ（品目別）

| ファイル | 列 | 説明 |
|----------|----|------|
| products.csv | `launch_ym` | 発売年月（YYYY-MM） |
| products.csv | `loe_ym` | LOE 年月（YYYY-MM） |
| products.csv | `post_loe_factor` | LOE 後 FTE 残存率 |
| products.csv | `max_fte` | 品目別 FTE 上限（空欄=上限なし） |
| products.csv | `indication_add_ym` | 効能追加年月（第1効能） |
| products.csv | `indication_fte_boost` | 効能追加時のブースト倍率 |
| products.csv | `indication_boost_months` | ブースト持続月数 |
| target_doctors.csv | `r_doctors` | R（重点医師）数 |
| target_doctors.csv | `w_doctors` | W（一般医師）数 |
| visit_freq.csv | `target_freq` | 目標訪問頻度（回/月） |
| visit_freq.csv | `achievement_rate` | 訪問達成率（0〜1） |
| fc_sc_ratio.csv | `fc_ratio` | FC 比率（0=全SC、1=全FC） |
| competitor_yearly.csv | `n_direct_competitors` | 同効能直接競合品数（年度別） |

### エリア区分

| エリア | 対象品目 | 特徴 |
|--------|----------|------|
| CS | GLI, GLO, CUV, HYQ, TAK881, INT, TRI, ENT, OVE, Zaso, REV, ALC, VYV, WSA, SYN | 一般専門領域（calls/day: 2.5） |
| PS遺伝 | LVM, TKZ, VPR, RPL, FIR | 希少疾患・遺伝（calls/day: 1.5） |
| PS血液 | VON, LIV, FEI, OBI, ADV, ADY, Meza, KLR | 希少疾患・血液（calls/day: 1.5） |
""")

# ─────────────────────────────────────────────
# 7. 用語集
# ─────────────────────────────────────────────
with st.expander("📚 用語集", expanded=False):
    st.markdown("""
    | 用語 | 説明 |
    |------|------|
    | **FTE** | Full-Time Equivalent。1名のMRが1ヶ月フル稼働した場合を1.0とした工数単位 |
    | **R医師（Regulars）** | 定期的に詳細説明を行う重点医師（高頻度訪問、高達成率） |
    | **W医師（Wides）** | より広いターゲット層の医師（低頻度訪問） |
    | **FC（First Call）** | 主訪問。フルコストでFTEを計上 |
    | **SC（Second Call）** | FC訪問に内包される付随訪問。コスト係数0.1で計上 |
    | **LOE** | Loss of Exclusivity（特許切れ）。後発品参入が始まるタイミング |
    | **BU2** | ビジネスユニット2。CS＋PS遺伝＋PS血液の総称 |
    | **donor品目** | 新製品の成長に合わせてFTEを提供する既存品目 |
    | **combined_mode** | 全エリアを1つのプールとして正規化するモード（エリア間移動あり） |
    | **Hill関数** | S字型の用量反応曲線。限界ROI算出に使用 |
    | **MMM** | Marketing Mix Modeling。ROI最適化の根拠となるパラメータを提供 |
    """)
