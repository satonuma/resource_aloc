"""
FTE シミュレーター ─ 社内共有ガイド
"""

import streamlit as st
import socket

st.set_page_config(
    page_title="社内共有ガイド | FTE シミュレーター",
    page_icon="🔗",
    layout="wide",
)

st.title("🔗 社内共有ガイド")
st.caption("このアプリを同僚と共有する方法を説明します。")

# ─────────────────────────────────────────────
# 現在の IP を表示
# ─────────────────────────────────────────────
try:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
except Exception:
    local_ip = "取得できませんでした"

st.info(f"このマシンの IP アドレス: **{local_ip}**（社内 LAN 用）", icon="💻")

# ─────────────────────────────────────────────
# 方法 1: 社内 LAN 共有（最速）
# ─────────────────────────────────────────────
st.header("方法 1 ── 社内 LAN で即時共有（最速・無料）")

st.markdown("""
同じオフィス LAN（Wi-Fi含む）に接続している同僚ならすぐ使えます。
""")

with st.expander("▸ 手順を見る", expanded=True):
    st.markdown(f"""
    **① このマシンで Streamlit を起動する**
    ```bash
    cd "<アプリのフォルダパス>"
    streamlit run app_fte.py --server.port 8501 --server.address 0.0.0.0
    ```

    **② 同僚のブラウザでアクセスしてもらう**
    ```
    http://{local_ip}:8501
    ```

    **③ アクセスできない場合**
    - Mac: システム設定 → ファイアウォール → `Python` または `streamlit` を許可
    - Windows: Windows Defender ファイアウォール → Python を許可
    """)

st.warning("""
**注意点**
- あなたの PC が起動している間だけ使えます
- VPN 接続では社内 LAN と切断されることがあります
- 機密データを扱う場合は IT 部門に確認してください
""", icon="⚠️")

# ─────────────────────────────────────────────
# 方法 2: OneDrive 共有 + 各自起動
# ─────────────────────────────────────────────
st.header("方法 2 ── OneDrive でフォルダ共有（各自 PC で動かす）")

st.markdown("""
全員が手元の PC でアプリを動かす形式です。データの同期が自動で行われます。
""")

with st.expander("▸ 手順を見る", expanded=False):
    st.markdown("""
    **① OneDrive でフォルダを共有**
    1. OneDrive でアプリフォルダを右クリック → 「共有」
    2. 同僚のメールアドレスを入力して招待

    **② 同僚側の初回セットアップ（1回だけ）**
    ```bash
    # Python 3.10+ が必要
    pip install -r requirements.txt
    ```

    **③ アプリ起動**
    ```bash
    cd "<OneDrive同期フォルダのパス>"
    streamlit run app_fte.py
    ```

    > `requirements.txt` には必要なパッケージが記載されています。
    """)

# ─────────────────────────────────────────────
# 方法 3: 社内サーバー（IT部門対応）
# ─────────────────────────────────────────────
st.header("方法 3 ── 社内サーバーへのデプロイ（常時稼働・推奨）")

st.markdown("IT 部門と連携して社内サーバーに常設すると、全員がいつでもアクセスできます。")

with st.expander("▸ Docker を使う場合（IT部門向け）", expanded=False):
    st.markdown("""
    **Dockerfile（IT 担当者に渡す）**
    ```dockerfile
    FROM python:3.11-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    EXPOSE 8501
    CMD ["streamlit", "run", "app_fte.py",
         "--server.port=8501",
         "--server.address=0.0.0.0",
         "--server.headless=true"]
    ```

    **起動コマンド**
    ```bash
    docker build -t fte-simulator .
    docker run -d -p 8501:8501 --name fte-app fte-simulator
    ```

    **アクセス URL**
    ```
    http://<社内サーバーIP>:8501
    ```
    """)

with st.expander("▸ Azure App Service を使う場合（Microsoft 環境向け）", expanded=False):
    st.markdown("""
    OneDrive が会社の Microsoft 365 環境にある場合、Azure との親和性が高いです。

    1. Azure Portal → 「App Service」を作成（Python 3.11, Linux）
    2. GitHub Actions または Azure CLI でデプロイ
    3. データファイルは Azure Blob Storage または SharePoint に配置

    詳細は IT 部門または Microsoft ドキュメントを参照してください。
    """)

# ─────────────────────────────────────────────
# 方法 4: Streamlit Community Cloud（外部）
# ─────────────────────────────────────────────
st.header("方法 4 ── Streamlit Community Cloud（インターネット経由）")

st.warning("機密データを含む場合はセキュリティポリシーを確認してから使用してください。", icon="🔒")

with st.expander("▸ 手順を見る", expanded=False):
    st.markdown("""
    **無料プランでプライベート公開が可能（GitHub アカウント必要）**

    1. [github.com](https://github.com) にリポジトリを作成（**Private** に設定）
    2. アプリファイルをプッシュ
       ```bash
       git init
       git add app_fte.py pages/ src/ data/ requirements.txt
       git commit -m "initial commit"
       git remote add origin https://github.com/<your-org>/<repo>.git
       git push -u origin main
       ```
    3. [share.streamlit.io](https://share.streamlit.io) にアクセス
    4. 「New app」→ GitHub リポジトリを選択 → Deploy
    5. 「Settings → Sharing」でメールアドレスを追加（招待制）

    **注意:** 無料プランはアプリが非活動時にスリープします。
    """)

# ─────────────────────────────────────────────
# requirements.txt の確認
# ─────────────────────────────────────────────
st.header("requirements.txt の確認")

from pathlib import Path
req_path = Path(__file__).parent.parent / "requirements.txt"
if req_path.exists():
    with open(req_path) as f:
        content = f.read()
    st.code(content, language="text")
    st.caption(f"パス: `{req_path}`")
else:
    st.warning("requirements.txt が見つかりません。アプリフォルダに配置してください。")
    st.markdown("""
    **最低限必要なパッケージ（例）:**
    ```
    streamlit>=1.32
    pandas>=2.0
    numpy>=1.26
    plotly>=5.18
    scipy>=1.12
    ```
    """)

st.divider()
st.caption("ご不明点はアプリ担当者までお問い合わせください。")
