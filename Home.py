import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from google import genai
from datetime import datetime, timedelta
import sys
import os

# 確保 utils 路徑可以被找到
sys.path.insert(0, os.path.dirname(__file__))
from utils.google_drive import (
    get_google_auth_url,
    handle_oauth_callback,
    load_from_drive,
    save_to_drive,
    is_logged_in,
    logout,
)

# --- 設定頁面配置 ---
st.set_page_config(page_title="AI 戰情雷達 (雲端永久版)", layout="wide")

st.title("🚀 AI 戰情雷達 - 雲端永久版")

# ===========================================================
# Google OAuth 回呼處理 (必須在所有 UI 之前執行)
# ===========================================================
params = st.query_params
if "code" in params and not is_logged_in():
    with st.spinner("🔐 正在完成 Google 登入..."):
        state_val = params.get("state", "")
        success = handle_oauth_callback(params["code"], state_val)
    if success:
        # 清除網址中的 code 參數，避免重複處理
        st.query_params.clear()
        st.success("✅ Google 登入成功！")
        st.rerun()
    else:
        err = st.session_state.get("google_auth_error", "未知錯誤")
        st.error(f"❌ 登入失敗：{err}")
        st.query_params.clear()


# ===========================================================
# 觀察清單管理（整合 Drive + 網址備用）
# ===========================================================
DEFAULT_TICKERS = "2330, 2317, 3034, 2376, 2383, 2027, 0050"

def get_tickers():
    """依優先順序讀取清單：Drive > session_state > 網址 > 預設"""
    # 若已登入且尚未從 Drive 載入，則嘗試從 Drive 讀取
    if is_logged_in() and "drive_tickers_loaded" not in st.session_state:
        drive_tickers = load_from_drive()
        if drive_tickers:
            st.session_state["tickers"] = drive_tickers
        st.session_state["drive_tickers_loaded"] = True

    if "tickers" in st.session_state:
        return st.session_state["tickers"]

    # 備用：從網址讀取
    if "tickers" in params:
        st.session_state["tickers"] = params["tickers"]
        return params["tickers"]

    return DEFAULT_TICKERS

def update_tickers(new_tickers: str):
    """更新清單：存入 session_state，若已登入則同步到 Drive"""
    st.session_state["tickers"] = new_tickers
    if is_logged_in():
        save_to_drive(new_tickers)
    else:
        # 未登入時維持網址書籤功能
        st.query_params["tickers"] = new_tickers


# ===========================================================
# 側邊欄：Google 登入區塊
# ===========================================================
st.sidebar.header("⚙️ 核心設定")

# --- Google 帳號登入 ---
st.sidebar.subheader("☁️ Google Drive 同步")

if is_logged_in():
    email = st.session_state.get("google_user_email", "使用者")
    st.sidebar.success(f"✅ 已登入\n{email}")
    if st.sidebar.button("🔄 立即從 Drive 同步"):
        st.session_state.pop("drive_tickers_loaded", None)
        st.rerun()
    if st.sidebar.button("🚪 登出 Google"):
        logout()
        st.session_state.pop("drive_tickers_loaded", None)
        st.rerun()
else:
    st.sidebar.info("登入 Google 後，觀察清單將自動儲存於您的 Google Drive，換裝置也不怕遺失！")
    auth_url, err = get_google_auth_url()
    if err:
        st.sidebar.error(f"⚠️ 設定錯誤：{err}")
    else:
        st.sidebar.link_button("🔑 登入 Google 帳號", auth_url, use_container_width=True)
    st.sidebar.caption("未登入時，可用下方網址書籤儲存清單。")

st.sidebar.divider()

# --- AI 模型選擇 ---
st.sidebar.subheader("🧠 AI 模型引擎")
model_map = {
    "🔥 最新深度旗艦版 (Gemini 3.1 Pro)": "gemini-3.1-pro-preview",
    "🚀 最新極速版 (Gemini 3.0 Flash)": "gemini-3-flash-preview",
    "⚡ 穩定極速版 (Gemini 2.5 Flash)": "gemini-2.5-flash",
    "🧠 最新深度版 (Gemini 2.5 Pro)": "gemini-2.5-pro",
    "⚡ 穩定極速版 (Gemini 2.0 Flash)": "gemini-2.0-flash",
    "🎁 驚喜自動升級版 (gemini-flash-latest)": "gemini-flash-latest",
}
selected_label = st.sidebar.selectbox("選擇分析大腦", list(model_map.keys()), index=0)
model_name = model_map[selected_label]
st.session_state['selected_gemini_model'] = model_name

# --- 觀察清單輸入 ---
st.sidebar.subheader("📋 觀察清單")
sync_label = "（已連結 Google Drive ☁️）" if is_logged_in() else "（網址書籤模式）"
st.sidebar.caption(sync_label)

current_tickers = get_tickers()

user_input = st.sidebar.text_area(
    "輸入代號 (逗號分隔)",
    value=current_tickers,
    height=150,
    help="登入 Google 後修改清單會自動儲存到雲端；未登入時請將更新後的網址存為書籤。"
)

if user_input != current_tickers:
    update_tickers(user_input)
    if is_logged_in():
        st.sidebar.toast("☁️ 已同步到 Google Drive！", icon="✅")
    st.rerun()


# ===========================================================
# 爬蟲抓中文名
# ===========================================================
@st.cache_data(ttl=86400)
def get_stock_name_from_web(code):
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string
            if title: return title.split('(')[0].strip()
    except: pass
    return f"{code}"


# ===========================================================
# 技術指標計算核心
# ===========================================================
def calculate_technical_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    k_values = [50]; d_values = [50]
    for i in range(1, len(df)):
        rsv = df['RSV'].iloc[i]
        if pd.isna(rsv):
            k_values.append(k_values[-1]); d_values.append(d_values[-1])
        else:
            k = (1/3) * rsv + (2/3) * k_values[-1]
            d = (1/3) * k + (2/3) * d_values[-1]
            k_values.append(k); d_values.append(d)
    df['K'] = k_values; df['D'] = d_values
    return df


# ===========================================================
# Gemini AI 分析
# ===========================================================
def get_gemini_analysis(df, model_id):
    api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("請輸入"):
        return "❌ 錯誤：找不到 API Key"

    data_text = df.to_string(index=False)
    prompt = f"""
    現在是 2026 年，請擔任王牌操盤手。根據以下數據（含 RSI, MACD, KD, 量能）進行分析。
    【數據清單】：\n{data_text}\n
    【分析要求】：
    1. 🏆 冠軍像：點名「三線合一」(RSI強, MACD紅, KD金叉)最強股。
    2. ⚠️ 未爆彈：找出指標背離股。
    3. 🎯 操作建議：強力買進 / 拉回買進 / 觀望 / 賣出。
    使用繁體中文，專業犀利。
    """
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"AI 錯誤: {e}"


# ===========================================================
# 抓取數據主程式
# ===========================================================
def get_stock_data(tickers):
    data_list = []
    clean_tickers = tickers.replace("，", ",").split(',')
    ticker_list = [t.strip() for t in clean_tickers if t.strip()]

    my_bar = st.progress(0, text="連線 Yahoo 股市...")

    for i, code in enumerate(ticker_list):
        name = get_stock_name_from_web(code)
        symbol = f"{code}.TW"

        try:
            end_date = datetime.now() + timedelta(days=1)
            start_date = end_date - timedelta(days=180)

            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 20:
                symbol = f"{code}.TWO"
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
        except Exception as e:
            st.warning(f"⚠️ 獲取 {code} 數據時遭遇連線限制，將暫時跳過。")
            continue

        if not df.empty and len(df) > 30:
            df = calculate_technical_indicators(df)
            last = df.iloc[-1]; prev = df.iloc[-2]
            price = last['Close']
            change_pct = ((price - prev['Close']) / prev['Close']) * 100
            vol_ratio = last['Volume'] / df['Volume'].rolling(5).mean().iloc[-1] if df['Volume'].rolling(5).mean().iloc[-1] > 0 else 0
            macd_signal = "🟢 偏多" if last['MACD_Hist'] > 0 else "🔴 偏空"
            kd_signal = "✨ 金叉" if last['K'] > last['D'] and prev['K'] < prev['D'] else ("💀 死叉" if last['K'] < last['D'] and prev['K'] > prev['D'] else "")

            final_signal = "觀察"
            if last['MACD_Hist'] > 0 and last['K'] > last['D'] and vol_ratio > 1.0: final_signal = "★ 強勢進攻"
            elif last['RSI'] < 30 and last['K'] < 20: final_signal = "🔫 超跌反彈"

            data_list.append({
                "代號": code, "名稱": name, "現價": round(price, 1),
                "漲跌%": f"{change_pct:+.2f}%", "量能": f"{round(vol_ratio, 1)}x",
                "RSI": round(last['RSI'], 1), "KD值": f"K{int(last['K'])}/D{int(last['D'])}",
                "MACD": macd_signal, "狀態": final_signal + f" {kd_signal}"
            })
        my_bar.progress((i + 1) / len(ticker_list), text=f"正在分析: {name}")
    my_bar.empty()
    return pd.DataFrame(data_list)


# ===========================================================
# 主畫面
# ===========================================================
if is_logged_in():
    st.markdown(f"☁️ **雲端同步模式** — 觀察清單已連結至 Google Drive（`{st.session_state.get('google_user_email', '')}`），修改後自動儲存。")
else:
    st.markdown("📌 目前為**書籤模式**。登入 Google 後可自動儲存清單至雲端，免手動存書籤。")

if user_input:
    result_df = get_stock_data(user_input)
    if not result_df.empty:
        st.markdown("💡 **提示：直接點擊下方表格中的任意一列，即可自動跳轉到「個股深度解析」進行深入分析！**")

        event = st.dataframe(
            result_df,
            use_container_width=True,
            height=400,
            on_select="rerun",
            selection_mode="single-row"
        )

        if len(event.selection.rows) > 0:
            selected_idx = event.selection.rows[0]
            selected_code = result_df.iloc[selected_idx]['代號']
            st.session_state['auto_analyze_ticker'] = str(selected_code)
            st.switch_page("pages/1_個股深度解析.py")

        st.divider()
        st.subheader("🤖 Gemini 戰情室")
        if st.button("呼叫 AI 操盤手"):
            with st.spinner(f'AI 分析中...'):
                analysis_result = get_gemini_analysis(result_df, model_name)
                st.markdown(analysis_result)
    else:
        st.warning("查無數據。")
else:
    st.info("請輸入代號。")
