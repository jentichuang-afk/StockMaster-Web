import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from google import genai

# --- 設定頁面配置 ---
st.set_page_config(page_title="AI 戰情雷達 (雲端永久版)", layout="wide")

st.title("🚀 AI 戰情雷達 - 雲端永久版")
st.markdown("由於雲端伺服器沒有硬碟儲存空間，請利用**網址參數**來存檔清單，修改後只要將**網址存成書籤**即可永久保存！")

# --- 核心功能：從網址讀取寫入清單 (雲端最穩定存檔方式) ---
def get_tickers_from_url():
    """從網址參數讀取清單，如果沒有則使用預設值"""
    # Streamlit 新版 query_params 用法
    params = st.query_params
    if "tickers" in params:
        return params["tickers"]
    return "2330, 2317, 3034, 2376, 2383, 2027, 0050"

def update_url_tickers(new_tickers):
    """更新網址參數"""
    st.query_params["tickers"] = new_tickers

# --- 側邊欄：設定 ---
st.sidebar.header("⚙️ 核心設定")

# 1. 模型選擇
st.sidebar.subheader("🧠 AI 模型引擎")
model_map = {
    "🚀 最新極速版 (Gemini 3.0 Flash)": "gemini-3-flash-preview",
    "⚡ 穩定極速版 (Gemini 2.5 Flash)": "gemini-2.5-flash",
    "🧠 最新深度版 (Gemini 2.5 Pro)": "gemini-2.5-pro",
    "⚡ 穩定極速版 (Gemini 2.0 Flash)": "gemini-2.0-flash",
}
selected_label = st.sidebar.selectbox("選擇分析大腦", list(model_map.keys()), index=0)
model_name = model_map[selected_label]

# 2. 觀察清單 (改用網址記憶)
st.sidebar.subheader("📋 觀察清單 (網址記憶)")

# A. 讀取目前的清單 (從網址)
current_tickers = get_tickers_from_url()

# B. 顯示輸入框
user_input = st.sidebar.text_area(
    "輸入代號 (修改後請點擊外側空白處)", 
    value=current_tickers, 
    height=150,
    help="修改此處內容後，網頁最上方網址列會自動把名單加在後面。請將【更新後的網址加入書籤】，下次點開書籤清單就在！"
)

# C. 如果使用者修改了清單，更新網址
if user_input != current_tickers:
    update_url_tickers(user_input)
    # 強制重新執行以更新畫面
    st.rerun()

# --- 爬蟲抓中文名 ---
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

# --- 技術指標計算核心 ---
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

# --- Gemini AI 分析 ---
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

# --- 抓取數據主程式 ---
def get_stock_data(tickers):
    data_list = []
    clean_tickers = tickers.replace("，", ",").split(',')
    ticker_list = [t.strip() for t in clean_tickers if t.strip()]
    
    my_bar = st.progress(0, text="連線 Yahoo 股市...")
    
    for i, code in enumerate(ticker_list):
        name = get_stock_name_from_web(code)
        symbol = f"{code}.TW"
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        if len(df) < 20: 
            symbol = f"{code}.TWO"; stock = yf.Ticker(symbol); df = stock.history(period="6mo")
        
        if len(df) > 30:
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

# --- 主畫面 ---
if user_input:
    result_df = get_stock_data(user_input)
    if not result_df.empty:
        def highlight_signal(val):
            if '強勢' in val: return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            if '死叉' in val or '偏空' in val: return 'color: #dc3545;'
            return ''
        st.dataframe(result_df.style.map(highlight_signal, subset=['狀態', 'MACD']).map(lambda x: 'color: red' if '-' in x else 'color: green', subset=['漲跌%']), use_container_width=True, height=400)
        
        st.divider()
        st.subheader("🤖 Gemini 戰情室")
        if st.button("呼叫 AI 操盤手"):
            with st.spinner(f'AI 分析中...'):
                analysis_result = get_gemini_analysis(result_df, model_name)
                st.markdown(analysis_result)
    else: st.warning("查無數據。")
else: st.info("請輸入代號。")
