import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google import genai
from groq import Groq
import requests
from bs4 import BeautifulSoup
import openai
from utils.google_drive import is_logged_in, load_expert_config_from_drive, save_expert_config_to_drive

# --- 1. 頁面設定 ---
st.set_page_config(page_title="股票大師：個股深度解析", layout="wide", page_icon="🔍")
st.title("🔍 股票大師：個股全方位深度解析")

# --- 注入自定義 CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

.tech-card {
    background: linear-gradient(145deg, #1e2230, #161a24);
    border: 1px solid #2d3348;
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    font-family: 'Inter', sans-serif;
    position: relative;
    overflow: hidden;
}
.tech-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.tech-card.bullish::before { background: linear-gradient(90deg, #ff4b4b, #ff8c00); }
.tech-card.bearish::before { background: linear-gradient(90deg, #00c805, #00ff88); }
.tech-card.neutral::before { background: linear-gradient(90deg, #888, #aaa); }
.tech-card .label { font-size: 0.75rem; color: #888; margin-bottom: 4px; letter-spacing: 0.05em; }
.tech-card .value { font-size: 1.1rem; font-weight: 700; }
.tech-card .value.red { color: #ff4b4b; }
.tech-card .value.green { color: #00c805; }
.tech-card .value.gray { color: #aaa; }

.pattern-section { margin: 20px 0; }
.pattern-row { display: flex; gap: 16px; }
.pattern-box {
    flex: 1;
    background: linear-gradient(145deg, #1a1e2e, #111420);
    border: 1px solid #2d3348;
    border-radius: 12px;
    padding: 16px;
    font-family: 'Inter', sans-serif;
}
.pattern-box.bullish-box { border-left: 3px solid #ff4b4b; }
.pattern-box.bearish-box { border-left: 3px solid #00c805; }
.pattern-title { font-weight: 700; font-size: 1rem; margin-bottom: 10px; }
.pattern-title.red { color: #ff4b4b; }
.pattern-title.green { color: #00c805; }
.pattern-desc { font-size: 0.85rem; color: #ccc; line-height: 1.6; margin-top: 8px; }
.pattern-status { font-size: 0.8rem; margin-top: 8px; padding: 4px 10px; border-radius: 20px; display: inline-block; }
.pattern-status.active-bull { background: rgba(255,75,75,0.15); color: #ff6b6b; border: 1px solid rgba(255,75,75,0.3); }
.pattern-status.active-bear { background: rgba(0,200,5,0.15); color: #00e806; border: 1px solid rgba(0,200,5,0.3); }
.pattern-status.watch { background: rgba(150,150,150,0.15); color: #aaa; border: 1px solid rgba(150,150,150,0.3); }

.kline-table { width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif; margin-top: 8px; }
.kline-table th { background: #1f2533; color: #888; font-weight: 500; font-size: 0.78rem; padding: 10px 12px; text-align: left; border-bottom: 1px solid #2d3348; }
.kline-table td { padding: 9px 12px; font-size: 0.82rem; color: #ccc; border-bottom: 1px solid #1e2230; }
.kline-table tr:hover td { background: rgba(255,255,255,0.03); }
.kline-icon { font-size: 1.3rem; }
.bull-tag { color: #ff4b4b; font-weight: 600; }
.bear-tag { color: #00c805; font-weight: 600; }
.neu-tag  { color: #aaa; font-weight: 600; }

.signal-item {
    display: flex; align-items: center; gap: 12px;
    background: #1a1e2e; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 8px;
    border-left: 3px solid #2d3348;
    font-family: 'Inter', sans-serif; font-size: 0.85rem;
}
.signal-item.bull-signal { border-left-color: #ff4b4b; }
.signal-item.bear-signal { border-left-color: #00c805; }
.signal-item.neu-signal  { border-left-color: #888; }
</style>
""", unsafe_allow_html=True)

# --- 安全性設定 ---
# API keys are fetched dynamically when `call_ai` is executed


# --- 2. 側邊欄 ---
st.sidebar.header("⚙️ 參數設定")

# 處理跨頁面連動邏輯
auto_run = False

# 初始化儲存輸入框狀態的 key
if 'ticker_input_key' not in st.session_state:
    st.session_state['ticker_input_key'] = "2027"

# 如果從首頁點擊過來，強制更新 key
if 'auto_analyze_ticker' in st.session_state and st.session_state['auto_analyze_ticker'] is not None:
    st.session_state['ticker_input_key'] = st.session_state['auto_analyze_ticker']
    auto_run = True
    # 讀取後馬上清除，避免下次進入頁面又重複觸發
    st.session_state['auto_analyze_ticker'] = None

ticker_input = st.sidebar.text_input("輸入股票代碼", key="ticker_input_key", help="台股請輸入如 2330, 8155")
days_input = st.sidebar.slider("K線觀察天數", 60, 730, 180)

if st.sidebar.button("🔄 刷新圖表"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("📊 指標開關")
show_ma = st.sidebar.checkbox("顯示均線 (MA)", value=True)
show_macd = st.sidebar.checkbox("顯示 MACD", value=True)
show_obv = st.sidebar.checkbox("顯示 OBV", value=True)

run_btn = st.sidebar.button("🚀 AI 技術分析", type="primary")

# 如果是跳轉過來的，強制觸發執行
if auto_run:
    run_btn = True

# --- 3. 核心數據處理 (只抓 K 線) ---
@st.cache_data(ttl=300)
def get_stock_data(symbol, days):
    try:
        # 抓取包含「今天」的數據
        end_date = datetime.now() + timedelta(days=1)
        start_date = end_date - timedelta(days=days+150) # 多抓一些算長天期 MA
        
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty: return None
        return df
    except: return None

# --- 3.1. 輔助功能：爬取真實中文公司名稱 (防 AI 幻覺) ---
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
    return f"代號 {code}"

# --- 4. 技術指標計算 ---
def add_indicators(df):
    # 確保是數值
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # 均線
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean() # 這裡算出了 MA60
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # KD
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    k_list = [50]; d_list = [50]
    for r in df['RSV']:
        if pd.isna(r): k_list.append(50); d_list.append(50)
        else:
            k = (2/3) * k_list[-1] + (1/3) * r
            d = (2/3) * d_list[-1] + (1/3) * k
            k_list.append(k); d_list.append(d)   
    df['K'] = k_list[1:]; df['D'] = d_list[1:]
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    return df

# --- 5. 動態模型獲取 ---
@st.cache_data(ttl=3600)
def fetch_google_models(api_key):
    try:
        if not api_key or api_key.startswith("請輸入"): return ["請先設定 API Key"]
        client = genai.Client(api_key=api_key)
        models = []
        for m in client.models.list():
            if "gemini" in m.name and "vision" not in m.name: # Filter basic ones
                models.append(m.name)
        # Ensure some defaults if API doesn't return cleanly or returns too many
        default_list = ['gemini-3.1-pro-preview', 'gemini-3-flash-preview', 'gemini-2.5-pro', 'gemini-2.5-flash']
        for d in default_list:
            if d not in models:
                models.insert(0, d)
        return models[:20]
    except Exception as e:
        return ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro"]

@st.cache_data(ttl=3600)
def fetch_nvidia_models(api_key):
    try:
        if not api_key or api_key.startswith("請輸入"): return ["請先設定 API Key"]
        client = openai.OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")
        models = client.models.list()
        return sorted([m.id for m in models.data])
    except Exception as e:
        return ["meta/llama-3.3-70b-instruct", "mistralai/mistral-large-2-instruct"]

@st.cache_data(ttl=3600)
def fetch_openrouter_models(api_key):
    try:
        if not api_key or api_key.startswith("請輸入"): return ["請先設定 API Key"]
        client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        models = client.models.list()
        # 只保留名稱中含有 "free" 的免費模型
        free_models = sorted([m.id for m in models.data if "free" in m.id.lower()])
        return free_models if free_models else ["openrouter/free"]
    except Exception as e:
        return ["meta-llama/llama-3.1-8b-instruct:free", "meta-llama/llama-3.3-70b-instruct:free", "mistralai/mistral-7b-instruct:free"]

# --- 6. 技術面視覺化輔助函式 ---

def get_tech_status(df):
    """分析近期資料，回傳各指標狀態字典"""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    status = {}

    # 均線排列
    ma5, ma20, ma60 = last['MA5'], last['MA20'], last['MA60']
    if ma5 > ma20 > ma60:
        status['ma_struct'] = ('多頭排列', 'bullish', 'red')
    elif ma5 < ma20 < ma60:
        status['ma_struct'] = ('空頭排列', 'bearish', 'green')
    else:
        status['ma_struct'] = ('均線糾結', 'neutral', 'gray')

    # 趨勢方向 (價格 vs MA60)
    price = last['Close']
    if price > ma60 * 1.03:
        status['trend'] = ('強勢多頭', 'bullish', 'red')
    elif price > ma60:
        status['trend'] = ('偏多', 'bullish', 'red')
    elif price < ma60 * 0.97:
        status['trend'] = ('強勢空頭', 'bearish', 'green')
    else:
        status['trend'] = ('偏空', 'bearish', 'green')

    # KD 狀態
    k, d = last['K'], last['D']
    pk, pd_ = prev['K'], prev['D']
    if k > 80:
        status['kd'] = ('高檔鈍化', 'bullish', 'red')
    elif k < 20:
        status['kd'] = ('低檔鈍化', 'bearish', 'green')
    elif k > d and pk < pd_:
        status['kd'] = ('黃金交叉', 'bullish', 'red')
    elif k < d and pk > pd_:
        status['kd'] = ('死亡交叉', 'bearish', 'green')
    else:
        status['kd'] = (f'K{k:.0f}/D{d:.0f}', 'neutral', 'gray')

    # MACD 狀態
    hist = last['MACD_Hist']
    prev_hist = prev['MACD_Hist']
    if hist > 0 and hist > prev_hist:
        status['macd'] = ('多頭擴張', 'bullish', 'red')
    elif hist > 0 and hist < prev_hist:
        status['macd'] = ('多頭收斂', 'neutral', 'gray')
    elif hist < 0 and hist < prev_hist:
        status['macd'] = ('空頭擴張', 'bearish', 'green')
    else:
        status['macd'] = ('空頭收斂', 'neutral', 'gray')

    # 量能狀態
    vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
    vol_ratio = last['Volume'] / vol_ma5 if vol_ma5 > 0 else 1
    if vol_ratio > 1.5:
        status['volume'] = (f'爆量 {vol_ratio:.1f}x', 'bullish', 'red')
    elif vol_ratio > 1.0:
        status['volume'] = (f'量增 {vol_ratio:.1f}x', 'bullish', 'red')
    elif vol_ratio < 0.6:
        status['volume'] = (f'縮量 {vol_ratio:.1f}x', 'bearish', 'green')
    else:
        status['volume'] = (f'量平 {vol_ratio:.1f}x', 'neutral', 'gray')

    return status


def render_tech_overview(status):
    """產生技術總覽儀表板 HTML"""
    items = [
        ('趨勢方向', status['trend']),
        ('均線結構', status['ma_struct']),
        ('KD 指標', status['kd']),
        ('MACD', status['macd']),
        ('量能狀態', status['volume']),
    ]
    cards = ""
    for label, (val, card_cls, color_cls) in items:
        cards += f"""
        <div class="tech-card {card_cls}">
            <div class="label">{label}</div>
            <div class="value {color_cls}">{val}</div>
        </div>"""
    return f'<div style="display:flex;gap:12px;margin-bottom:20px;">{cards}</div>'


def render_pattern_diagrams(status):
    """產生 W底/M頭 型態圖解 HTML"""
    # 判斷目前型態狀態 (簡易版：依均線 + MACD 組合)
    trend_cls, ma_cls = status['trend'][1], status['ma_struct'][1]
    if trend_cls == 'bullish' and ma_cls == 'bullish':
        w_status = '<span class="pattern-status active-bull">✅ W底完成・多頭延續</span>'
        m_status = '<span class="pattern-status watch">👀 觀察中・尚未成形</span>'
    elif trend_cls == 'bearish' and ma_cls == 'bearish':
        w_status = '<span class="pattern-status watch">👀 觀察中・尚未成形</span>'
        m_status = '<span class="pattern-status active-bear">⚠️ M頭觀察中・留意破線</span>'
    else:
        w_status = '<span class="pattern-status watch">👀 盤整中</span>'
        m_status = '<span class="pattern-status watch">👀 盤整中</span>'

    return f"""
<div class="pattern-section">
  <div class="pattern-row">
    <div class="pattern-box bullish-box">
      <div class="pattern-title red">📈 W底型態（多頭訊號）</div>
      <svg width="100%" height="70" viewBox="0 0 220 70">
        <line x1="10" y1="38" x2="210" y2="38" stroke="#444" stroke-dasharray="5,3"/>
        <text x="110" y="28" fill="#888" font-size="10" text-anchor="middle">頸線壓力區</text>
        <polyline points="10,20 50,60 90,35 130,60 170,10" fill="none" stroke="#ff4b4b" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="50" cy="60" r="4" fill="#ff8c00"/><text x="50" y="70" fill="#ff8c00" font-size="9" text-anchor="middle">左底</text>
        <circle cx="130" cy="60" r="4" fill="#ff8c00"/><text x="130" y="70" fill="#ff8c00" font-size="9" text-anchor="middle">右底</text>
        <text x="175" y="8" fill="#ff4b4b" font-size="9">突破!</text>
      </svg>
      <div class="pattern-desc">突破頸線 → 趨勢轉強，目標量測：<br>頸線 + (頸線 - 底部) 的距離</div>
      {w_status}
    </div>
    <div class="pattern-box bearish-box">
      <div class="pattern-title green">📉 M頭型態（轉弱訊號）</div>
      <svg width="100%" height="70" viewBox="0 0 220 70">
        <line x1="10" y1="38" x2="210" y2="38" stroke="#444" stroke-dasharray="5,3"/>
        <text x="110" y="50" fill="#888" font-size="10" text-anchor="middle">頸線支撐區</text>
        <polyline points="10,60 50,10 90,35 130,10 170,60" fill="none" stroke="#00c805" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="50" cy="10" r="4" fill="#00c805"/><text x="50" y="8" fill="#00c805" font-size="9" text-anchor="middle">頭部</text>
        <circle cx="130" cy="10" r="4" fill="#00c805"/><text x="130" y="8" fill="#00c805" font-size="9" text-anchor="middle">頭部</text>
        <text x="175" y="65" fill="#00c805" font-size="9">跌破!</text>
      </svg>
      <div class="pattern-desc">跌破頸線 → 多翻空，下跌目標：<br>頸線 - (頭部 - 頸線) 的距離</div>
      {m_status}
    </div>
  </div>
</div>"""


def detect_kline_patterns(df):
    """分析最近幾根 K 棒，回傳一個包含目前已滿足型態名稱的 set"""
    detected = set()
    if len(df) < 3:
        return detected

    c0 = df.iloc[-1]  # 最近一根
    c1 = df.iloc[-2]  # 前一根
    c2 = df.iloc[-3]  # 前兩根

    # --- 單根型態 (最近一根) ---
    o, h, l, c = float(c0['Open']), float(c0['High']), float(c0['Low']), float(c0['Close'])
    body = abs(c - o)
    total_range = h - l if h != l else 1e-9
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    # 十字線：實體 < 全段 10%，且上下影線都存在
    if body / total_range < 0.10 and upper_shadow > 0 and lower_shadow > 0:
        detected.add("十字線")
        # 長十字線：上下影線各 >= 全段 30%
        if upper_shadow / total_range >= 0.30 and lower_shadow / total_range >= 0.30:
            detected.add("長十字線")

    # 錘子線：下影線 >= 實體 2倍，上影線很短，實體在上半
    if lower_shadow >= body * 2 and upper_shadow <= body * 0.5 and body > 0:
        if c1['Close'] > c2['Close']:  # 前兩根下跌趨勢才算
            detected.add("吊人線")
        else:
            detected.add("錘子線")

    # --- 三根型態 ---
    o0, h0, l0, c0_ = float(c0['Open']), float(c0['High']), float(c0['Low']), float(c0['Close'])
    o1, h1, l1, c1_ = float(c1['Open']), float(c1['High']), float(c1['Low']), float(c1['Close'])
    o2, h2, l2, c2_ = float(c2['Open']), float(c2['High']), float(c2['Low']), float(c2['Close'])

    bull0 = c0_ > o0  # 最近一根是紅K
    bull1 = c1_ > o1
    bull2 = c2_ > o2

    # 紅三軍：連續三根紅K，收盤逐日墊高
    if bull0 and bull1 and bull2 and c0_ > c1_ and c1_ > c2_:
        detected.add("紅三軍")

    # 三隻烏鴉：連續三根黑K，收盤逐日下移
    if not bull0 and not bull1 and not bull2 and c0_ < c1_ and c1_ < c2_:
        detected.add("三隻烏鴉")

    # 晨星：第一根大黑K，第二根小實體(任何顏色)，第三根大紅K，且第三根收盤 > 第一根中點
    body2 = abs(c2_ - o2)
    body1 = abs(c1_ - o1)
    body0 = abs(c0_ - o0)
    mid2 = (o2 + c2_) / 2
    if (not bull2) and (body1 < body2 * 0.4) and bull0 and (c0_ > mid2):
        detected.add("晨星")

    # 夜星：第一根大紅K，第二根小實體，第三根大黑K，且第三根收盤 < 第一根中點
    if bull2 and (body1 < body2 * 0.4) and (not bull0) and (c0_ < mid2):
        detected.add("夜星")

    return detected


def render_kline_table(detected_patterns=None):
    """產生 K線百科表 HTML，如有偵測到的型態則高亮對應列"""
    if detected_patterns is None:
        detected_patterns = set()
    rows = [
        ("十字線",     "＋",       "開盤與收盤接近，上下影線均存在",  "多空僵持，轉折訊號", "neu-tag",  "觀望"),
        ("長十字線",   "✛",       "上下影線極長",                    "劇烈震盪，變盤訊號", "neu-tag",  "觀望"),
        ("紅三軍",     "🏮🏮🏮",  "連續三根紅K，收盤逐日墊高",       "多頭強勢訊號",       "bull-tag", "多"),
        ("三隻烏鴉",   "🖤🖤🖤",  "連續三根黑K，收盤逐日下移",       "空頭強勢訊號",       "bear-tag", "空"),
        ("晨星",       "🌟",       "下跌後出現小實體+長紅",           "底部反轉訊號",       "bull-tag", "多"),
        ("夜星",       "⭐",       "上漲後出現小實體+長黑",           "頂部反轉訊號",       "bear-tag", "空"),
        ("錘子線",     "🔨",       "長下影線，實體在上方",            "下跌末端支撐訊號",   "bull-tag", "多"),
        ("吊人線",     "🪢",       "長下影線，出現在高點",            "上漲末端警示訊號",   "bear-tag", "空"),
    ]
    trs = ""
    for name, icon, desc, meaning, tag, side in rows:
        is_active = name in detected_patterns
        if is_active:
            # 決定高亮背景色（依多空分類）
            if tag == "bull-tag":
                row_style = 'background:rgba(255,75,75,0.12); border-left: 3px solid #ff4b4b;'
            elif tag == "bear-tag":
                row_style = 'background:rgba(0,200,5,0.12); border-left: 3px solid #00c805;'
            else:
                row_style = 'background:rgba(255,200,0,0.10); border-left: 3px solid #ffcc00;'
            active_badge = ' <span style="background:#ff4b4b;color:#fff;font-size:0.7rem;padding:2px 7px;border-radius:10px;margin-left:6px;font-weight:700;">今日出現</span>' if tag == 'bull-tag' else ' <span style="background:#00c805;color:#fff;font-size:0.7rem;padding:2px 7px;border-radius:10px;margin-left:6px;font-weight:700;">今日出現</span>' if tag == 'bear-tag' else ' <span style="background:#ffcc00;color:#111;font-size:0.7rem;padding:2px 7px;border-radius:10px;margin-left:6px;font-weight:700;">今日出現</span>'
        else:
            row_style = ''
            active_badge = ''
        trs += f"""<tr style="{row_style}">
            <td><b>{name}</b>{active_badge}</td>
            <td style="text-align:center;font-size:1.1rem;">{icon}</td>
            <td>{desc}</td>
            <td>{meaning}</td>
            <td class="{tag}">{side}</td>
        </tr>"""

    active_count = len(detected_patterns)
    header_note = f' <span style="font-size:0.8rem;color:#ffcc00;font-weight:400;">⚡ 今日偵測到 {active_count} 個型態</span>' if active_count > 0 else ' <span style="font-size:0.8rem;color:#666;font-weight:400;">— 今日無出現</span>'
    no_pattern_msg = '' if active_count > 0 else '<div style="margin-top:10px;padding:10px 14px;background:rgba(255,255,255,0.04);border-radius:8px;border-left:3px solid #444;font-size:0.85rem;color:#888;">🔍 今日 K 線未出現任何特定型態，市場處於常態整理，請繼續觀察後續 K 棒變化。</div>'
    return f"""
<div style="margin-top:24px;">
  <div style="font-weight:700;font-size:1rem;color:#ddd;margin-bottom:10px;">📖 K線型態百科{header_note}</div>
  <table class="kline-table">
    <thead><tr><th>型態名稱</th><th>圖示</th><th>說明</th><th>市場意義</th><th>多/空</th></tr></thead>
    <tbody>{trs}</tbody>
  </table>
  {no_pattern_msg}
</div>"""



def render_recent_signals(df):
    """分析近 10 根 K 棒，產生近期訊號列表 HTML"""
    signals = []
    d10 = df.tail(10).copy()

    for i in range(1, len(d10)):
        row = d10.iloc[i]
        prev = d10.iloc[i - 1]
        date_str = str(row.name)[:10]

        # KD 黃金/死亡交叉
        if row['K'] > row['D'] and prev['K'] < prev['D']:
            signals.append(('bull-signal', '✨', f'{date_str}　KD 黃金交叉 (K={row["K"]:.0f} 上穿 D={row["D"]:.0f})', '多'))
        elif row['K'] < row['D'] and prev['K'] > prev['D']:
            signals.append(('bear-signal', '💀', f'{date_str}　KD 死亡交叉 (K={row["K"]:.0f} 跌穿 D={row["D"]:.0f})', '空'))

        # MACD 柱轉正/負
        if row['MACD_Hist'] > 0 and prev['MACD_Hist'] < 0:
            signals.append(('bull-signal', '📶', f'{date_str}　MACD 柱翻紅，動能轉多', '多'))
        elif row['MACD_Hist'] < 0 and prev['MACD_Hist'] > 0:
            signals.append(('bear-signal', '📉', f'{date_str}　MACD 柱翻黑，動能轉空', '空'))

        # 量能異常
        vol_ma = df['Volume'].rolling(5).mean().iloc[-1]
        if row['Volume'] > vol_ma * 2:
            signals.append(('bull-signal' if row['Close'] > row['Open'] else 'bear-signal',
                           '💥', f'{date_str}　爆量 ({row["Volume"]/vol_ma:.1f}x均量)，{"攻擊量" if row["Close"] > row["Open"] else "出貨量"}', '注意'))

    if not signals:
        signals.append(('neu-signal', '🔍', '近期無明顯技術訊號，市場處於觀望階段', '-'))

    items_html = ""
    for cls, icon, text, side in signals[-5:]:  # 最多顯示 5 個
        color = "#ff4b4b" if "bull" in cls else ("#00c805" if "bear" in cls else "#888")
        items_html += f'<div class="signal-item {cls}"><span style="font-size:1.1rem">{icon}</span><span style="flex:1;color:#ccc">{text}</span><span style="color:{color};font-weight:700;font-size:0.8rem">{side}</span></div>'

    return f"""
<div style="margin-top:20px;">
  <div style="font-weight:700;font-size:1rem;color:#ddd;margin-bottom:10px;">🎯 近期技術訊號偵測</div>
  {items_html}
</div>"""


# --- 6. AI Prompt ---
def get_prompt(symbol, last_close, technical_data):
    now = datetime.now().strftime("%Y-%m-%d")
    
    return f"""
    角色：你是一位精通「技術分析 (Technical Analysis)」的華爾街操盤手。
    
    標的：{symbol}
    現價：{last_close:.2f}
    日期：{now}
    
    請根據下方提供的【近 5 日技術指標數據】，進行純技術面判讀。
    (數據包含：收盤價, MA5, MA20, MA60, KD值, MACD, OBV)
    
    {technical_data}
    
    請撰寫一份【技術操作策略】：
    1. 🕵️‍♂️ **趨勢判讀**：
       - **均線排列**：請根據 MA5, MA20, MA60 的數值大小，判斷是多頭排列 (>MA20>MA60) 還是空頭排列？
       - **乖離率**：股價是否遠離 MA20 或 MA60？
    
    2. ⚔️ **指標訊號**：
       - **KD 指標**：黃金交叉/死亡交叉/鈍化？
       - **MACD**：多空力道變化。
       
    3. 🎯 **關鍵價位與策略**：
       - **操作建議**：(強力買進 / 拉回買進 / 觀望 / 反彈空 / 強力賣出)
       - **理由**：請引用上方的 MA60 或其他數據作為支撐。
    """

def call_ai(model_type, prompt):
    try:
        if model_type == 'gemini':
            gemini_key = st.secrets.get("GEMINI_API_KEY")
            if not gemini_key or gemini_key.startswith("請輸入"):
                 return "API Key 未設定 (請在 secrets.toml 填寫有效的 GEMINI_API_KEY)"
            
            client = genai.Client(api_key=gemini_key)
            
            # 從 session_state 抓取使用者首頁選取的模型，如果沒有則用預設值
            selected_model = st.session_state.get('selected_gemini_model', 'gemini-3-flash-preview')
            
            response = client.models.generate_content(
                model=selected_model,
                contents=prompt,
            )
            return response.text
            
        elif model_type == 'groq':
            groq_key = st.secrets.get("GROQ_API_KEY")
            if not groq_key or groq_key.startswith("請輸入"):
                 return "API Key 未設定 (請在 secrets.toml 填寫有效的 GROQ_API_KEY)"
                 
            groq_client = Groq(api_key=groq_key)
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            # Force string to properly decode as utf-8 if environment enforces ascii
            content = response.choices[0].message.content
            if isinstance(content, bytes):
                return content.decode('utf-8')
            return content
            
    except Exception as e:
        return f"AI 忙碌中或發生錯誤: {str(e)}"
    return "未知的模型類型"


def fetch_stock_news(symbol: str, max_items: int = 10) -> str:
    """使用 yfinance 抓取個股最新新聞標題，回傳格式化字串供注入 system prompt。"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        if not news:
            return "（暫無最新新聞）"
        lines = []
        for i, item in enumerate(news[:max_items], 1):
            content = item.get("content", {})
            title = content.get("title") or item.get("title", "(無標題)")
            pub = content.get("pubDate") or item.get("providerPublishTime", "")
            if pub and not isinstance(pub, str):
                try:
                    pub = datetime.utcfromtimestamp(pub).strftime("%Y-%m-%d")
                except Exception:
                    pub = str(pub)
            pub_str = f" ({pub})"
            lines.append(f"{i}. {title}{pub_str}")
        return "\n".join(lines)
    except Exception as e:
        return f"（新聞抓取失敗: {e}）"


def call_expert_chat(provider, model_name, system_prompt, history, symbol=""):
    """Non-streaming fallback: returns full response as string."""
    from google.genai import types as genai_types
    try:
        if provider == 'Google':
            api_key = st.secrets.get("GEMINI_API_KEY")
            if not api_key: return "Google API Key 未設定"
            client = genai.Client(api_key=api_key)
            contents = []
            for msg in history:
                role = 'user' if msg['role'] == 'user' else 'model'
                text = f"{msg['name']}說：{msg['content']}" if msg['role'] == 'assistant' else msg['content']
                contents.append({'role': role, 'parts': [{'text': text}]})
            # 啟用 Google Search 聯網搜尋工具
            search_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
            response = client.models.generate_content(
                model=model_name, contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[search_tool]
                )
            )
            return response.text

        elif provider in ('Nvidia', 'OpenRouter'):
            if provider == 'Nvidia':
                api_key = st.secrets.get("NVIDIA_API_KEY")
                base_url = "https://integrate.api.nvidia.com/v1"
            else:
                api_key = st.secrets.get("OPENROUTER_API_KEY")
                base_url = "https://openrouter.ai/api/v1"
            if not api_key: return f"{provider} API Key 未設定"
            # 預抓最新新聞注入 system prompt
            news_text = fetch_stock_news(symbol) if symbol else "（未指定股票代號）"
            enriched_prompt = system_prompt + f"\n\n【最新市場新聞（來自 Yahoo Finance）】：\n{news_text}"
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            messages = [{"role": "system", "content": enriched_prompt}]
            for msg in history:
                text = f"[{msg['name']}] {msg['content']}" if msg['role'] == 'assistant' else msg['content']
                messages.append({"role": msg['role'], "content": text})
            response = client.chat.completions.create(model=model_name, messages=messages)
            return response.choices[0].message.content
    except Exception as e:
        return f"API 呼叫失敗: {str(e)}"
    return "未知的 Provider"


def stream_expert_chat(provider, model_name, system_prompt, history, symbol=""):
    """Streaming version: yields text chunks so st.write_stream() can render in real time.
    - Google (Gemini): uses native google_search grounding tool for real-time web search.
    - Nvidia / OpenRouter: pre-fetches latest Yahoo Finance news and injects into system prompt.
    """
    from google.genai import types as genai_types
    try:
        if provider == 'Google':
            api_key = st.secrets.get("GEMINI_API_KEY")
            if not api_key:
                yield "Google API Key 未設定"; return
            client = genai.Client(api_key=api_key)
            contents = []
            for msg in history:
                role = 'user' if msg['role'] == 'user' else 'model'
                text = f"{msg['name']}說：{msg['content']}" if msg['role'] == 'assistant' else msg['content']
                contents.append({'role': role, 'parts': [{'text': text}]})
            # 啟用 Google Search 聯網搜尋工具（Gemini 會自動決定何時搜尋）
            search_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
            for chunk in client.models.generate_content_stream(
                model=model_name, contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[search_tool]
                )
            ):
                if chunk.text:
                    yield chunk.text

        elif provider in ('Nvidia', 'OpenRouter'):
            if provider == 'Nvidia':
                api_key = st.secrets.get("NVIDIA_API_KEY")
                base_url = "https://integrate.api.nvidia.com/v1"
            else:
                api_key = st.secrets.get("OPENROUTER_API_KEY")
                base_url = "https://openrouter.ai/api/v1"
            if not api_key:
                yield f"{provider} API Key 未設定"; return
            # 預抓最新新聞注入 system prompt，補強模型的即時資訊
            news_text = fetch_stock_news(symbol) if symbol else "（未指定股票代號）"
            enriched_prompt = system_prompt + f"\n\n【最新市場新聞（來自 Yahoo Finance，請務必參考）】：\n{news_text}"
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            messages = [{"role": "system", "content": enriched_prompt}]
            for msg in history:
                text = f"[{msg['name']}] {msg['content']}" if msg['role'] == 'assistant' else msg['content']
                messages.append({"role": msg['role'], "content": text})
            stream = client.chat.completions.create(
                model=model_name, messages=messages, stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        else:
            yield "未知的 Provider"
    except Exception as e:
        yield f"API 呼叫失敗: {str(e)}"

def update_dynamic_questions(final_symbol, history, status_desc):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return
    try:
        client = genai.Client(api_key=api_key)
        recent_history = history[-4:] if len(history) >= 4 else history
        history_text = "\n".join([f"{msg['name']}: {msg['content']}" for msg in recent_history])
        
        prompt = f"""
        請擔任頂尖的投資顧問。根據以下股票數據與最近的專家討論內容，產生 3 個最值得深入探討的後續追問問題。
        
        【股票數據】：
        {status_desc}
        
        【最近對話】：
        {history_text}
        
        【要求】：
        1. 必須是繁體中文。
        2. 問題要簡潔犀利，每條問題字數控制在 30 字以內，極具探討價值。
        3. 回傳格式「必須」為純 JSON 陣列，例如：
        ["問題一", "問題二", "問題三"]
        不要有防 Markdown 標籤 (例如 ```json) 或其他文字！
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        
        import json
        questions = json.loads(text)
        if isinstance(questions, list) and len(questions) >= 3:
            st.session_state[f"dynamic_questions_{final_symbol}"] = questions[:3]
    except Exception as e:
        pass

# --- 6. 主程式 ---
if run_btn or auto_run:
    st.session_state['show_analysis_page'] = True

if st.session_state.get('show_analysis_page', False) and ticker_input:
    raw_ticker = ticker_input.strip().upper()
    
    final_symbol = raw_ticker
    df = None
    
    with st.spinner(f"正在繪製 {raw_ticker} 技術線圖..."):
        if raw_ticker.isdigit():
            for s in ['.TW', '.TWO']:
                df = get_stock_data(raw_ticker + s, days_input)
                if df is not None:
                    final_symbol = raw_ticker + s
                    break
        else:
            df = get_stock_data(raw_ticker, days_input)
    
    if df is None:
        st.error(f"❌ 查無代碼 {raw_ticker}")
    else:
        df = add_indicators(df)
        df_display = df.iloc[-days_input:]
        
        last = df.iloc[-1]
        chg = last['Close'] - df['Close'].iloc[-2]
        pct = (chg / df['Close'].iloc[-2]) * 100
        
        st.markdown(f"## 🔥 {final_symbol} 技術戰情室")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("收盤價", f"{last['Close']:.2f}", f"{pct:.2f}%")
        c2.metric("MA5", f"{last['MA5']:.2f}")
        c3.metric("MA20 (月線)", f"{last['MA20']:.2f}")
        c4.metric("MA60 (季線)", f"{last['MA60']:.2f}") # 這裡有顯示，代表有算出來

        st.markdown("""---""")
        mega_btn = st.button("🌟 一鍵啟動四大 AI 分析 (節省 API 額度與時間)", type="primary", use_container_width=True)
        if mega_btn:
            with st.spinner("🚀 AI 正在彙整所有數據並全面解析，請稍候... (約需 15-30 秒)"):
                import re
                
                target_cols = ['Close', 'MA5', 'MA20', 'MA60', 'K', 'D', 'MACD', 'MACD_Hist', 'OBV']
                tech_data_str = df.tail(5)[target_cols].to_string()
                
                fin_data = {}
                stock_info = {'名稱': get_stock_name_from_web(raw_ticker), '產業': '未知'}
                news_text = ""
                try:
                    ticker_obj = yf.Ticker(final_symbol)
                    info = ticker_obj.info
                    stock_info['產業'] = info.get('industry', '未知')
                    
                    fin_data['本益比(PE)'] = info.get('trailingPE', '未知')
                    fin_data['預估本益比(F-PE)'] = info.get('forwardPE', '未知')
                    fin_data['股價淨值比(PB)'] = info.get('priceToBook', '未知')
                    fin_data['ROE'] = info.get('returnOnEquity', '未知')
                    fin_data['營收 YoY'] = info.get('revenueGrowth', '未知')
                    fin_data['毛利率'] = info.get('grossMargins', '未知')
                    fin_data['營業利益率'] = info.get('operatingMargins', '未知')
                    fin_data['負債權益比'] = info.get('debtToEquity', '未知')
                    fin_data['自由現金流'] = info.get('freeCashflow', '未知')
                    
                    news_list = ticker_obj.news
                    if news_list:
                        news_text = "\n".join([f"- {item['title']}" for item in news_list[:5] if 'title' in item])
                except:
                    pass
                
                debate_bg = "\n".join([f"- {k}: {v}" for k, v in fin_data.items() if v != '未知'])
                if not debate_bg: debate_bg = "無法獲取最新財務數據。"
                sentiment_info = f"新聞與焦點：\n{news_text}" if news_text else "查無近期特定新聞。"

                mega_prompt = f"""
                你現在是一組頂尖的「華爾街全方位 AI 投研團隊」。
                我們正在分析標的：{final_symbol} (真實公司名稱: {stock_info.get('名稱')}, 產業: {stock_info.get('產業')})
                現在時間：{datetime.now().strftime("%Y-%m-%d")}
                現價：{last['Close']:.2f}
                
                ⚠️【絕對防幻覺指令】⚠️：本公司的絕對真實名稱為「{stock_info.get('名稱')}」，請嚴格以此名稱展開所有分析，絕對禁止你根據股票代號去猜測其他不相干的公司（例如絕對不能把 4573 錯認為萬潤，它就是高明鐵）！所有的產業地位、護城河與新聞情緒，都必須 100% 針對「{stock_info.get('名稱')}」這家公司來評估！

                【提供的情報】
                [1. 近期技術面數據]
                {tech_data_str}
                
                [2. 基本與財務核心數據]
                {debate_bg}
                
                [3. 近期新聞與市場焦點]
                {sentiment_info}
                
                【🌟 最高任務指令與格式要求 🌟】
                請你根據上述所有的情報，同時產出四份獨立的專業分析報告。
                你「必須」嚴格輸出以下四個 XML 標籤區塊，並將對應的報告內容寫在其內。絕對不能遺漏任何一個標籤。

                <technical_analysis>
                (分析任務：根據技術數據判斷趨勢、指標訊號(如MACD, KD)、乖離率，給出明確的操作指導與支撐壓力理由)
                </technical_analysis>

                <fundamental_analysis>
                (分析任務：根據核心業務、競爭對手與護城河、未來催化劑、潛在總經風險，對這家公司的長線價值進行深度定調)
                </fundamental_analysis>

                <sentiment_analysis>
                (分析任務：根據新聞與市場預期，分析散戶風向、聰明錢/法人的可能動向，給出目前的極端情緒溫度定調與反直覺的警告)
                </sentiment_analysis>

                <ai_debate>
                (分析任務：舉辦投資委員會多空激辯。同時扮演「火箭老哥🚀(樂觀)」、「巴菲特信徒👴(看重估值)」、「放空大王🐻(挑剔財報弱點)」、「投資總監👨‍⚖️(結語裁決)」。每人至少發言1到2次。
                ⚠️ 特別規定：辯論時他們彼此互相攻擊的論點「必須具體引用數字」，例如我在上方提供的【財務核心數據】或【技術價格】，不能只說空話！)
                </ai_debate>
                
                (請確保每個 XML 標籤都有正確閉合，以利程式系統解析。並且全程使用繁體中文)
                """

                def parse_mega(text):
                    res = {}
                    for tag in ['technical_analysis', 'fundamental_analysis', 'sentiment_analysis', 'ai_debate']:
                        m = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
                        content = m.group(1).strip() if m else f"⚠️ 解析失敗，AI 回覆可能被截斷或未按 XML 格式輸出此區塊。\n\n原始回應預覽：{text[:200]}..."
                        res[tag] = content
                    return res
                
                mega_gemini = call_ai('gemini', mega_prompt)
                parsed_gemini = parse_mega(mega_gemini)
                st.session_state[f"tech_result_gemini_{final_symbol}"] = parsed_gemini['technical_analysis']
                st.session_state[f"fundamental_result_gemini_{final_symbol}"] = parsed_gemini['fundamental_analysis']
                st.session_state[f"sentiment_result_gemini_{final_symbol}"] = parsed_gemini['sentiment_analysis']
                st.session_state[f"debate_result_gemini_{final_symbol}"] = parsed_gemini['ai_debate']

                mega_groq = call_ai('groq', mega_prompt)
                parsed_groq = parse_mega(mega_groq)
                st.session_state[f"tech_result_groq_{final_symbol}"] = parsed_groq['technical_analysis']
                st.session_state[f"fundamental_result_groq_{final_symbol}"] = parsed_groq['fundamental_analysis']
                st.session_state[f"sentiment_result_groq_{final_symbol}"] = parsed_groq['sentiment_analysis']
                st.session_state[f"debate_result_groq_{final_symbol}"] = parsed_groq['ai_debate']
                
                st.success("✅ 四大分析報告已全面生成完畢！請直接點擊下方各分頁查看結果。")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 技術分析圖表", "🤖 AI 操盤建議", "🏛️ 基本面分析", "📰 市場情緒分析", "🗣️ AI 多空辯論", "💬 專家聯合會診"])
        
        with tab1:
            # --- 技術面總覽儀表板 ---
            tech_status = get_tech_status(df)
            st.markdown("#### 📊 技術指標總覽")
            st.markdown(render_tech_overview(tech_status), unsafe_allow_html=True)

            # --- K 線圖表 ---
            rows = 2
            if show_macd: rows += 1
            if show_obv: rows += 1
            row_heights = [0.6] + [0.4/(rows-1)] * (rows-1)
            
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.03)
            
            fig.add_trace(go.Candlestick(x=df_display.index, open=df_display['Open'], high=df_display['High'], 
                                         low=df_display['Low'], close=df_display['Close'], name='K線'), row=1, col=1)
            if show_ma:
                fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MA5'], line=dict(color='yellow', width=1), name='MA5'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MA20'], line=dict(color='orange', width=1.5), name='MA20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MA60'], line=dict(color='purple', width=1.5), name='MA60'), row=1, col=1)
            
            curr_row = 2
            colors = ['red' if c >= o else 'green' for c, o in zip(df_display['Close'], df_display['Open'])]
            fig.add_trace(go.Bar(x=df_display.index, y=df_display['Volume'], marker_color=colors, name='成交量'), row=curr_row, col=1)
            curr_row += 1
            
            if show_macd:
                hist_color = ['red' if v >= 0 else 'green' for v in df_display['MACD_Hist']]
                fig.add_trace(go.Bar(x=df_display.index, y=df_display['MACD_Hist'], marker_color=hist_color, name='MACD柱'), row=curr_row, col=1)
                fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MACD'], line=dict(color='orange', width=1), name='DIF'), row=curr_row, col=1)
                fig.add_trace(go.Scatter(x=df_display.index, y=df_display['Signal'], line=dict(color='blue', width=1), name='DEM'), row=curr_row, col=1)
                curr_row += 1
                
            if show_obv:
                fig.add_trace(go.Scatter(x=df_display.index, y=df_display['OBV'], line=dict(color='cyan', width=1), name='OBV', fill='tozeroy'), row=curr_row, col=1)
            
            fig.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # --- 近期訊號偵測 ---
            st.markdown(render_recent_signals(df), unsafe_allow_html=True)

            st.markdown("---")

            # --- 型態圖解 (W底 / M頭) ---
            st.markdown("#### 📐 技術型態圖解")
            st.markdown(render_pattern_diagrams(tech_status), unsafe_allow_html=True)

            # --- K 線百科（含今日型態高亮）---
            detected_patterns = detect_kline_patterns(df)
            st.markdown(render_kline_table(detected_patterns), unsafe_allow_html=True)


        with tab2:
            st.markdown("### 🤖 AI 技術面操作建議")
            if st.button("單獨啟動技術面分析 (Technical AI)"):
                with st.spinner("AI 正在針對技術面與量價結構進行單獨解析..."):
                    target_cols = ['Close', 'MA5', 'MA20', 'MA60', 'K', 'D', 'MACD', 'MACD_Hist', 'OBV']
                    tech_data_str = df.tail(5)[target_cols].to_string()
                    
                    prompt = get_prompt(final_symbol, last['Close'], tech_data_str)
                    
                    st.session_state[f"tech_result_gemini_{final_symbol}"] = call_ai('gemini', prompt)
                    st.session_state[f"tech_result_groq_{final_symbol}"] = call_ai('groq', prompt)
            
            if f"tech_result_gemini_{final_symbol}" in st.session_state and f"tech_result_groq_{final_symbol}" in st.session_state:
                res_gemini = st.session_state[f"tech_result_gemini_{final_symbol}"]
                res_groq = st.session_state[f"tech_result_groq_{final_symbol}"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🔵 Gemini 操盤建議")
                    if "未設定" in res_gemini or "錯誤" in res_gemini:
                        st.error(res_gemini)
                    else:
                        st.info(res_gemini)
                
                with col2:
                    st.markdown("### 🟠 Llama 3 操盤建議")
                    if "未設定" in res_groq or "錯誤" in res_groq:
                        st.error(res_groq)
                    else:
                        st.warning(res_groq)
                        
        with tab3:
            st.markdown(f"### 🏛️ {final_symbol} 基本面與產業分析")
            st.markdown("利用 AI 結合常識與最新市場洞察，深入剖析該公司的基本面體質。")
            
            if st.button("啟動基本面分析深潛 (Deep Dive)"):
                with st.spinner("AI 正在調閱該公司的產業定位、護城河與財務特徵..."):
                    
                    # 嘗試抓取基本的公司資訊給 AI 參考 (非必須，但能提升回答品質)
                    stock_info = {'名稱': get_stock_name_from_web(raw_ticker), '產業': '未知', '市值': '未知'}
                    try:
                        ticker_obj = yf.Ticker(final_symbol)
                        info = ticker_obj.info
                        stock_info['產業'] = info.get('industry', '未知')
                        stock_info['市值'] = info.get('marketCap', '未知')
                        stock_info['本益比(PE)'] = info.get('trailingPE', '未知')
                        stock_info['股東權益報酬率(ROE)'] = info.get('returnOnEquity', '未知')
                    except:
                        pass
                    
                    bg_info = f"參考數據：名稱={stock_info.get('名稱')}, 產業={stock_info.get('產業')}, 本益比={stock_info.get('本益比(PE)')}, ROE={stock_info.get('股東權益報酬率(ROE)')}" if stock_info else "無額外參考數據"
                    
                    fundamental_prompt = f"""
                    你現在是一位頂尖的「基本面分析師 (Fundamental Analyst)」與「產業研究員」。
                    
                    分析標的：{final_symbol}
                    ⚠️【絕對防幻覺指令】⚠️：本公司的絕對真實名稱為「{stock_info.get('名稱')}」，這是鐵錚錚的事實。分析時必須全程針對「{stock_info.get('名稱')}」展開，嚴禁你因為代號而去猜測、硬套到其他公司名稱上！任何張冠李戴的行為都將視為嚴重失職。
                    目前已知背景資訊：{bg_info}
                    現在時間：{datetime.now().strftime("%Y-%m-%d")}
                    
                    請利用你龐大的資料庫與對全球產業鏈的理解，針對「{stock_info.get('名稱')}」這家公司撰寫一份深入且專業的基本面分析報告。
                    
                    報告請嚴格依循以下架構撰寫，並使用繁體中文，語氣需專業、客觀且具備洞察力：
                    
                    ### 🏢 1. 公司介紹與核心業務 (Business Model)
                       - 這家公司主要靠什麼賺錢？
                       - 它在產業鏈(上下游)中扮演什麼角色？
                    
                    ### 🏰 2. 產業護城河 (Economic Moat)
                       - 它擁有什麼樣的競爭優勢？(例如：規模經濟、專利技術、轉換成本、品牌效應或特許經營權)
                       - 競爭對手是誰？它憑什麼贏過對手？
                    
                    ### 🚀 3. 未來成長動能與催化劑 (Growth Catalysts)
                       - 短中期內，有什麼關鍵趨勢、新產品或市場題材能推動它的營收或獲利成長？(例如 AI 趨勢、政策利多等)
                    
                    ### ⚠️ 4. 潛在風險與逆風 (Risks)
                       - 投資這家公司需要留意什麼致命傷或總經風險？(例如：匯率、原物料價格、地緣政治、競爭加劇)
                    
                    ### 💡 5. 總結與長線投資價值定調
                       - 總結這家公司的體質。
                       - 給予一句話的長線投資人建議 (例如：「適合防禦型存股族」、「適合承擔高風險追求成長的投資人」等)。
                    """
                    
                    result_gemini = call_ai('gemini', fundamental_prompt)
                    st.session_state[f"fundamental_result_gemini_{final_symbol}"] = result_gemini
                    
                    result_groq = call_ai('groq', fundamental_prompt)
                    st.session_state[f"fundamental_result_groq_{final_symbol}"] = result_groq

            # 顯示分析結果 (如果是之前已經分析過的，也會顯示出來)
            if f"fundamental_result_gemini_{final_symbol}" in st.session_state and f"fundamental_result_groq_{final_symbol}" in st.session_state:
                res_gemini = st.session_state[f"fundamental_result_gemini_{final_symbol}"]
                res_groq = st.session_state[f"fundamental_result_groq_{final_symbol}"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🔵 Gemini 基本面報告")
                    if "未設定" in res_gemini or "錯誤" in res_gemini:
                        st.error(res_gemini)
                    else:
                        st.info(res_gemini)
                
                with col2:
                    st.markdown("### 🟠 Llama 3 基本面報告")
                    if "未設定" in res_groq or "錯誤" in res_groq:
                        st.error(res_groq)
                    else:
                        st.warning(res_groq)

        with tab4:
            st.markdown(f"### 📰 {final_symbol} 市場情緒分析")
            st.markdown("分析近期市場新聞、論壇風向與機構觀點，抓出市場對這家公司的真實看法與情緒溫度。")
            
            if st.button("啟動市場情緒雷達 (Sentiment Scanner)"):
                with st.spinner("AI 正在掃描全網新聞標題與市場輿論風向..."):
                    
                    # 嘗試抓取近期的 Yahoo 財經新聞
                    news_text = ""
                    try:
                        ticker_obj = yf.Ticker(final_symbol)
                        news_list = ticker_obj.news
                        if news_list:
                            # 提取最多 5 則新聞標題作為市場情緒參考
                            news_titles = [f"- {item['title']}" for item in news_list[:5] if 'title' in item]
                            news_text = "\n".join(news_titles)
                    except:
                        pass
                    
                    sentiment_info = f"【近期盤面對應新聞與焦點】：\n{news_text}" if news_text else "查無近期特定新聞，請透過 AI 本身對這家公司近期話題的知識進行分析。"
                    
                    sentiment_prompt = f"""
                    你現在是一位敏銳的「市場情緒分析師 (Sentiment Analyst)」與「行為金融學專家」。
                    
                    分析標的：{final_symbol}
                    現在時間：{datetime.now().strftime("%Y-%m-%d")}
                    
                    以下是近期市場上關於這家公司的最新新聞標題或是近期焦點：
                    {sentiment_info}
                    
                    請利用這些資訊，並結合你對總體經濟、近期科技趨勢與投資人心理的理解，分析市場目前對這家公司的「真實情緒」與「預期心理」。
                    
                    報告請嚴格依循以下架構撰寫，並使用繁體中文，語氣需具備市場敏銳度、客觀且一針見血：
                    
                    ### 🌡️ 1. 整體市場情緒溫度表 (Sentiment Gauge)
                       - 極度狂熱 / 偏向樂觀 / 中立觀望 / 偏向悲觀 / 極度恐慌？請給出一個明確的定調。
                       - 市場目前對這家公司最大的「期待」和「恐懼」分別是什麼？
                    
                    ### 🗣️ 2. 大眾與散戶的真實風向 (Retail Perspective)
                       - 近期散戶在討論什麼？(例如：股息該不該領、利多出盡、還是買不到好焦慮？)
                       - 散戶目前是正在瘋狂追價，還是急著停損解套？
                    
                    ### 🏦 3. 法人機構與聰明錢的動向預測 (Smart Money View)
                       - 法人通常用什麼角度看這家公司近期的題材？(例如：認為新聞是短期炒作，還是長線實質利多？)
                       - 外資或主力近期可能正在做什麼佈局(請合乎常理與現況推測)？
                    
                    ### ⚖️ 4. 逆思考與潛在反轉點 (Contrarian View)
                       - 人多的地方不要去。根據目前的極端情緒（如果有的話），是不是有超跌錯殺，或者是股價透支未來的狀況？
                       - 你會給現在想「進場」或「出場」的投資人什麼反直覺的逆勢操作警告？
                    """
                    
                    res_sent_gemini = call_ai('gemini', sentiment_prompt)
                    st.session_state[f"sentiment_result_gemini_{final_symbol}"] = res_sent_gemini
                    
                    res_sent_groq = call_ai('groq', sentiment_prompt)
                    st.session_state[f"sentiment_result_groq_{final_symbol}"] = res_sent_groq

            # 顯示分析結果
            if f"sentiment_result_gemini_{final_symbol}" in st.session_state and f"sentiment_result_groq_{final_symbol}" in st.session_state:
                sent_gemini = st.session_state[f"sentiment_result_gemini_{final_symbol}"]
                sent_groq = st.session_state[f"sentiment_result_groq_{final_symbol}"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🔵 Gemini 情緒解析")
                    if "未設定" in sent_gemini or "錯誤" in sent_gemini:
                        st.error(sent_gemini)
                    else:
                        st.info(sent_gemini)
                
                with col2:
                    st.markdown("### 🟠 Llama 3 情緒解析")
                    if "未設定" in sent_groq or "錯誤" in sent_groq:
                        st.error(sent_groq)
                    else:
                        st.warning(sent_groq)

        with tab5:
            pass
            st.markdown(f"### 🗣️ {final_symbol} AI 多空辯論")
            st.markdown("讓 AI 同時扮演**成長型主管**、**價值型老手**、以及**惡意做空機構**，展開精彩的投資辯論大會！")
            
            if st.button("舉辦投研辯論會 (Investment Debate)"):
                with st.spinner("AI 正在切換多重人格並調閱財務數據，準備召開圓桌會議..."):
                    
                    # 抓取真實財務數據供辯論使用，避免 AI 只講空話
                    fin_data = {}
                    try:
                        ticker_obj = yf.Ticker(final_symbol)
                        info = ticker_obj.info
                        fin_data['本益比(Trailing PE)'] = info.get('trailingPE', '未知')
                        fin_data['預估本益比(Forward PE)'] = info.get('forwardPE', '未知')
                        fin_data['股價淨值比(PB)'] = info.get('priceToBook', '未知')
                        fin_data['股東權益報酬率(ROE)'] = info.get('returnOnEquity', '未知')
                        fin_data['營收成長率(YoY)'] = info.get('revenueGrowth', '未知')
                        fin_data['毛利率(Gross Margin)'] = info.get('grossMargins', '未知')
                        fin_data['營業利益率(Operating Margin)'] = info.get('operatingMargins', '未知')
                        fin_data['負債權益比(Debt to Equity)'] = info.get('debtToEquity', '未知')
                        fin_data['自由現金流'] = info.get('freeCashflow', '未知')
                        fin_data['52週高點'] = info.get('fiftyTwoWeekHigh', '未知')
                        fin_data['52週低點'] = info.get('fiftyTwoWeekLow', '未知')
                    except Exception as e:
                        pass
                    
                    debate_bg = "\n".join([f"- {k}: {v}" for k, v in fin_data.items() if v != '未知'])
                    if not debate_bg:
                        debate_bg = "無法獲取最新財務數據，請根據你對該公司的過往認知進行推論。"

                    debate_prompt = f"""
                    你現在是華爾街最頂尖對沖基金的「投資委員會」。
                    
                    我們現在要針對以下標的進行投資決策會議：
                    分析標的：{final_symbol}
                    現在時間：{datetime.now().strftime("%Y-%m-%d")}
                    目前最新技術面收盤價：{last['Close']}
                    
                    【🔥 核心財務數據 (關鍵證據) 🔥】
                    {debate_bg}
                    
                    請你**同時扮演**以下四位角色，並讓他們進行一場真實、激烈、充滿火花的辯論會議。各角色的發言必須符合其人設，並盡可能指出其他人的盲點。
                    
                    ⚠️ **【最高指令】** ⚠️
                    每位角色發言時，**絕對必須引用上述提供的「核心財務數據」或「技術面價格」中的具體數字**來作為支持自己論點的鐵證或攻擊對手的武器！不能只講空泛的理論，例如不能只說「這家公司財報不好」，必須說出「毛利率只有 X% 或 ROE 掉到 Y%」！
                    
                    **【角色列表】**：
                    🚀 **「火箭老哥」(成長型多頭)**：極度樂觀，專看 AI、科技革命、未來十年爆發力。喜歡拿最新的營收成長或是預估本益比來說嘴，認為現在的高估值只是為未來的超額利潤買單。
                    👴 **「巴菲特信徒」(保守價值投資人)**：謹慎嚴謹，非常在意目前的本益比(PE)、股價淨值比(PB)、現金流與護城河。對於火箭老哥的高估值論點嗤之以鼻，堅持只買便宜且具備安全邊際的數字。
                    🐻 **「放空大王」(惡意做空機構 CEO)**：挑剔、刻薄、喜歡雞蛋裡挑骨頭。專挑負面數據打，例如拿高額負債(Debt to Equity)、下滑的利潤率(Margin)或技術面的弱勢來痛打多頭。他的工作就是戳破火箭老哥的泡泡。
                    👨‍⚖️ **「投資總監」(宏觀中立裁判)**：負責開場與最後拍板定案，結語一定要客觀點出三方提到的這間公司「最關鍵的財務數字衝突」，並給出「委員會最終決議」。
                    
                    **【會議劇本要求】**：
                    請用生動如同劇本對話的格式呈現，包含每個人彼此互相吐槽和反駁，最後由投資總監做出客觀的總結裁決。
                    字數不用太長，每人發言 1~2 次即可，但句句都要切中這家公司的核心！
                    請全程使用繁體中文。
                    """
                    
                    res_debate_gemini = call_ai('gemini', debate_prompt)
                    st.session_state[f"debate_result_gemini_{final_symbol}"] = res_debate_gemini
                    
                    res_debate_groq = call_ai('groq', debate_prompt)
                    st.session_state[f"debate_result_groq_{final_symbol}"] = res_debate_groq

            # 顯示分析結果
            if f"debate_result_gemini_{final_symbol}" in st.session_state and f"debate_result_groq_{final_symbol}" in st.session_state:
                deb_gemini = st.session_state[f"debate_result_gemini_{final_symbol}"]
                deb_groq = st.session_state[f"debate_result_groq_{final_symbol}"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🔵 Gemini 多空激辯現場")
                    if "未設定" in deb_gemini or "錯誤" in deb_gemini:
                        st.error(deb_gemini)
                    else:
                        st.info(deb_gemini)
                
                with col2:
                    st.markdown("### 🟠 Llama 3 多空激辯現場")
                    if "未設定" in deb_groq or "錯誤" in deb_groq:
                        st.error(deb_groq)
                    else:
                        st.warning(deb_groq)

        with tab6:
            st.markdown(f"### 💬 {final_symbol} AI 專家聯合會診")
            st.markdown("在這裡，您可以同時與多位不同模型設定的 AI 投資專家針對該股進行深度探討。每一位專家都能看到討論串中其他專家的意見，進行交談與辯論。")
            
            # --- 1. 初始化 Session State ---
            if f"expert_chat_history_{final_symbol}" not in st.session_state:
                st.session_state[f"expert_chat_history_{final_symbol}"] = []
                
            if "expert_configs" not in st.session_state:
                st.session_state["expert_configs"] = [
                    {"name": "巴菲特價值專員", "provider": "Google", "model": "gemini-2.5-flash"},
                    {"name": "凱薩琳科技女皇", "provider": "Nvidia", "model": "meta/llama-3.3-70b-instruct"},
                    {"name": "索羅斯總經獵手", "provider": "OpenRouter", "model": "meta-llama/llama-3-70b-instruct"}
                ]
                
            # --- 1.1 從 Google Drive 載入專家設定 (若已登入且尚未載入) ---
            if is_logged_in() and "expert_configs_loaded_from_drive" not in st.session_state:
                with st.spinner("⏳ 正在從 Google Drive 讀取您的專家設定..."):
                    drive_configs = load_expert_config_from_drive()
                    if drive_configs and isinstance(drive_configs, list) and len(drive_configs) == 3:
                        st.session_state["expert_configs"] = drive_configs
                        st.toast("📅 已成功從 Google Drive 套用您的專屬專家設定！", icon="📅")
                    st.session_state["expert_configs_loaded_from_drive"] = True
                
            # --- 2. 專家設定面板 ---
            with st.expander("⚙️ 專家陣容與模型配置", expanded=False):
                st.markdown("您可以配置最多 3 位專家，並為每位專家指定名稱、API 供應商及特定的 AI 模型：")
                
                cols = st.columns(3)
                updated_configs = []
                for idx in range(3):
                    with cols[idx]:
                        st.markdown(f"#### 👤 專家 {idx+1}")
                        cfg = st.session_state["expert_configs"][idx] if idx < len(st.session_state["expert_configs"]) else {"name": f"AI專家 {idx+1}", "provider": "Google", "model": ""}
                        
                        exp_name = st.text_input(f"名稱", value=cfg["name"], key=f"exp_name_{idx}")
                        provider = st.selectbox(f"API 供應商", ["Google", "Nvidia", "OpenRouter"], index=["Google", "Nvidia", "OpenRouter"].index(cfg["provider"]) if cfg["provider"] in ["Google", "Nvidia", "OpenRouter"] else 0, key=f"exp_provider_{idx}")
                        
                        # 根據 provider 動態獲取模型
                        if provider == "Google":
                            models = fetch_google_models(st.secrets.get("GEMINI_API_KEY"))
                        elif provider == "Nvidia":
                            models = fetch_nvidia_models(st.secrets.get("NVIDIA_API_KEY"))
                        else:
                            models = fetch_openrouter_models(st.secrets.get("OPENROUTER_API_KEY"))
                            
                        # 選擇模型
                        default_model = cfg["model"]
                        if default_model not in models:
                            default_model = models[0] if models else ""
                        
                        model = st.selectbox(f"模型", models, index=models.index(default_model) if default_model in models else 0, key=f"exp_model_{idx}")
                        updated_configs.append({"name": exp_name, "provider": provider, "model": model})
                
                if st.button("💾 儲存並套用專家設定"):
                    st.session_state["expert_configs"] = updated_configs
                    if is_logged_in():
                        with st.spinner("💾 正在同步設定至 Google Drive..."):
                            success = save_expert_config_to_drive(updated_configs)
                            if success:
                                st.toast("✅ 專家配置已更新並儲存至 Google Drive！", icon="💾")
                            else:
                                st.toast("⚠️ 儲存至 Google Drive 失敗，但已在本地生效。", icon="⚠️")
                    else:
                        st.toast("✅ 專家配置已在本地生效！(登入 Google 帳號可永久保存設定)", icon="💾")
                    st.rerun()
            
            # --- 3. 渲染對話紀錄 ---
            st.markdown("---")
            chat_container = st.container()
            
            with chat_container:
                if not st.session_state[f"expert_chat_history_{final_symbol}"]:
                    st.info("💡 聊天室目前空空如也。在下方輸入問題，並點選專家來開啟對話吧！")
                else:
                    for msg in st.session_state[f"expert_chat_history_{final_symbol}"]:
                        if msg["role"] == "user":
                            with st.chat_message("user", avatar="👤"):
                                st.markdown(f"**您**：{msg['content']}")
                        else:
                            expert_colors = {0: "blue", 1: "green", 2: "orange"}
                            exp_idx = 0
                            for idx, cfg in enumerate(st.session_state["expert_configs"]):
                                if cfg["name"] == msg["name"]:
                                    exp_idx = idx
                                    break
                            
                            color = expert_colors.get(exp_idx, "grey")
                            avatar_emoji = ["👨‍💼", "👩‍💼", "🕵️‍♂️"][exp_idx % 3]
                            
                            with st.chat_message("assistant", avatar=avatar_emoji):
                                st.markdown(f"### {avatar_emoji} {msg['name']} ({msg['model']})")
                                st.markdown(msg["content"])
                                
            # --- 4. 對話輸入區 ---
            st.markdown("---")
            
            # 準備系統背景 prompt，包含個股財報與技術指標數據
            tech_status = get_tech_status(df)
            status_desc = f"""
            分析股票標的：{final_symbol}
            目前最新價格：{last['Close']}
            技術面狀態：
            - 收盤價: {last['Close']:.2f}
            - 均線狀態: MA5={last['MA5']:.2f}, MA20={last['MA20']:.2f}, MA60={last['MA60']:.2f} ({tech_status.get('ma_struct', ('未知',))[0]})
            - 趨勢方向: {tech_status.get('trend', ('未知',))[0]}
            - KD指標狀態: K={last['K']:.1f}, D={last['D']:.1f} ({tech_status.get('kd', ('未知',))[0]})
            - MACD柱狀體: {last['MACD_Hist']:.4f} ({tech_status.get('macd', ('未知',))[0]})
            - 量能狀態: {tech_status.get('volume', ('未知',))[0]}
            """
            
            expert_options = [cfg["name"] for cfg in st.session_state["expert_configs"]]
            
            # 1. 專家選取區（多選）
            st.write("🗣️ **指定發言專家（可複選）**：")
            col_e1, col_e2, col_e3 = st.columns(3)
            
            if "selected_speakers" not in st.session_state:
                st.session_state["selected_speakers"] = [True, False, False]
                
            with col_e1:
                e1 = st.checkbox(f"👨‍💼 {expert_options[0]}", value=st.session_state["selected_speakers"][0], key="cb_exp_0")
            with col_e2:
                e2 = st.checkbox(f"👩‍💼 {expert_options[1]}", value=st.session_state["selected_speakers"][1], key="cb_exp_1")
            with col_e3:
                e3 = st.checkbox(f"🕵️‍♂️ {expert_options[2]}", value=st.session_state["selected_speakers"][2], key="cb_exp_2")
                
            st.session_state["selected_speakers"] = [e1, e2, e3]
            active_experts = [st.session_state["expert_configs"][i] for i, sel in enumerate(st.session_state["selected_speakers"]) if sel]
            
            # 2. 快速推薦問題按鈕帶入
            if "query_input_val" not in st.session_state:
                st.session_state["query_input_val"] = ""
                
            if f"dynamic_questions_{final_symbol}" not in st.session_state:
                st.session_state[f"dynamic_questions_{final_symbol}"] = [
                    "從技術面來看，這檔股票近期有跌破均線或支撐點的風險嗎？您會建議怎麼佈局？",
                    "根據目前的股價與基本面，您認為現在的估值合理嗎？是否有足夠的安全邊際？",
                    "請問你贊同前一位專家剛剛提出的論點與策略嗎？你有沒有看到他忽略的盲點或風險？"
                ]
                
            st.write("💡 **快速推薦問題**（點選後自動帶入輸入框）：")
            q_cols = st.columns(3)
            dyn_qs = st.session_state[f"dynamic_questions_{final_symbol}"]
            
            for col_i, q_text in enumerate(dyn_qs):
                label = q_text[:15] + "..." if len(q_text) > 15 else q_text
                if col_i == 0: label = "📉 " + label
                elif col_i == 1: label = "💰 " + label
                else: label = "🗣️ " + label
                
                with q_cols[col_i]:
                    if st.button(label, key=f"rec_btn_{col_i}", use_container_width=True):
                        # Must set BOTH the helper var AND the widget's own session_state key
                        # so that text_input renders the new value after rerun
                        st.session_state["query_input_val"] = q_text
                        st.session_state["user_question_input"] = q_text
                        st.rerun()
            
            # 3. 對話輸入文字框
            user_question = st.text_input("輸入您的問題：", value=st.session_state.get("query_input_val", ""), placeholder="請輸入問題或追問內容...", key="user_question_input")
            
            col_btn1, col_btn2, _ = st.columns([1, 1, 3])
            submit_clicked = col_btn1.button("📤 送出提問與會診", use_container_width=True)
            clear_clicked = col_btn2.button("🗑️ 清除聊天紀錄", use_container_width=True)
            
            if clear_clicked:
                st.session_state[f"expert_chat_history_{final_symbol}"] = []
                st.session_state.pop("query_input_val", None)
                st.rerun()
                
            if submit_clicked and user_question.strip():
                if not active_experts:
                    st.warning("⚠️ 請至少勾選一位發言專家！")
                else:
                    # A. 將使用者問題加入歷史
                    st.session_state[f"expert_chat_history_{final_symbol}"].append({
                        "role": "user",
                        "name": "User",
                        "content": user_question.strip(),
                        "model": "User"
                    })
                    
                    # B. 依次呼叫被選取專家的 API，使用 streaming 讓每位專家邊想邊顯示
                    for active_cfg in active_experts:
                        expert_name = active_cfg["name"]
                        
                        system_instruction = f"""
                        你現在是投資大師「{expert_name}」。你目前正在參與一場針對【{final_symbol}】的線上圓桌投資研討會。
                        你的性格與分析流派為：
                        - 巴菲特價值專員：非常注重基本面、本益比、護城河與安全邊際，語氣沉穩保守。
                        - 凱薩琳科技女皇：極度熱愛破壞性創新、AI與未來大趨勢，語氣樂觀犀利，能承受高波動。
                        - 索羅斯總經獵手：著重於全球資金流向、反射性理論、技術分析背離、量價結構與投機反轉點，語氣冷靜且投機。
                        
                        【最新股票數據】：
                        {status_desc}
                        
                        【任務說明】：
                        1. 請嚴格根據上述的流派人設，對使用者或前面專家提出的問題給出分析。
                        2. 討論串中可能包含其他專家的發言（格式為 `[專家名字] 說：...` 或類似脈絡）。你**絕對必須**看清前面的討論，並針對前面的論點進行有理有據的贊同、反駁或補充！
                        3. 請用精煉且有說服力的繁體中文回答，語氣要活生生像個獨立的專家。
                        """
                        
                        # 即時串流顯示該專家的回覆
                        expert_avatar = {"巴菲特價值專員": "🎩", "凱薩琳科技女皇": "👑", "索羅斯總經獵手": "🦅"}.get(expert_name, "🤖")
                        search_badge = "🔍 Google Search" if active_cfg["provider"] == "Google" else "📰 Yahoo Finance 新聞"
                        with st.chat_message("assistant", avatar=expert_avatar):
                            st.markdown(f"**{expert_name}** *({active_cfg['model']})* &nbsp; `{search_badge}`")
                            # write_stream 接收 generator，邊產生 token 邊渲染，並回傳完整字串
                            ai_response = st.write_stream(
                                stream_expert_chat(
                                    provider=active_cfg["provider"],
                                    model_name=active_cfg["model"],
                                    system_prompt=system_instruction,
                                    history=st.session_state[f"expert_chat_history_{final_symbol}"],
                                    symbol=final_symbol
                                )
                            )
                        
                        # 串流結束後才存入 history，供下一位專家參考
                        st.session_state[f"expert_chat_history_{final_symbol}"].append({
                            "role": "assistant",
                            "name": expert_name,
                            "content": ai_response,
                            "model": active_cfg["model"]
                        })
                    
                    # C. 更新推薦問題
                    update_dynamic_questions(
                        final_symbol,
                        st.session_state[f"expert_chat_history_{final_symbol}"],
                        status_desc
                    )
                    
                    # D. 清空輸入字串暫存
                    # 注意：不能在 widget 渲染後直接 set 其 key（會觸發 StreamlitAPIException）
                    # 改用 pop 移除 widget key，讓 st.rerun() 時 text_input 以空白初始化
                    st.session_state["query_input_val"] = ""
                    st.session_state.pop("user_question_input", None)
                    st.rerun()
