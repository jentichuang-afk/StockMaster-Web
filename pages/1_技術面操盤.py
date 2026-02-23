import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google import genai
from groq import Groq

# --- 1. 頁面設定 ---
st.set_page_config(page_title="股票大師：純技術戰情室", layout="wide", page_icon="📈")
st.title("📈 股票大師：純技術面操盤 (Technical Only)")

# --- 安全性設定 ---
# API keys are fetched dynamically when `call_ai` is executed


# --- 2. 側邊欄 ---
st.sidebar.header("⚙️ 參數設定")

# 處理跨頁面連動邏輯
auto_run = False
default_ticker = "2027"
if 'auto_analyze_ticker' in st.session_state and st.session_state['auto_analyze_ticker'] is not None:
    default_ticker = st.session_state['auto_analyze_ticker']
    auto_run = True
    # 讀取後馬上清除，避免下次進入頁面又重複觸發
    st.session_state['auto_analyze_ticker'] = None

ticker_input = st.sidebar.text_input("輸入股票代碼", value=default_ticker, help="台股請輸入如 2330, 8155")
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

# --- 5. AI Prompt ---
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
            # Utilizing the recommended gemini-3-flash-preview model
            response = client.models.generate_content(
                model='gemini-3-flash-preview',
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

# --- 6. 主程式 ---
if run_btn and ticker_input:
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

        tab1, tab2 = st.tabs(["📈 技術分析圖表", "🤖 AI 操盤建議"])
        
        with tab1:
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

        with tab2:
            # 🛠️ 關鍵修正：將 'MA60' 加入到要傳給 AI 的字串中
            target_cols = ['Close', 'MA5', 'MA20', 'MA60', 'K', 'D', 'MACD', 'MACD_Hist', 'OBV']
            tech_data_str = df.tail(5)[target_cols].to_string()
            
            prompt = get_prompt(final_symbol, last['Close'], tech_data_str)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔵 Gemini")
                with st.spinner("Gemini 思考中..."):
                    result = call_ai('gemini', prompt)
                    if "未設定" in result or "錯誤" in result:
                        st.error(result)
                    else:
                        st.info(result)
            
            with col2:
                st.markdown("### 🟠 Llama 3")
                with st.spinner("Llama 思考中..."):
                    result = call_ai('groq', prompt)
                    if "未設定" in result or "錯誤" in result:
                        st.error(result)
                    else:
                        st.warning(result)
