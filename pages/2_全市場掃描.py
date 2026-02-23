import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. 頁面設定 ---
st.set_page_config(page_title="股票大師：智能選股雷達 (中文版)", layout="wide", page_icon="📡")
st.title("📡 股票大師：策略 2 (RSI + 200MA) 全市場掃描")

# --- 2. 核心數據處理 (內建中文對照表) ---
def get_stock_map():
    stock_map = {
        # --- 半導體/電子龍頭 ---
        "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電", 
        "2303.TW": "聯電", "3711.TW": "日月光", "2379.TW": "瑞昱", "3034.TW": "聯詠",
        "2382.TW": "廣達", "3231.TW": "緯創", "2376.TW": "技嘉", "2357.TW": "華碩",
        "2356.TW": "英業達", "2301.TW": "光寶科", "2324.TW": "仁寶", "2353.TW": "宏碁",
        "2395.TW": "研華", "3037.TW": "欣興", "3035.TW": "智原", "3661.TW": "世芯-KY",
        "3443.TW": "創意", "6669.TW": "緯穎", "2408.TW": "南亞科", "2344.TW": "華邦電",
        "3008.TW": "大立光", "2409.TW": "友達", "3481.TW": "群創", "2354.TW": "鴻準",
        "4938.TW": "和碩", "4958.TW": "臻鼎-KY", "3044.TW": "健鼎", "2368.TW": "金像電",
        
        # --- 金融股 ---
        "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2886.TW": "兆豐金",
        "2884.TW": "玉山金", "2885.TW": "元大金", "2892.TW": "第一金", "2890.TW": "永豐金",
        "2880.TW": "華南金", "2883.TW": "開發金", "5880.TW": "合庫金", "2887.TW": "台新金",
        "2801.TW": "彰銀", "5876.TW": "上海商銀", "2834.TW": "臺企銀", "2812.TW": "台中銀",
        "2867.TW": "三商壽", "2889.TW": "國票金", "2897.TW": "王道銀", "5871.TW": "中租-KY",

        # --- 傳產/航運/其他 ---
        "1101.TW": "台泥", "1102.TW": "亞泥", "2002.TW": "中鋼", "2006.TW": "東和鋼鐵",
        "6505.TW": "台塑化", "1301.TW": "台塑", "1303.TW": "南亞", "1326.TW": "台化",
        "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海", "2610.TW": "華航",
        "2618.TW": "長榮航", "2912.TW": "統一超", "1216.TW": "統一", "2207.TW": "和泰車",
        "9910.TW": "豐泰", "9904.TW": "寶成", "1402.TW": "遠東新", "1504.TW": "東元",
        "1513.TW": "中興電", "1519.TW": "華城", "1605.TW": "華新", "2542.TW": "興富發",
        "2412.TW": "中華電", "3045.TW": "台灣大", "4904.TW": "遠傳", "9914.TW": "美利達",
        "9921.TW": "巨大", "8454.TW": "富邦媒", "8046.TW": "南電", "8069.TW": "元太",
        "8299.TW": "群聯", "6269.TW": "台郡", "6239.TW": "力成", "5347.TWO": "世界",
        "5483.TWO": "中美晶", "6488.TW": "環球晶", "6770.TW": "力積電", "2449.TW": "京元電"
    }
    return stock_map

# RSI 計算函數
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- 3. 掃描引擎 ---
def scan_market(stock_map):
    results_buy = []
    results_sell = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    tickers = list(stock_map.keys())
    
    start_date = datetime.now() - timedelta(days=400)
    end_date = datetime.now() + timedelta(days=1)
    
    status_text.text(f"正在連線 Yahoo Finance 下載 {len(tickers)} 檔股票數據...")
    
    try:
        batch_size = 50
        all_data = pd.DataFrame()
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            status_text.text(f"正在下載第 {i+1} ~ {min(i+batch_size, len(tickers))} 檔... (請稍候)")
            batch_data = yf.download(batch_tickers, start=start_date, end=end_date, group_by='ticker', progress=False)
            
            if all_data.empty:
                all_data = batch_data
            else:
                all_data = pd.concat([all_data, batch_data], axis=1)

        status_text.text("數據下載完成，正在進行策略運算...")
        
        total = len(tickers)
        for i, ticker in enumerate(tickers):
            progress_bar.progress((i + 1) / total)
            
            try:
                stock_name = stock_map.get(ticker, ticker)
                
                if ticker not in all_data.columns.get_level_values(0):
                    continue
                    
                df = all_data[ticker].copy()
                
                if df.empty or len(df) < 200:
                    continue
                    
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df = df.dropna(subset=['Close'])
                
                df['MA200'] = df['Close'].rolling(200).mean()
                
                delta = df['Close'].diff()
                up = delta.clip(lower=0)
                down = -1 * delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()
                ema_down = down.ewm(com=13, adjust=False).mean()
                rs = ema_up / ema_down
                df['RSI'] = 100 - (100 / (1 + rs))
                
                last_row = df.iloc[-1]
                price = last_row['Close']
                ma200 = last_row['MA200']
                rsi = last_row['RSI']
                date_str = df.index[-1].strftime('%Y-%m-%d')
                
                clean_ticker = ticker.replace(".TW", "").replace(".TWO", "")

                # 🟢 買入條件
                if price > ma200 and rsi < 30:
                    dist_ma200 = (price - ma200) / ma200 * 100
                    results_buy.append({
                        "代碼": clean_ticker,
                        "名稱": stock_name,
                        "收盤價": f"{price:.2f}",
                        "RSI": f"{rsi:.1f} 🔥",
                        "200MA": f"{ma200:.2f}",
                        "乖離率": f"{dist_ma200:.1f}%",
                        "日期": date_str
                    })
                
                # 🟡 觀察名單
                elif price > ma200 and rsi < 40:
                     dist_ma200 = (price - ma200) / ma200 * 100
                     results_buy.append({
                        "代碼": clean_ticker,
                        "名稱": stock_name,
                        "收盤價": f"{price:.2f}",
                        "RSI": f"{rsi:.1f}",
                        "200MA": f"{ma200:.2f}",
                        "乖離率": f"{dist_ma200:.1f}%",
                        "日期": date_str
                    })

                # 🔴 賣出條件
                if rsi > 70:
                    results_sell.append({
                        "代碼": clean_ticker,
                        "名稱": stock_name,
                        "收盤價": f"{price:.2f}",
                        "RSI": f"{rsi:.1f} ⚠️",
                        "200MA": f"{ma200:.2f}",
                        "日期": date_str
                    })

            except Exception as e:
                continue

        status_text.text("全市場掃描完成！")
        return pd.DataFrame(results_buy), pd.DataFrame(results_sell)

    except Exception as e:
        st.error(f"下載失敗，可能是網路不穩，請重試。錯誤: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- 4. 主介面 ---
st.markdown("""
### 策略 2：RSI + 200MA 長線保護短線
* **掃描範圍**：台灣 50 + 中型 100 (約 150 檔熱門權值股)
* **✅ 買進條件**：股價在 **200MA (年線)** 之上，且 **RSI < 30** (或 40)。
* **❌ 賣出條件**：**RSI > 70** (短線過熱)。
""")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🚀 開始掃描全市場 (含中文名)", type="primary"):
        stock_map = get_stock_map()
        # ⚠️ 這裡改名了！df_buy_v2
        df_buy, df_sell = scan_market(stock_map)
        
        st.session_state['df_buy_v2'] = df_buy
        st.session_state['df_sell_v2'] = df_sell

# 顯示結果 (讀取新的 v2 變數)
if 'df_buy_v2' in st.session_state:
    tab1, tab2 = st.tabs(["🟢 潛力買點 (回後買上漲)", "🔴 潛力賣點 (短線過熱)"])
    
    with tab1:
        if not st.session_state['df_buy_v2'].empty:
            st.success(f"共找到 {len(st.session_state['df_buy_v2'])} 檔符合條件！")
            cols = ["代碼", "名稱", "收盤價", "RSI", "乖離率", "200MA", "日期"]
            # 這裡加入了容錯機制，萬一沒有名稱也不會報錯
            try:
                st.dataframe(st.session_state['df_buy_v2'][cols], use_container_width=True)
            except:
                st.dataframe(st.session_state['df_buy_v2'], use_container_width=True)
            st.markdown("💡 **解讀**：這些是長線多頭但短線被錯殺的股票。")
            
            st.markdown("---")
            st.subheader("🤖 AI 深度分析連動")
            col_sel, col_btn = st.columns([3, 1])
            with col_sel:
                options = st.session_state['df_buy_v2']['代碼'] + " - " + st.session_state['df_buy_v2']['名稱']
                selected_stock = st.selectbox("請選擇一檔股票進行深度分析：", options.tolist(), key="buy_select")
            with col_btn:
                st.write("") # padding
                st.write("") # padding
                if st.button("🚀 執行 AI 深度分析", key="btn_buy"):
                    ticker = selected_stock.split(" - ")[0]
                    st.session_state['auto_analyze_ticker'] = ticker
                    st.switch_page("pages/1_技術面操盤.py")
        else:
            st.info("目前沒有股票符合「長多回檔 (RSI<40)」的條件。")

    with tab2:
        if not st.session_state['df_sell_v2'].empty:
            st.warning(f"共找到 {len(st.session_state['df_sell_v2'])} 檔過熱股！")
            cols = ["代碼", "名稱", "收盤價", "RSI", "200MA", "日期"]
            try:
                st.dataframe(st.session_state['df_sell_v2'][cols], use_container_width=True)
            except:
                st.dataframe(st.session_state['df_sell_v2'], use_container_width=True)
            st.markdown("💡 **解讀**：這些股票短線過熱，請注意風險。")
            
            st.markdown("---")
            st.subheader("🤖 AI 深度分析連動")
            col_sel, col_btn = st.columns([3, 1])
            with col_sel:
                options = st.session_state['df_sell_v2']['代碼'] + " - " + st.session_state['df_sell_v2']['名稱']
                selected_stock = st.selectbox("請選擇一檔股票進行深度分析：", options.tolist(), key="sell_select")
            with col_btn:
                st.write("") # padding
                st.write("") # padding
                if st.button("🚀 執行 AI 深度分析", key="btn_sell"):
                    ticker = selected_stock.split(" - ")[0]
                    st.session_state['auto_analyze_ticker'] = ticker
                    st.switch_page("pages/1_技術面操盤.py")
        else:
            st.info("目前沒有股票 RSI > 70。")
