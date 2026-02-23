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

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 技術分析圖表", "🤖 AI 操盤建議", "🏛️ 基本面分析", "📰 市場情緒分析", "🗣️ AI 多空辯論"])
        
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
                        
        with tab3:
            st.markdown(f"### 🏛️ {final_symbol} 基本面與產業分析")
            st.markdown("利用 AI 結合常識與最新市場洞察，深入剖析該公司的基本面體質。")
            
            if st.button("啟動基本面分析深潛 (Deep Dive)"):
                with st.spinner("AI 正在調閱該公司的產業定位、護城河與財務特徵..."):
                    
                    # 嘗試抓取基本的公司資訊給 AI 參考 (非必須，但能提升回答品質)
                    stock_info = {}
                    try:
                        ticker_obj = yf.Ticker(final_symbol)
                        info = ticker_obj.info
                        stock_info['名稱'] = info.get('shortName', '未知')
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
                    目前已知背景資訊：{bg_info}
                    現在時間：{datetime.now().strftime("%Y-%m-%d")}
                    
                    請利用你龐大的資料庫與對全球產業鏈的理解，針對這家公司撰寫一份深入且專業的基本面分析報告。
                    
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
