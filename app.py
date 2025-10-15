# --- START OF FILE app.py ---

# ==============================================================================
# 1. 导入所有需要的库
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from stqdm import stqdm
import os
import warnings
import re 
import docx
import time

# AI Agent 所需的库
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark

# 在程序最开始加载 .env 文件
load_dotenv()

# (关键) 解决 KMeans 内存泄漏警告
os.environ['OMP_NUM_THREADS'] = '1'

# 分析库
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.errors import SettingWithCopyWarning, DtypeWarning

# 抑制特定的Pandas警告，美化输出
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=DtypeWarning)

import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# 2. 配置与辅助函数
# ==============================================================================

def generate_market_report(ark_client, target_market, use_web_search):
    """
    第一轮：AI 市场分析模块
    """
    system_prompt = """
    你是一位资深的跨境电商市场分析师与选品专家。
    你的核心任务是根据用户指定的国家或地区，利用你的知识和网络搜索能力，生成一份专业、详尽、且具有前瞻性的选品分析报告。
    报告必须使用Markdown格式，结构清晰，包含标题、列表和加粗等元素。
    报告必须遵循以下结构和要求：
    ## 一、 市场机遇概要
    - 总结核心机遇，直接点出2-3个推荐产品大类。
    ## 二、 重点选品品类深度分析
    - 对每个大类，从以下四维度分析：
    1.  **市场潜力与趋势**: 结合数据解释增长空间。
    2.  **文化与消费习惯契合度**: 【核心】分析与当地文化、生活方式的结合点。
    3.  **法律法规与关税考量**: 【核心】明确指出进口限制、所需认证和关税政策。
    4.  **具体选品建议**: 给出2-3个具体的、可执行的细分产品建议。
    ## 三、 风险与挑战
    - 客观指出物流、支付、竞争等风险。
    ## 四、 总结与最终建议
    - 总结报告，给出明确的行动建议。
    你必须调用`web_search`工具确保数据最新。报告语言应专业、客观。
    """
    user_input = f"请为我生成一份关于'{target_market}'市场的跨境电商选品分析报告。"
    
    request_params = {
        "model": "doubao-seed-1-6-250615",
        "input": [{"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                  {"role": "user", "content": [{"type": "input_text", "text": user_input}]}],
        "stream": True,
        "extra_body": {"thinking": {"type": "auto"}},
    }
    if use_web_search:
        request_params["tools"] = [{"type": "web_search", "limit": 10}]

    try:
        response = ark_client.responses.create(**request_params)
        for chunk in response:
            delta_content = getattr(chunk, 'delta', None)
            if isinstance(delta_content, str): yield delta_content
    except Exception as e:
        yield f"\n\n❌ **AI Agent请求失败:**\n\n`{str(e)}`"

def generate_improvement_suggestions(ark_client, market_report, product_description, use_web_search):
    """
    第二轮：AI 产品改进建议模块
    """
    system_prompt = """
    你是一位顶级的跨境电商产品本地化战略顾问。
    你的任务是深入分析一份已有的市场报告和一份用户提供的产品描述，然后结合最新的网络搜索信息，为该产品进入目标市场提供一份专业、具体、可执行的改进建议报告。
    报告必须使用Markdown格式，并严格遵循以下结构：
    ### 📝 产品核心特性总结
    - 首先，用一两句话总结你理解的这款产品的核心功能和目标用户。
    ### 🚀 市场机遇结合点 (Opportunities)
    - 明确指出这款产品与市场报告中提到的哪些 **机遇** 和 **文化消费习惯** 高度契合，这是产品的核心优势。
    ### ⚠️ 潜在风险与挑战 (Challenges)
    - 明确指出这款产品可能会触碰到市场报告中提到的哪些 **风险**、**法律法规** 或 **认证要求**，这是需要优先解决的问题。
    ### 💡 具体改进建议 (Actionable Suggestions)
    - 这是报告的核心。提供一个包含具体建议的列表，至少覆盖以下3-4个方面：
        - **功能与设计调整**: 建议对产品的颜色、尺寸、功能点、包装设计等进行哪些调整以更符合当地审美和使用习惯。
        - **营销语言与卖点提炼**: 建议在产品详情页和广告中使用哪些关键词和营销角度，以精准触达当地消费者。
        - **定价与服务策略**: 建议一个初步的定价区间，并指出是否需要提供特殊的支付方式（如COD）或售后服务。
        - **合规性检查**: 提醒用户需要检查或获取哪些具体的认证或许可以及早准备。
    """
    
    user_input = f"""
    这是我之前生成的市场分析报告：
    --- [市场报告开始] ---
    {market_report}
    --- [市场报告结束] ---

    这是我的产品介绍：
    --- [产品介绍开始] ---
    {product_description}
    --- [产品介绍结束] ---

    请根据以上信息，为我的产品生成一份详细的本地化改进建议报告。
    """

    request_params = {
        "model": "doubao-seed-1-6-250615",
        "input": [{"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                  {"role": "user", "content": [{"type": "input_text", "text": user_input}]}],
        "stream": True,
        "extra_body": {"thinking": {"type": "auto"}},
    }
    if use_web_search:
        request_params["tools"] = [{"type": "web_search", "limit": 5}]

    try:
        response = ark_client.responses.create(**request_params)
        for chunk in response:
            delta_content = getattr(chunk, 'delta', None)
            if isinstance(delta_content, str): yield delta_content
    except Exception as e:
        yield f"\n\n❌ **AI Agent请求失败:**\n\n`{str(e)}`"

def generate_review_summary_report(ark_client, positive_reviews_sample, negative_reviews_sample):
    """
    新增：AI 评论分析模块
    """
    system_prompt = """
    你是一位高级用户洞察分析师，专注于从大量用户评论中提炼核心观点和商业洞见。
    你的任务是分析给你的正面和负面评论样本，并生成一份简洁、深刻、结构化的分析报告。
    报告必须使用Markdown格式，并严格遵循以下结构：
    ### 📝 评论总体情绪概述
    - 用一两句话，基于你看到的评论，总结产品的整体市场反响和用户情绪。
    ### 👍 产品核心优势 (从正面评论中提炼)
    - 使用列表形式，总结出用户最常称赞的2-3个核心优点。
    - 每一个优点后面，用括号引用一句最能代表该观点的 **原始评论**。
    ### 👎 产品主要痛点 (从负面评论中提炼)
    - 使用列表形式，总结出用户抱怨最多的2-3个核心问题或缺点。
    - 每一个痛点后面，用括号引用一句最能代表该观点的 **原始评论**。
    ### 💡 可执行的改进建议 (Actionable Suggestions)
    - 基于以上分析，为产品经理或运营团队提供2-3条具体的、可执行的改进建议。
    """
    
    user_input = f"""
    以下是关于某款产品的用户评论样本。
    --- [正面评论样本] ---
    {positive_reviews_sample}
    --- [正面评论样本结束] ---
    --- [负面评论样本] ---
    {negative_reviews_sample}
    --- [负面评论样本结束] ---
    请根据以上评论，为我生成一份用户洞察分析报告。
    """

    # --- 【代码风格统一修正】 ---
    request_params = {
        "model": "doubao-seed-1-6-250615",
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_input}]}
        ],
        "stream": True,
        "extra_body": {"thinking": {"type": "auto"}},
    }

    try:
        response = ark_client.responses.create(**request_params)
        for chunk in response:
            delta_content = getattr(chunk, 'delta', None)
            if isinstance(delta_content, str):
                yield delta_content
    except Exception as e:
        yield f"\n\n❌ **AI Agent请求失败:**\n\n`{str(e)}`"

@st.cache_data
def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8-sig')

def load_uploaded_file(uploaded_file, dtype_spec=None):
    if uploaded_file is None: return None;
    try:
        file_name = uploaded_file.name
        if file_name.endswith('.parquet'): return pd.read_parquet(uploaded_file)
        elif file_name.endswith('.csv'): return pd.read_csv(uploaded_file, on_bad_lines='skip', dtype=dtype_spec)
        else: st.error(f"不支持的文件类型: {file_name}。"); return None
    except Exception as e: st.error(f"读取文件 '{uploaded_file.name}' 时出错: {e}"); return None

@st.cache_data
def perform_lstm_forecast(_df):
    sales_ts = _df.groupby('Date')['Amount'].sum().asfreq('D', fill_value=0); sales_values = sales_ts.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1)); scaled_values = scaler.fit_transform(sales_values)
    def create_dataset(data, look_back=7):
        X, y = [], [];
        for i in range(len(data) - look_back): X.append(data[i:(i + look_back), 0]); y.append(data[i + look_back, 0])
        return np.array(X), np.array(y)
    look_back = 7; X, y = create_dataset(scaled_values, look_back); X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential([Input(shape=(look_back, 1)), LSTM(50), Dense(1)]); model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    last_days_scaled = scaled_values[-look_back:]; current_input = np.reshape(last_days_scaled, (1, look_back, 1)); future_predictions_scaled = []
    for _ in range(30):
        next_pred_scaled = model.predict(current_input, verbose=0); future_predictions_scaled.append(next_pred_scaled[0, 0])
        new_pred_reshaped = np.reshape(next_pred_scaled, (1, 1, 1)); current_input = np.append(current_input[:, 1:, :], new_pred_reshaped, axis=1)
    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    last_date = sales_ts.index[-1]; future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30); fig = go.Figure()
    fig.add_trace(go.Scatter(x=sales_ts.index, y=sales_ts.values, name='历史销售额', line=dict(color='royalblue', width=2), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.2)'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), name='LSTM 预测销售额', line=dict(color='darkorange', dash='dash', width=2), fill='tozeroy', fillcolor='rgba(255, 140, 0, 0.2)'))
    fig.update_layout(title='未来30天销售额深度学习预测 (LSTM模型)', xaxis_title='日期', yaxis_title='销售额'); return fig

@st.cache_data
def perform_product_clustering(_df):
    required_cols = ['SKU', 'Amount', 'Qty', 'Order ID'];
    if not all(col in _df.columns for col in _df.columns): return None, None, f"聚类分析失败：缺少必要的列。"
    product_agg_df = _df.groupby('SKU').agg(total_amount=('Amount', 'sum'), total_qty=('Qty', 'sum'), order_count=('Order ID', 'nunique')).reset_index()
    features = product_agg_df[['total_amount', 'total_qty', 'order_count']]; scaler = StandardScaler(); features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42); product_agg_df['cluster'] = kmeans.fit_predict(features_scaled)
    cluster_summary = product_agg_df.groupby('cluster')[['total_amount', 'total_qty', 'order_count']].mean().sort_values(by='total_amount', ascending=False)
    hot_product_cluster_id = cluster_summary.index[0]
    hot_products = product_agg_df[product_agg_df['cluster'] == hot_product_cluster_id].sort_values(by='total_amount', ascending=False)
    return cluster_summary, hot_products, None

def find_review_column(df):
    priority_cols = ['reviews.text', 'review_text', 'content', 'comment', 'review']
    for p_col in priority_cols:
        if p_col in df.columns and df[p_col].dropna().astype(str).str.strip().any(): return p_col
    possible_cols = [col for col in df.columns if any(key in str(col).lower() for key in ['text', 'review', 'content', 'comment'])]
    if possible_cols:
        string_cols = [col for col in possible_cols if df[col].dtype == 'object'];
        if string_cols: return max(string_cols, key=lambda col: df[col].dropna().astype(str).str.len().mean())
    object_cols = df.select_dtypes(include=['object']).columns
    if not object_cols.empty:
        for col in object_cols:
            if df[col].dropna().astype(str).str.strip().any(): return col
    return None
    
@st.cache_data
def perform_sentiment_analysis(reviews_df):
    def sentiment_to_rating(sentiment):
        if sentiment >= 0.5: return 5
        elif sentiment >= 0.05: return 4
        elif sentiment > -0.05: return 3
        elif sentiment > -0.5: return 2
        else: return 1
    analyzer = SentimentIntensityAnalyzer(); review_column_name = find_review_column(reviews_df);
    if review_column_name is None: return None, "错误: 未能在评论文件中找到有效的文本列。"
    reviews_df[review_column_name] = reviews_df[review_column_name].astype(str).dropna()
    reviews_df = reviews_df[reviews_df[review_column_name].str.strip() != 'None'].copy()
    stqdm.pandas(desc="正在计算情感分数"); reviews_df['sentiment'] = reviews_df[review_column_name].progress_apply(lambda text: analyzer.polarity_scores(text)['compound'])
    if 'rating' not in reviews_df.columns: reviews_df['rating'] = reviews_df['sentiment'].apply(sentiment_to_rating)
    reviews_df.rename(columns={review_column_name: 'review_text'}, inplace=True); return reviews_df, None

@st.cache_data
def create_category_sales_plot(_df):
    category_means = _df.groupby('Category')['Amount'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(category_means, x='Category', y='Amount', color='Category', text_auto='.2f', labels={'Category': '产品类别', 'Amount': '平均销售额 (Amount)'}, title='各产品类别平均销售额对比')
    fig.update_layout(width=800, height=500, template='plotly_white', showlegend=False)
    fig.update_traces(textposition='outside')
    return fig

# ==============================================================================
# 3. Streamlit 用户界面布局
# ==============================================================================
st.set_page_config(layout="wide"); st.title('📈 WeaveAI智能分析助手')
st.caption('##### 还在凭感觉选品？让数据与AI为您引航')

with st.sidebar:
    st.header("📂 上传您的数据"); st.info("推荐您将大的CSV文件转换为Parquet格式以提升速度。")
    uploaded_amazon = st.file_uploader('1. 上传 Amazon 销售报告', type=['csv', 'parquet'])
    uploaded_reviews = st.file_uploader('2. 上传 Amazon 评论数据 (可选)', type=['csv', 'parquet'])

if 'report_generated' not in st.session_state: st.session_state.report_generated = False
if 'market_report_content' not in st.session_state: st.session_state.market_report_content = ""

tabs = ["🤖 AI 市场选品顾问", "🧠 LSTM销售预测", "🛍️ 品类表现", "🔥 热销品聚类", "💬 情感分析"]
tab_agent, tab_lstm, tab_category, tab_cluster, tab_sentiment = st.tabs(tabs)

with tab_agent:
    st.header("第一轮：AI 市场选品顾问"); 
    col1, col2 = st.columns([3, 2])
    with col1:
        target_market = st.text_input("输入您想分析的目标国家或地区", placeholder="例如：德国、东南亚、巴西")
    with col2:
        use_weaveai_db = st.toggle("调用'WeaveAI'实时数据库", value=True, help="开启后，AI将使用实时更新的最新数据库以提供更具时效性的分析。关闭则仅依赖模型自身知识。")

    if st.button("生成分析报告", key="generate_report"):
        st.session_state.report_generated = False; st.session_state.market_report_content = ""
        api_key_from_env = os.environ.get("ARK_API_KEY")
        if not api_key_from_env: st.error("无法找到 API Key。请确保 .env 文件配置正确。")
        elif not target_market.strip(): st.warning("请输入目标国家或地区。")
        else:
            st.markdown("---"); expander = st.expander("点击查看AI的思考过程 🧠", expanded=False); thinking_placeholder = expander.empty()
            st.markdown(f"### 🌍 {target_market}市场跨境电商选品分析报告"); report_placeholder = st.empty()
            full_response_text = ""; separator_pattern = re.compile(r"\n#+ ")
            try:
                ark_client = Ark(api_key=api_key_from_env)
                for chunk in generate_market_report(ark_client, target_market, use_weaveai_db):
                    full_response_text += chunk; match = separator_pattern.search(full_response_text)
                    if match:
                        thinking_part = full_response_text[:match.start()]; report_part = full_response_text[match.start():]
                        thinking_placeholder.markdown(thinking_part); report_placeholder.markdown(report_part)
                    else:
                        thinking_placeholder.markdown(full_response_text)
                match = separator_pattern.search(full_response_text)
                if match: st.session_state.market_report_content = full_response_text[match.start():]
                else: st.session_state.market_report_content = full_response_text
                st.session_state.report_generated = True
            except Exception as e: st.error(f"客户端初始化或请求失败: {e}")

    if st.session_state.report_generated:
        st.markdown("---"); st.header("第二轮：产品本地化改进建议")
        st.info("您已了解市场概况，现在可以上传您的产品介绍（.txt, .md, .docx），AI将结合市场报告为您提供改进建议。")
        uploaded_product_doc = st.file_uploader("上传您的产品介绍文档", type=['txt', 'md', 'docx'])
        if uploaded_product_doc is not None:
            if st.button("获取改进建议", key="generate_suggestions"):
                api_key_from_env = os.environ.get("ARK_API_KEY")
                if not api_key_from_env: st.error("无法找到 API Key 用于第二轮分析。")
                else:
                    with st.spinner("🚀 AI正在结合市场报告和您的产品信息，生成改进建议..."):
                        product_description = "";
                        try:
                            if uploaded_product_doc.name.endswith('.docx'):
                                doc = docx.Document(uploaded_product_doc); product_description = '\n'.join([para.text for para in doc.paragraphs])
                            else: product_description = uploaded_product_doc.read().decode("utf-8")
                        except Exception as e: st.error(f"读取文件失败: {e}")
                        if product_description:
                            st.markdown("---"); sugg_expander = st.expander("点击查看第二轮AI的思考过程 🧠", expanded=False)
                            sugg_thinking_placeholder = sugg_expander.empty(); st.markdown("#### 💡 AI 产品改进建议"); sugg_report_placeholder = st.empty()
                            sugg_full_response_text = ""; sugg_separator_pattern = re.compile(r"\n#+ ")
                            try:
                                ark_client = Ark(api_key=api_key_from_env)
                                for chunk in generate_improvement_suggestions(ark_client, st.session_state.market_report_content, product_description, use_weaveai_db):
                                    sugg_full_response_text += chunk; sugg_match = sugg_separator_pattern.search(sugg_full_response_text)
                                    if sugg_match:
                                        sugg_thinking_part = sugg_full_response_text[:sugg_match.start()]; sugg_report_part = sugg_full_response_text[sugg_match.start():]
                                        sugg_thinking_placeholder.markdown(sugg_thinking_part); sugg_report_placeholder.markdown(sugg_report_part)
                                    else:
                                        sugg_thinking_placeholder.markdown(sugg_full_response_text)
                            except Exception as e: st.error(f"生成改进建议时出错: {e}")

data_analysis_container = st.container()
with data_analysis_container:
    if uploaded_amazon:
        dtype_spec = {'ASIN': str, 'asin': str}; amazon_df = load_uploaded_file(uploaded_amazon, dtype_spec=dtype_spec)
        reviews_df_loaded = load_uploaded_file(uploaded_reviews)
        if amazon_df is not None:
            with st.status("⚙️ 正在清洗和适配您的销售数据...", expanded=True) as status:
                for old, new in {'Total Sales':'Amount','Product':'SKU','Quantity':'Qty','Order_ID':'Order ID'}.items():
                    if old in amazon_df.columns: amazon_df.rename(columns={old:new}, inplace=True)
                req_cols = ["Amount","Category","Date","Status","SKU","Order ID","Qty"]; missing = [c for c in req_cols if c not in amazon_df.columns]
                if missing: status.update(label="数据清洗失败!", state="error"); st.error(f"Amazon 文件中缺少关键列: {', '.join(missing)}")
                else:
                    amazon_df.dropna(subset=["Amount", "Category", "Date"], inplace=True)
                    try: amazon_df["Date"] = pd.to_datetime(amazon_df["Date"], format='%m-%d-%y')
                    except ValueError: amazon_df["Date"] = pd.to_datetime(amazon_df["Date"], errors='coerce')
                    amazon_df["Amount"] = pd.to_numeric(amazon_df["Amount"], errors='coerce')
                    amazon_df = amazon_df[amazon_df["Status"].isin(["Shipped","Shipped - Delivered to Buyer","Completed","Pending","Cancelled"])]
                    amazon_df.dropna(subset=['Date','Amount','SKU','Order ID','Qty'], inplace=True)
                    status.update(label="数据清洗与适配完成!", state="complete", expanded=False)
                    st.write("⏳ 正在为您计算历史销售数据分析模型...");
                    total_steps = 2 if reviews_df_loaded is not None else 1
                    progress_bar = st.progress(0, text=f"开始计算... (0/{total_steps})")
                    cluster_summary, hot_products, cluster_error = perform_product_clustering(amazon_df)
                    progress_bar.progress(100 if total_steps == 1 else 50, text=f"产品聚类分析完成... (1/{total_steps})")
                    sentiment_df, sentiment_error = (None, None)
                    if reviews_df_loaded is not None:
                        sentiment_df, sentiment_error = perform_sentiment_analysis(reviews_df_loaded)
                        progress_bar.progress(100, text=f"情感分析完成... (2/{total_steps})")
                    time.sleep(0.5); progress_bar.empty()
                    st.success("✅ 核心计算完成！现在您可以浏览所有分析结果。")
                    with tab_lstm: st.header("销售额深度学习预测 (LSTM)"); lstm_fig = perform_lstm_forecast(amazon_df); st.plotly_chart(lstm_fig, use_container_width=True)
                    with tab_category: st.header("产品类别销售表现"); cat_fig = create_category_sales_plot(amazon_df); st.plotly_chart(cat_fig)
                    with tab_cluster:
                        st.header("热销商品聚类分析")
                        if cluster_error: st.error(cluster_error)
                        elif cluster_summary is not None and hot_products is not None:
                            st.subheader("各商品簇特征均值"); st.dataframe(cluster_summary)
                            st.subheader(f"🔥 热销商品列表 (共 {len(hot_products)} 个)"); st.dataframe(hot_products.head(20));
                            if st.checkbox('显示所有热销商品'): st.dataframe(hot_products)
                            st.download_button("下载热销商品列表 (CSV)", convert_df_to_csv(hot_products), "hot_products.csv")
                    with tab_sentiment:
                        st.header("客户评论情感分析")
                        if uploaded_reviews:
                            if sentiment_error: st.error(sentiment_error)
                            elif sentiment_df is not None:
                                st.subheader("按星级筛选评论"); rating_range = st.slider('选择星级范围:',1,5,(4,5)); min_r, max_r = rating_range
                                filtered_reviews = sentiment_df[(sentiment_df['rating']>=min_r)&(sentiment_df['rating']<=max_r)]
                                st.markdown(f"**显示 {len(filtered_reviews)} 条评分为 {min_r} 到 {max_r} 星的评论**"); st.dataframe(filtered_reviews[['rating','review_text','sentiment']])
                                st.subheader("情感分数统计"); avg_filtered = filtered_reviews['sentiment'].mean() if not filtered_reviews.empty else 0; avg_all = sentiment_df['sentiment'].mean()
                                c1,c2 = st.columns(2); c1.metric(f"所选评论 ({min_r}-{max_r} 星) 的平均情感分", f"{avg_filtered:.2f}"); c2.metric("所有评论的平均情感分", f"{avg_all:.2f}")
                                
                                st.markdown("---")
                                st.subheader("🤖 AI 评论深度分析报告")
                                if st.button("生成 AI 分析报告", key="generate_review_report"):
                                    api_key_from_env = os.environ.get("ARK_API_KEY")
                                    if not api_key_from_env: st.error("无法找到 API Key。请确保 .env 文件配置正确。")
                                    elif filtered_reviews.empty: st.warning("当前筛选范围内没有评论可供AI分析，请调整上方的星级滑块。")
                                    else:
                                        with st.spinner("🚀 正在为您抽样评论并调用AI进行分析..."):
                                            positive_samples = filtered_reviews.nlargest(20, 'sentiment')
                                            negative_samples = filtered_reviews.nsmallest(20, 'sentiment')
                                            positive_text = "\n".join([f"- {row['review_text']}" for index, row in positive_samples.iterrows()])
                                            negative_text = "\n".join([f"- {row['review_text']}" for index, row in negative_samples.iterrows()])
                                            
                                            rev_expander = st.expander("点击查看AI的思考过程 🧠", expanded=False)
                                            rev_thinking_placeholder = rev_expander.empty(); rev_report_placeholder = st.empty()
                                            rev_full_response_text = ""; rev_separator_pattern = re.compile(r"\n#+ ")
                                            try:
                                                ark_client = Ark(api_key=api_key_from_env)
                                                for chunk in generate_review_summary_report(ark_client, positive_text, negative_text):
                                                    rev_full_response_text += chunk
                                                    rev_match = rev_separator_pattern.search(rev_full_response_text)
                                                    if rev_match:
                                                        rev_thinking_part = rev_full_response_text[:rev_match.start()]
                                                        rev_report_part = rev_full_response_text[rev_match.start():]
                                                        rev_thinking_placeholder.markdown(rev_thinking_part)
                                                        rev_report_placeholder.markdown(rev_report_part)
                                                    else:
                                                        rev_thinking_placeholder.markdown(rev_full_response_text)
                                            except Exception as e: st.error(f"生成评论分析报告时出错: {e}")
                        else: st.info("请在左侧上传评论文件以启用此分析。")
    else:
        with tab_lstm: st.info("👋 欢迎使用！请在左侧边栏上传您的 Amazon 销售报告以启用数据分析功能。")
        with tab_category: st.info("👋 欢迎使用！请在左侧边栏上传您的 Amazon 销售报告以启用数据分析功能。")
        with tab_cluster: st.info("👋 欢迎使用！请在左侧边栏上传您的 Amazon 销售报告以启用数据分析功能。")
        with tab_sentiment: st.info("👋 欢迎使用！请在左侧边栏上传您的 Amazon 销售报告和评论文件以启用数据分析功能。")