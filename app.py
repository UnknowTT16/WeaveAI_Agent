# ==============================================================================
# WeaveAI 智能分析助手 v2.8 - 最终版：全面流式输出与提示词终极优化
# ==============================================================================

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
# 2. 状态管理：初始化应用的“大脑” (Session State)
# ==============================================================================
def initialize_session_state():
    if 'profile_created' not in st.session_state: st.session_state.profile_created = False
    if 'profile' not in st.session_state: st.session_state.profile = {}
    if 'current_step' not in st.session_state: st.session_state.current_step = '机会洞察'
    if 'market_report_content' not in st.session_state: st.session_state.market_report_content = ""
    if 'report_generating' not in st.session_state: st.session_state.report_generating = False
    if 'validation_data' not in st.session_state: st.session_state.validation_data = {'amazon_df': None, 'reviews_df': None}
    if 'validation_summary' not in st.session_state: st.session_state.validation_summary = None

# ==============================================================================
# 3. AI Agent 模块 (在实际项目中, 这部分应独立为 ai_agents.py)
# ==============================================================================
def generate_full_report_stream(ark_client, user_profile):
    """【核心优化】使用优化后的高级System Prompt"""
    market = user_profile['target_market']
    categories = user_profile['supply_chain']
    seller = user_profile['seller_type']
    price_range = f"${user_profile['min_price']} - ${user_profile['max_price']}"

    system_prompt = f"""
    你是 "WeaveAI" 应用内的一位高级战略顾问，你的报告是为一位计划进入'{market}'市场的'{seller}'，他/她专注于'{categories}'品类，目标售价在'{price_range}'。你的报告必须专业、详尽、具有前瞻性、数据驱动，
    并使用精美的Markdown格式，结构清晰，包含标题、列表和加粗等元素。
    报告必须遵循以下结构和要求：

    **第一阶段：输出思考过程**
    在正式开始报告前，你必须先输出你的思考过程。这部分内容必须以 "我需要..." 或 "首先..." 开始，概述你将如何为用户分析。不要使用任何Markdown标题。
    
    **重要指令 1：** 在思考过程结束后，你必须另起一行，并只输出 `<<<<THINKING_ENDS>>>>` 这个特殊标记。
    
    **重要指令 2：** 在上一个标记之后，你必须立即另起一行并输出 `<<<<REPORT_STARTS>>>>`，然后才能开始生成严格按照以下Markdown格式的正式报告，中间不能有任何其他文字。

    **第二阶段：输出正式报告**
    ---
    
    ## 报告摘要 (Executive Summary)
    *   在此处用2-3个要点，**加粗**核心关键词，高度概括整个报告的核心发现和最终建议。
    
    ---
    
    ## 🎯 市场机遇洞察 (Market Opportunities)
    
    ### 一、 宏观环境分析
    1.  **市场潜力与趋势**: 结合**量化数据**解释增长空间 (必须注明来源和年份)。
    2.  **文化与消费习惯**: 【核心】深入分析当地文化、节假日、生活方式如何影响'{categories}'品类的消费偏好。
    3.  **法律法规与关税**: 【核心】明确指出进口限制、所需**具体认证** (如CE, RoHS) 和大致的关税税率。
    
    ### 二、 高潜力细分品类机会点
    *   你必须利用网络搜索，识别出2个最符合用户画像的细分机会。
    *   对于每一个机会点，必须严格按照以下模板进行分析：
    
    #### 机会点 1: [在此处填写具体品类名称]
    *   **产品定义:** 清晰描述这个品类的核心功能、形态和目标用户。
    *   **需求驱动与市场规模:** 解释为什么当地市场需要这个产品。**必须包含量化数据，并注明来源和年份** (例如: 市场规模预计在2025年达到 **€5000万**，年增长率 **15%** [来源: Statista, 2023])。
    *   **SWOT 分析:**
        *   **优势 (Strength):** 
        *   **劣势 (Weakness):** 
        *   **机会 (Opportunity):** 
        *   **威胁 (Threat):** 

    #### 机会点 2: [在此处填写具体品类名称]
    *   (同上结构)

    ---
    
    ## ⚔️ 核心竞争格局 (Competitive Landscape)
    
    *   针对上面识别出的每一个机会点，进行独立的竞争分析。
    
    ### 竞争分析: [机会点1的品类名称]
  *   **主要竞争对手分析:** 你的表格必须严格遵守Markdown语法，**并且表格本身必须另起新的一行开始**，其前后不能有任何文字。请参考以下完美范例：
    
*主要竞争对手分析表*
| 代表性竞品品牌 | 主流定价 | 核心卖点 | 主要用户痛点 |
| :--- | :--- | :--- | :--- |
| Anker | €45-€60 | GaN技术, 多口快充 | 部分型号体积较大 |
| Belkin | €50-€75 | 苹果官方认证, 设计简约 | 性价比不高 |
| TP-Link | €40-€60 | 稳定传输, 多设备兼容 | 电池续航一般 |
-根据实际情况补充更多竞品行。
-  **用户评价洞察:** 基于用户评论，总结出2-3个用户最关注的核心需求和痛点。

    *   **竞争策略建议:** 基于以上分析，提出1-2条针对性的、可操作的竞争策略建议。

    ### 竞争分析: [机会点2的品类名称]
    *   (同上结构)

    ---

    ## 📜 数据来源与说明
    *   **数据来源:** 在此列出本报告引用的主要数据来源网站或报告名称。
    *   **免责声明:** 本报告基于公开数据和AI模型分析生成，仅供决策参考，不构成最终投资建议。具体数据请以官方发布为准。
    """
    user_input = f"请基于我的画像，为我生成一份关于'{market}'市场的机会识别与竞争分析报告，重点关注'{categories}'品类。"

    try:
        request_params = {
            "model": "doubao-seed-1-6-250615",
            "input": [{"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                      {"role": "user", "content": [{"type": "input_text", "text": user_input}]}],
            "tools": [{"type": "web_search", "limit": 15}],
            "stream": True
        }
        response = ark_client.responses.create(**request_params)
        for chunk in response:
            delta_content = getattr(chunk, 'delta', None)
            if isinstance(delta_content, str):
                yield delta_content
    except Exception as e:
        yield f"❌ AI Agent请求失败: {e}"

def agent_action_planner(ark_client, market_report, validation_summary):
    """【优化】Agent 3: 行动规划师 - 增加流式思考过程"""
    system_prompt = f"""
    你是 "WeaveAI" 应用内的一位经验丰富的电商战略顾问，擅长将分析转化为清晰、可执行的行动计划。你的报告必须专业、结构化，并使用精美的Markdown格式。

    **第一阶段：输出思考过程**
    在正式开始报告前，你必须先输出你的思考过程。这部分内容必须以 "我需要..." 或 "首先..." 开始，概述你将如何整合市场报告和内部数据，并制定行动计划。不要使用任何Markdown标题。
    
    **重要指令 1：** 在思考过程结束后，你必须另起一行，并只输出 `<<<<THINKING_ENDS>>>>` 这个特殊标记。
    
    **重要指令 2：** 在上一个标记之后，你必须立即另起一行并输出 `<<<<REPORT_STARTS>>>>`，然后才能开始生成严格按照以下Markdown格式的正式报告，中间不能有任何其他文字。

    **第二阶段：输出正式报告**
    ---

    ## 📋 您的专属行动计划

    基于市场机会洞察与内部数据验证，我们为您制定了以下三阶段行动路线图：

    ### 🚀 产品开发与优化 (Product)
    *   **核心目标:** [此处填写产品层面的核心目标]
    *   **行动项:**
        1.  **[行动项1]:** [详细描述，将关键动词和指标 **加粗**]。
        2.  **[行动项2]:** [详细描述]。
        - 根据实际情况补充更多行动项。

    ---

    ### 📢 营销与推广 (Marketing)
    *   **核心目标:** [此处填写营销层面的核心目标]
    *   **行动项:**
        1.  **[行动项1]:** [详细描述，将关键渠道和活动 **加粗**]。
        2.  **[行动项2]:** [详细描述]。
        - 根据实际情况补充更多行动项。
    ---

    ### 📦 库存与运营 (Operations)
    *   **核心目标:** [此处填写运营层面的核心目标]
    *   **行动项:**
        1.  **[行动项1]:** [详细描述，将关键流程和认证 **加粗**]。
        2.  **[行动项2]:** [详细描述]。
        - 根据实际情况补充更多行动项。
    """
    user_input = f"以下是我的决策依据：\n--- [市场机会报告] ---\n{market_report}\n--- [内部数据验证摘要] ---\n{validation_summary}\n---\n请基于以上信息，为我生成一份具体的行动计划。"
    try:
        request_params = {"model": "doubao-seed-1-6-250615", "input": [{"role": "system", "content": [{"type": "input_text", "text": system_prompt}]}, {"role": "user", "content": [{"type": "input_text", "text": user_input}]}], "stream": True}
        response = ark_client.responses.create(**request_params)
        for chunk in response:
            delta_content = getattr(chunk, 'delta', None)
            if isinstance(delta_content, str):
                yield delta_content
    except Exception as e:
        yield f"❌ 行动规划师Agent请求失败: {e}"

def generate_review_summary_report(ark_client, positive_reviews_sample, negative_reviews_sample):
    """【优化】Agent 4: 用户洞察分析师 - 增加流式思考过程"""
    system_prompt = f"""
    你是一位 "WeaveAI" 应用内的高级用户洞察分析师，专注于从用户评论中提炼商业洞见。你的报告必须专业、结构化，并使用精美的Markdown格式。

    **第一阶段：输出思考过程**
    在正式开始报告前，你必须先输出你的思考过程。这部分内容必须以 "我需要..." 或 "首先..." 开始，概述你将如何分析这些评论。不要使用任何Markdown标题。
    
    **重要指令 1：** 在思考过程结束后，你必须另起一行，并只输出 `<<<<THINKING_ENDS>>>>` 这个特殊标记。
    
    **重要指令 2：** 在上一个标记之后，你必须立即另起一行并输出 `<<<<REPORT_STARTS>>>>`，然后才能开始生成严格按照以下Markdown格式的正式报告，中间不能有任何其他文字。

    **第二阶段：输出正式报告**
    ---

    ### 📝 评论总体情绪概述
    *   基于你看到的评论，用一两句话总结产品的整体市场反响和用户情绪。

    ---

    ### 👍 产品核心优势 (从正面评论中提炼)
    *   使用列表形式，总结出用户最常称赞的2-3个核心优点。
    *   对于每个优点，请使用 blockquote 格式引用一句最能代表该观点的 **原始评论**。
    
        > 示例：这是一个引用的原始评论。

    ---

    ### 👎 产品主要痛点 (从负面评论中提炼)
    *   使用列表形式，总结出用户抱怨最多的2-3个核心问题或缺点。
    *   对于每个痛点，同样使用 blockquote 格式引用一句最能代表该观点的 **原始评论**。

    ---
    
    ### 💡 可执行的改进建议
    *   基于以上分析，为产品或运营团队提供2-3条具体的、可操作的改进建议。每个建议都应该清晰地说明 **“做什么”** 和 **“为什么”**。
    """
    user_input = f"以下是关于某款产品的用户评论样本。\n--- [正面评论样本] ---\n{positive_reviews_sample}\n--- [正面评论样本结束] ---\n--- [负面评论样本] ---\n{negative_reviews_sample}\n--- [负面评论样本结束] ---\n请根据以上评论，为我生成一份用户洞察分析报告。"
    try:
        request_params = {"model": "doubao-seed-1-6-250615", "input": [{"role": "system", "content": [{"type": "input_text", "text": system_prompt}]}, {"role": "user", "content": [{"type": "input_text", "text": user_input}]}], "stream": True}
        response = ark_client.responses.create(**request_params)
        for chunk in response:
            delta_content = getattr(chunk, 'delta', None)
            if isinstance(delta_content, str):
                yield delta_content
    except Exception as e:
        yield f"❌ AI评论分析请求失败: {e}"

# ==============================================================================
# 4. 数据处理模块 (保持不变)
# ==============================================================================
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
# 5. UI 渲染函数 (将每个页面的UI逻辑封装)
# ==============================================================================
def render_profile_creation_page():
    """渲染战略档案创建页面"""
    st.header("🚀 创建您的专属战略档案"); st.info("请告诉我们您的商业画像，AI将为您提供高度定制化的分析。")
    with st.form(key='profile_form'):
        st.subheader("1. 您的商业身份"); seller_type = st.selectbox("卖家类型", ["品牌方", "工厂转型", "贸易商", "个人卖家"], help="您的商业模式决定了AI建议的侧重点。")
        st.subheader("2. 您的核心供应链"); supply_chain = st.text_input("主营/感兴趣品类 (用逗号分隔)", "消费电子, 户外用品, 宠物家居", help="输入您有优势或感兴趣的产品大类。")
        st.subheader("3. 您的目标市场与定价策略"); target_market = st.text_input("意向市场", "德国", help="您想进入哪个国家或地区？"); price_range = st.slider("期望产品售价区间 ($)", 1, 500, (20, 80), help="设定价格范围有助于AI精准推荐。")
        if st.form_submit_button(label='✔️ 保存档案并开始分析'):
            st.session_state.profile = {"seller_type": seller_type, "supply_chain": supply_chain, "target_market": target_market, "min_price": price_range[0], "max_price": price_range[1]}; st.session_state.profile_created = True; st.rerun()

def render_insight_page():
    """渲染第一步：机会洞察页面，实现UI状态分离"""
    st.title("第一步：机会洞察 (Insight)"); st.markdown("基于您的战略档案，AI专家团队将为您分析目标市场，识别高潜力机会。")
    if not (api_key_from_env := os.environ.get("ARK_API_KEY")): st.error("无法找到 API Key。请确保 .env 文件配置正确。"); return

    if st.session_state.market_report_content and not st.session_state.report_generating:
        st.markdown("---"); st.header("📈 您的动态机会档案"); st.markdown(st.session_state.market_report_content)
        st.success("机会洞察已完成！请前往下一步进行“自我验证”。")
        if st.button("🔄 重新生成报告"): st.session_state.market_report_content = ""; st.session_state.report_generating = True; st.rerun()
    elif st.session_state.report_generating:
        st.info("报告生成中，请稍候...")
        expander = st.expander("点击查看AI的思考过程 🧠", expanded=True)
        thinking_placeholder = expander.empty(); report_placeholder = st.empty()
        full_response_text = ""; separator_think_end = "<<<<THINKING_ENDS>>>>"; separator_report_start = "<<<<REPORT_STARTS>>>>"
        try:
            ark_client = Ark(api_key=api_key_from_env)
            for chunk in generate_full_report_stream(ark_client, st.session_state.profile):
                full_response_text += chunk
                if separator_report_start in full_response_text:
                    parts1 = full_response_text.split(separator_think_end, 1); thinking_placeholder.markdown(parts1[0])
                    report_part = parts1[1].split(separator_report_start, 1)[1]; report_placeholder.markdown(report_part)
                elif separator_think_end in full_response_text: thinking_placeholder.markdown(full_response_text.split(separator_think_end, 1)[0])
                else: thinking_placeholder.markdown(full_response_text + "...")
            if separator_report_start in full_response_text: st.session_state.market_report_content = full_response_text.split(separator_report_start, 1)[1].strip()
            else: st.session_state.market_report_content = "AI未能生成格式正确的报告，请重试。"
        except Exception as e: st.error(f"生成报告时发生严重错误: {e}"); st.session_state.market_report_content = ""
        st.session_state.report_generating = False; st.success("报告生成完毕！"); time.sleep(1); st.rerun()
    else:
        if st.button("🤖 调用AI专家团队生成报告", type="primary", use_container_width=True): st.session_state.report_generating = True; st.rerun()

def render_validation_page():
    """渲染第二步：自我验证页面"""
    st.title("第二步：自我验证 (Validation)"); st.info("上传您的历史销售和评论数据，看看您的业务现状与AI发现的机会点是否匹配。")
    uploaded_amazon = st.file_uploader('1. 上传 Amazon 销售报告', type=['csv', 'parquet'], key='amazon_uploader')
    uploaded_reviews = st.file_uploader('2. 上传 Amazon 评论数据 (可选)', type=['csv', 'parquet'], key='reviews_uploader')
    if uploaded_amazon: st.session_state.validation_data['amazon_df'] = load_uploaded_file(uploaded_amazon, {'ASIN': str, 'asin': str})
    if uploaded_reviews: st.session_state.validation_data['reviews_df'] = load_uploaded_file(uploaded_reviews)
    if (amazon_df := st.session_state.validation_data.get('amazon_df')) is not None:
        with st.status("⚙️ 正在清洗和适配您的销售数据...", expanded=True) as status:
            try:
                for old, new in {'Total Sales':'Amount','Product':'SKU','Quantity':'Qty','Order_ID':'Order ID'}.items():
                    if old in amazon_df.columns: amazon_df.rename(columns={old:new}, inplace=True)
                if missing := [c for c in ["Amount","Category","Date","Status","SKU","Order ID","Qty"] if c not in amazon_df.columns]: status.update(label="数据清洗失败!", state="error"); st.error(f"Amazon 文件中缺少关键列: {', '.join(missing)}"); return
                amazon_df.dropna(subset=["Amount", "Category", "Date"], inplace=True)
                try: amazon_df["Date"] = pd.to_datetime(amazon_df["Date"], format='%m-%d-%y')
                except ValueError: amazon_df["Date"] = pd.to_datetime(amazon_df["Date"], errors='coerce')
                amazon_df["Amount"] = pd.to_numeric(amazon_df["Amount"], errors='coerce'); amazon_df = amazon_df[amazon_df["Status"].isin(["Shipped","Shipped - Delivered to Buyer","Completed","Pending","Cancelled"])]; amazon_df.dropna(subset=['Date','Amount','SKU','Order ID','Qty'], inplace=True)
                status.update(label="数据清洗与适配完成!", state="complete", expanded=False)
            except Exception as e: status.update(label="数据处理出错!", state="error"); st.error(f"数据清洗时发生错误: {e}"); return
        st.success("数据已准备就绪！请浏览下方的分析结果。")
        tab_lstm, tab_category, tab_cluster, tab_sentiment = st.tabs(["🧠 LSTM销售预测", "🛍️ 品类表现", "🔥 热销品聚类", "💬 情感分析"])
        with tab_lstm: st.header("销售额深度学习预测 (LSTM)"); st.plotly_chart(perform_lstm_forecast(amazon_df), use_container_width=True)
        with tab_category: st.header("产品类别销售表现"); st.plotly_chart(create_category_sales_plot(amazon_df))
        with tab_cluster:
            st.header("热销商品聚类分析")
            cluster_summary, hot_products, cluster_error = perform_product_clustering(amazon_df)
            if cluster_error: st.error(cluster_error)
            elif cluster_summary is not None and hot_products is not None:
                st.subheader("各商品簇特征均值"); st.dataframe(cluster_summary); st.subheader(f"🔥 热销商品列表 (共 {len(hot_products)} 个)"); st.dataframe(hot_products.head(20))
                st.session_state.validation_summary = f"内部数据显示，我的热销商品主要集中在簇{cluster_summary.index[0]}，平均销售额为{cluster_summary.iloc[0]['total_amount']:.2f}。主要热销SKU包括：{', '.join(hot_products['SKU'].head(3).tolist())}。"
        with tab_sentiment:
            st.header("客户评论情感分析")
            if (reviews_df := st.session_state.validation_data.get('reviews_df')) is not None:
                sentiment_df, sentiment_error = perform_sentiment_analysis(reviews_df)
                if sentiment_error: st.error(sentiment_error)
                elif sentiment_df is not None:
                    st.subheader("按星级筛选评论"); rating_range = st.slider('选择星级范围:', 1, 5, (1, 5)); min_r, max_r = rating_range
                    filtered_reviews = sentiment_df[(sentiment_df['rating'] >= min_r) & (sentiment_df['rating'] <= max_r)]
                    st.markdown(f"**显示 {len(filtered_reviews)} 条评分为 {min_r} 到 {max_r} 星的评论**"); st.dataframe(filtered_reviews[['rating', 'review_text', 'sentiment']])
                    st.subheader("情感分数统计"); avg_filtered = filtered_reviews['sentiment'].mean() if not filtered_reviews.empty else 0; avg_all = sentiment_df['sentiment'].mean()
                    c1, c2 = st.columns(2); c1.metric(f"所选评论 ({min_r}-{max_r} 星) 的平均情感分", f"{avg_filtered:.2f}"); c2.metric("所有评论的平均情感分", f"{avg_all:.2f}")
                    st.markdown("---"); st.subheader("🤖 AI 评论深度分析报告")
                    if st.button("生成 AI 分析报告", key="generate_review_report"):
                        if not (api_key_from_env := os.environ.get("ARK_API_KEY")): st.error("无法找到 API Key。请确保 .env 文件配置正确。")
                        elif filtered_reviews.empty: st.warning("当前筛选范围内没有评论可供AI分析，请调整上方的星级滑块。")
                        else:
                            expander = st.expander("查看AI思考过程", expanded=True); thinking_placeholder = expander.empty(); report_placeholder = st.empty()
                            full_response_text = ""; separator_think_end = "<<<<THINKING_ENDS>>>>"; separator_report_start = "<<<<REPORT_STARTS>>>>"
                            positive_samples = filtered_reviews.nlargest(15, 'sentiment'); negative_samples = filtered_reviews.nsmallest(15, 'sentiment')
                            positive_text = "\n".join([f"- {row['review_text']}" for _, row in positive_samples.iterrows()]); negative_text = "\n".join([f"- {row['review_text']}" for _, row in negative_samples.iterrows()])
                            try:
                                ark_client = Ark(api_key=api_key_from_env)
                                for chunk in generate_review_summary_report(ark_client, positive_text, negative_text):
                                    full_response_text += chunk
                                    if separator_report_start in full_response_text:
                                        parts1 = full_response_text.split(separator_think_end, 1); thinking_placeholder.markdown(parts1[0])
                                        report_part = parts1[1].split(separator_report_start, 1)[1]; report_placeholder.markdown(report_part)
                                    elif separator_think_end in full_response_text: thinking_placeholder.markdown(full_response_text.split(separator_think_end, 1)[0])
                                    else: thinking_placeholder.markdown(full_response_text + "...")
                            except Exception as e: st.error(f"生成评论分析报告时出错: {e}")
            else: st.info("请上传评论文件以启用此分析。")
    else: st.warning("请上传您的销售数据以开始自我验证。")

def render_action_page():
    """渲染第三步：行动方案页面"""
    st.title("第三步：行动方案 (Action Plan)"); st.info("结合AI的市场洞察和您的内部数据验证，一键生成为您量身定制的行动计划。")
    if not st.session_state.market_report_content: st.warning("请先在“机会洞察”步骤生成市场分析报告。"); return
    if not st.session_state.validation_summary: st.warning("请先在“自我验证”步骤上传数据并完成分析，以提供内部数据参考。"); return
    st.subheader("决策依据预览")
    with st.expander("点击查看AI的市场洞察报告"): st.markdown(st.session_state.market_report_content)
    with st.expander("点击查看您的内部数据验证摘要"): st.markdown(st.session_state.validation_summary)
    if not (api_key_from_env := os.environ.get("ARK_API_KEY")): st.error("无法找到 API Key。请确保 .env 文件配置正确。"); return
    if st.button("💡 生成我的行动计划", type="primary", use_container_width=True):
        expander = st.expander("查看AI思考过程", expanded=True); thinking_placeholder = expander.empty(); report_placeholder = st.empty()
        full_response_text = ""; separator_think_end = "<<<<THINKING_ENDS>>>>"; separator_report_start = "<<<<REPORT_STARTS>>>>"
        try:
            ark_client = Ark(api_key=api_key_from_env)
            for chunk in agent_action_planner(ark_client, st.session_state.market_report_content, st.session_state.validation_summary):
                full_response_text += chunk
                if separator_report_start in full_response_text:
                    parts1 = full_response_text.split(separator_think_end, 1); thinking_placeholder.markdown(parts1[0])
                    report_part = parts1[1].split(separator_report_start, 1)[1]; report_placeholder.markdown(report_part)
                elif separator_think_end in full_response_text: thinking_placeholder.markdown(full_response_text.split(separator_think_end, 1)[0])
                else: thinking_placeholder.markdown(full_response_text + "...")
        except Exception as e: st.error(f"生成行动计划时出错: {e}")

# ==============================================================================
# 6. 主应用布局与逻辑
# ==============================================================================
st.set_page_config(layout="wide", page_title="WeaveAI 智能分析助手")
st.title('📈 WeaveAI 智能分析助手 v2.8')
st.caption('##### 告别感觉，让数据与AI为您引航')

initialize_session_state()

if not st.session_state.profile_created:
    render_profile_creation_page()
else:
    with st.sidebar:
        st.header(f"🧭 WeaveAI 战略导航")
        with st.expander("📝 我的战略档案", expanded=True):
            profile = st.session_state.profile
            st.markdown(f"**商业身份:** {profile['seller_type']}\n\n**目标市场:** {profile['target_market']}\n\n**核心品类:** {profile['supply_chain']}\n\n**定价区间:** ${profile['min_price']} - ${profile['max_price']}")
            if st.button("✏️ 修改档案"): st.session_state.profile_created = False; st.rerun()
        st.divider()
        st.session_state.current_step = st.radio("分析流程", ['机会洞察', '自我验证', '行动方案'], captions=["AI驱动的市场机会发现", "用您的数据验证可行性", "生成可执行的行动计划"], key='navigation_radio')
    
    if st.session_state.current_step == '机会洞察': render_insight_page()
    elif st.session_state.current_step == '自我验证': render_validation_page()
    elif st.session_state.current_step == '行动方案': render_action_page()