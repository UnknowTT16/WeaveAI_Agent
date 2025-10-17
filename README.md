# 📈 WeaveAI 智能分析助手 (WeaveAI Intelligent Analysis Assistant)

**告别感觉，让数据与AI为您引航。**

---

## 📖 项目简介 (Project Overview)

**WeaveAI** 是一个专为跨境电商卖家设计的、集成化的智能决策平台。它巧妙地将前沿的生成式AI市场洞察，与卖家自身的历史销售数据深度挖掘相结合，构建了一个从“**机会洞察 -> 自我验证 -> 行动方案**”的完整商业分析闭环。

本应用基于 Python 和 Streamlit 构建，旨在通过一个直观、交互式的Web界面，赋能卖家在市场选择、产品优化和用户反馈分析等多个维度做出更明智、更自信的决策。

## ✨ 核心功能 (Key Features)

本应用通过一个引导式的三步战略工作流，将所有功能模块有机地串联起来：

#### **第一步：机会洞察 (Insight)**
*   **🚀 专属战略档案**: 在分析前创建您的商业画像（卖家类型、核心品类、目标市场等），让AI的每一次分析都“量身定制”。
*   **🤖 AI专家团队报告**: 调用大语言模型，模拟“高级战略顾问”的角色，生成一份包含**市场机遇、竞争格局、数据来源**的专业市场分析报告。
*   **🧠 流式思考过程**: 在AI生成报告时，实时展示其“思考过程”，并将思考与结果分离呈现，提供卓越的交互体验。

#### **第二步：自我验证 (Validation)**
上传您自己的销售和评论数据，对AI提出的机会点进行内部数据验证。
*   **🧠 LSTM销售预测**: 利用深度学习模型，预测未来30天的销售额。
*   **🔥 热销品聚类**: 通过K-Means算法，自动发现您产品组合中的“热销品”与“潜力品”。
*   **🛍️ 品类表现分析**: 直观对比不同产品类别的销售表现。
*   **💬 交互式情感分析**:
    *   通过滑块筛选不同星级的用户评论。
    *   实时对比“筛选后”与“全体”评论的情感分数。
    *   一键调用AI，对评论样本进行深度分析，提炼核心优缺点与改进建议。

#### **第三步：行动方案 (Action Plan)**
*   **💡 AI生成行动计划**: 综合第一步的市场洞察和第二步的数据验证结果，由AI为您生成一份包含**产品、营销、运营**三个方面的、可执行的待办事项列表。

## 🛠️ 技术栈 (Technology Stack)

*   **核心框架**: `Streamlit`
*   **数据处理**: `Pandas`, `Numpy`
*   **AI与机器学习**:
    *   `volcenginesdkarkruntime`: 用于调用大语言模型API。
    *   `Scikit-learn`: 用于K-Means聚类分析。
    *   `TensorFlow (Keras)`: 用于构建和训练LSTM深度学习模型。
    *   `vaderSentiment`: 用于进行快速的情感分析。
*   **数据可视化**: `Plotly`
*   **辅助工具**: `python-dotenv`, `stqdm`, `python-docx`

## 🚀 快速开始 (Getting Started)

请按照以下步骤在您的本地环境中设置并运行本项目。

### 1. 先决条件 (Prerequisites)

*   **Python**: 强烈推荐使用 **Python 3.10** 或 **Python 3.11** 版本，以确保与所有机器学习库的最佳兼容性。
*   **Git**: 用于克隆本项目。

### 2. 安装步骤 (Installation)

1.  **克隆仓库**
    ```bash
    git clone [您的仓库SSH或HTTPS链接]
    cd [您的仓库名称]
    ```

2.  **创建并激活虚拟环境**
    ```bash
    # 使用 Python 3.10 创建一个名为 venv 的虚拟环境
    python3.10 -m venv venv

    # 激活环境 (macOS / Linux)
    source venv/bin/activate

    # 激活环境 (Windows)
    .\venv\Scripts\activate
    ```

3.  **安装依赖项**
    在**已激活**的虚拟环境中，运行以下命令：
    ```bash
    pip install -r requirements.txt
    ```

4.  **设置环境变量**
    *   在项目根目录下，创建一个名为 `.env` 的文件。
    *   在 `.env` 文件中，添加您的API密钥，格式如下：
        ```
        ARK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ```
    *(请将 `sk-xxxxxxxx...` 替换为您自己的真实API密钥)*

### 3. 运行应用

确保您的虚拟环境已激活，然后运行：
```bash
streamlit run WAIapp.py
