# 🌟 **Multi-Agent AI Application with Web Search and Financial Capabilities** 🚀

This project demonstrates how to build a powerful multi-agent AI system using **Groq**, **DuckDuckGo**, and **YFinanceTools**. The AI system provides robust web search functionality and detailed financial analysis, making it versatile for various real-world applications. 💡

---

## 📚 **Overview**
### 🔗 **Core Functionalities**:
1. **Web Search Agent**: Searches the web and retrieves relevant information.
2. **Finance Agent**: Fetches financial data, including stock prices, fundamentals, news, and analyst recommendations.
3. **Multi-Agent Collaboration**: Combines the expertise of individual agents to deliver comprehensive responses.

---

## 🛠️ **Code Breakdown**

### 🔑 **Environment Setup**:
```python
import os
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
```
- Loads environment variables, including the OpenAI API key for authentication.

---

### 🤖 **Agent Configuration**

#### 1️⃣ **Web Search Agent**:
```python
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)
```
- **Role**: Searches the web using DuckDuckGo.
- **Instructions**: Ensures results include sources for credibility.
- **Model**: Uses `Groq (llama3-groq-70b)` for natural language processing.

---

#### 2️⃣ **Finance Agent**:
```python
finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        ),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)
```
- **Tools**: Retrieves financial data via `YFinanceTools`.
- **Capabilities**:
  - Fetches **stock prices** and **fundamentals**.
  - Displays **analyst recommendations** in a table.
  - Shares **company news**.
- **Instructions**: Outputs data in an organized table format.

---

#### 3️⃣ **Multi-Agent AI System**:
```python
multi_ai_agent = Agent(
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)
```
- Combines **Web Search Agent** and **Finance Agent**.
- Ensures collaborative functionality for complex queries.

---

### 🎯 **Usage Example**
```python
multi_ai_agent.print_response(
    "Summarize analyst recommendations and share the latest news for NVDA",
    stream=True
)
```
- **Query**: Requests a summary of analyst recommendations and the latest news for NVIDIA (NVDA).
- **Response**: Combines insights from both agents with sources and tables.

---

## 🌟 **Features**
- 🌐 **Web Search**: Real-time search results with source attribution.
- 📊 **Financial Analysis**: Comprehensive stock-related insights.
- 🤝 **Collaboration**: Multi-agent teamwork for robust outputs.
- 📝 **Markdown Support**: Well-formatted, readable responses.

---

## 🚀 **How to Run**

1. Install Dependencies:
   ```bash
   pip install phi dotenv
   ```
2. Set Environment Variables:
   - Add `OPENAI_API_KEY` to your `.env` file.
3. Run the Script:
   ```bash
   python financial_agent.py
   ```

---

## ✨ **Why This is Amazing**
- 💼 **Professional Applications**: Ideal for financial advisors, researchers, and developers.
- 🤖 **Scalable AI**: Easily extendable to include more specialized agents.
- 🛠️ **Tool Integration**: Combines advanced APIs for maximum efficiency.

---

## 🔗 **Resources**
- **Groq Model**: [Learn More](https://www.groq.com/)
- **DuckDuckGo API**: [Documentation](https://duckduckgo.com/)
- **YFinanceTools**: [GitHub Repo](https://github.com/ranaroussi/yfinance)

🚀 *Unlock the power of multi-agent systems today!* 🌟
