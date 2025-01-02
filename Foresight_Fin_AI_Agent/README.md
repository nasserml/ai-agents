# 🚀 Foresight_Fin_AI_Agent:  A Suite of Powerful AI Agents 🚀

This folder contains a collection of AI agents designed for financial analysis and information retrieval, leveraging cutting-edge language models and specialized tools.

---

## 1. 🤖 Simple Groq Agent (`1_simple_groq_agent.py`)

This script demonstrates a basic agent using the **Groq** language model (specifically, "llama-3.3-70b-versatile"). It's a great starting point for interacting with the model.

```python
from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile")  # 🦙 The powerful Llama model!
)

agent.print_response("Write 2 sentences poem about a cat and a dog")  # 📝 Let's get creative!
```

### Key Features:

*   **Fast Inference:** Groq is known for its speed. ⚡
*   **Simple Setup:** Easy to get started with a basic agent. 👍

---

## 2. 💰 Finance Agent (`2_finance_agent.py`)

This is where things get interesting! This agent is equipped with **YFinanceTools** to access real-time and historical financial data.

```python
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(
        stock_price=True,          # 📈 Get current prices
        analyst_recommendations=True, # 🤔 Expert opinions
        stock_fundamentals=True,     # 📊 Key financial metrics
        company_news=True           # 📰 Latest headlines
    )],
    show_tool_calls=True,         # 🔍 See how the agent uses its tools
    markdown=True,                # ✨ Beautifully formatted output
    instructions=["Use tables to display the data"], # 🏓 Structured results
    debug_mode=True
)

agent.print_response(
    "Summarize and compare analyst recommendations and fundamentals for NVDA and TSLA" # 🆚 Let's compare these tech giants!
)
```

## Key Features:

*   **Financial Data Access:**  Leverages `yfinance` to get stock data, recommendations, fundamentals, and news. 📊
*   **Structured Output:**  The agent is instructed to use tables for clear data presentation. 🏓
*   **Transparency:**  `show_tool_calls` lets you see exactly how the agent is using its tools. 👁️‍🗨️

---

## 3. 🤝 Agent Teams with Groq (`3_agent_teams_groq.py`)

This script showcases the power of collaboration! It defines two specialized agents:

*   **🌐 Web Agent:** Uses **DuckDuckGo** to search the web.
*   **💰 Finance Agent:**  (Same as in example 2)

These agents work together as a team under a managing agent to provide comprehensive answers.

```python
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
load_dotenv()

web_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],            # 🦆 Web searching
    instructions=["Always include sources"], # 📚 Cite your sources!
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_agent, finance_agent],  # 🤝 Teamwork makes the dream work!
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

agent_team.print_response(
    "Summarize and compare analyst recommendations and fundamentals for NVDA and TSLA", stream=True # ⚡ Get results as they come!
)
```

### Key Features:

*   **Collaboration:**  Agents work together to provide more complete answers. 🤝
*   **Specialization:** Each agent has its own area of expertise. 🧠
*   **Web Integration:**  The Web Agent can bring in information from the internet. 🌐
*   **Streaming:** Results are streamed in real-time. ⚡

---

**🌟 Conclusion 🌟**

The **Foresight_Fin_AI_Agent** project offers a powerful and flexible framework for building AI agents that can access, analyze, and present financial information. With its use of advanced language models, specialized tools, and collaborative agent teams, this project demonstrates the exciting possibilities of AI in the world of finance. 
