# 🔮 Foresight Fin-AI Agent 📈

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

Welcome to the **Foresight Fin-AI Agent** repository! 🚀 This project leverages the power of advanced language models and cutting-edge tools to create a sophisticated AI agent capable of providing insightful financial analysis and up-to-date market information. Whether you need to know the latest stock prices, analyst recommendations, or general market news, this agent has got you covered! ✨

## 🌟 Features

*   **Web Search Agent 🌐:** Utilizes `DuckDuckGo` to fetch and summarize information from the web, ensuring you always have access to the latest news and data.
*   **Finance Agent 💰:** Equipped with `YFinanceTools`, this agent can retrieve stock prices, analyst recommendations, company news, and fundamental financial data, presenting it in easy-to-read tables.
*   **Multi-Agent Collaboration 🤝:** Combines the power of the Web Search Agent and Finance Agent in a seamless team to provide comprehensive insights.
*   **Groq Integration 🔥:** Powered by `Groq`'s `llama3-groq-70b-8192-tool-use-preview` model for blazing-fast responses and accurate information retrieval.
*   **OpenAI Integration 🧠:** Leverages `OpenAI` for enhanced natural language processing and understanding.
*   **Interactive Playground 🎮:** Experiment with the agents in real-time using the built-in `Playground`, allowing for hands-on exploration and testing.
*   **Markdown Support 📝:** Responses are formatted using Markdown for clear, structured, and visually appealing output.
*   **Streamed Responses ⚡️:** Get responses streamed in real-time for a dynamic and responsive experience.

## 🛠️ Installation

Follow these steps to get your Foresight Fin-AI Agent up and running:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/nasserml/foresight-fin-ai-agent.git
    cd foresight-fin-ai-agent
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your environment variables:**

    Create a `.env` file in the root directory based on the provided `.env.example`:

    ```bash
    cp .env.example .env
    ```

    Then, fill in your API keys:

    ```
    PHI_API_KEY="your_phi_api_key"
    GROQ_API_KEY="your_groq_api_key"
    OPENAI_API_KEY="your_openai_api_key"
    ```

## 🚀 Usage

### Running the Financial Agent Script

To see the Financial Agent and Web Search Agent in action, simply run the `financial_agent.py` script:

```bash
python financial_agent.py
```

This script will:

1. Initialize a **Web Search Agent** using `DuckDuckGo` for web searching.
2. Initialize a **Finance Agent** using `YFinanceTools` to fetch financial data.
3. Combine both agents into a **Multi-AI Agent** team.
4. Prompt the Multi-AI Agent to summarize analyst recommendations and share the latest news for NVDA, and stream the response.

### Launching the Interactive Playground

To interact with the agents in a dynamic environment, start the Playground:

```bash
uvicorn playground:app --reload
```

This will launch a web server where you can:

*   Interact with the **Web Search Agent** and **Finance Agent**.
*   Experiment with different prompts and see their responses in real-time.
*   Experience the power of the agents in a user-friendly interface.

## 🤖 Agent Details

### Web Search Agent

*   **Name:** Web Search Agent
*   **Role:** Search the web for information.
*   **Model:** Groq (`llama3-groq-70b-8192-tool-use-preview`)
*   **Tools:** DuckDuckGo
*   **Instructions:** Always include sources.
*   **Features:** Show tool calls, Markdown output.

### Finance Agent

*   **Name:** Finance Agent
*   **Role:** Provide financial analysis and data.
*   **Model:** Groq (`llama3-groq-70b-8192-tool-use-preview`)
*   **Tools:** YFinanceTools (stock price, analyst recommendations, stock fundamentals, company news)
*   **Instructions:** Use tables to display data.
*   **Features:** Show tool calls, Markdown output.

### Multi-AI Agent

*   **Team:** Web Search Agent, Finance Agent
*   **Instructions:** Always include sources, Use tables to display data.
*   **Features:** Show tool calls, Markdown output.

## 🤝 Contributing

We welcome contributions to the Foresight Fin-AI Agent project! Whether it's improving the agent's capabilities, fixing bugs, or enhancing the documentation, your help is greatly appreciated.

1. **Fork** the repository.
2. **Create** a new branch for your feature or bug fix.
3. **Commit** your changes with clear and concise commit messages.
4. **Push** your branch to your fork.
5. **Open** a pull request to the main repository.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

*   **Phi:** For the powerful agent framework.
*   **Groq:** For the high-performance language model.
*   **OpenAI:** For the advanced NLP capabilities.
*   **YFinance:** For providing easy access to financial data.
*   **DuckDuckGo:** For the privacy-focused search engine.
*   And **you**, for your interest and contributions! ❤️

---

Made with ❤️ by the **AI Community**. Let's build the future of finance together! 🌠

