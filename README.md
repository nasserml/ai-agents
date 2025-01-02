# 🤖 AI-Agents: Your Comprehensive Toolkit for Building, Exploring, and Deploying Autonomous Intelligence 🚀

Welcome to the **AI-Agents** repository! This project is your comprehensive gateway to the exciting world of autonomous AI agents. Here, you'll find a powerful and versatile toolkit for building, experimenting with, and deploying intelligent agents capable of tackling a wide range of real-world tasks. Whether you're a seasoned AI developer or just starting your journey, this repository provides the resources and examples you need to unlock the full potential of AI.

![Agents](https://img.freepik.com/free-photo/robot-handshake-human-background-futuristic-digital-age_53876-129770.jpg)

## ✨ Project Highlights: Dive into the World of AI Agents

This repository offers a rich collection of pre-built agents, cutting-edge technologies, and user-friendly tools designed to empower you in your AI endeavors:

*   **Diverse Agent Capabilities:** Explore a suite of agents tailored for specific domains, showcasing the versatility of autonomous intelligence:
    *   **💰 Foresight_Fin_AI_Agent:** Your go-to expert for in-depth financial analysis. This agent leverages the power of **Groq's** lightning-fast language models and **YFinanceTools** to provide you with comprehensive stock insights, analyst recommendations, and the latest company news. Data is presented in clear, concise tables for easy understanding.
    *   **🌐 Web Search Agent:** Harness the power of the privacy-focused search engine **DuckDuckGo** to scour the web for information. This agent ensures you always get accurate results with proper source attribution, making it a reliable research companion.
    *   **🧑‍🍳 PDF Assistant:**  Need help deciphering complex documents or finding specific information within a PDF? This agent is designed to be a domain-specific expert, extracting knowledge from PDF files using a combination of **Groq**, **PgVector2**, and **PostgreSQL**. The provided example focuses on Thai recipes, demonstrating how you can adapt this agent to any field.
*   **Cutting-Edge Technology Stack:**  We've integrated some of the most advanced technologies in the AI landscape to provide you with a powerful and efficient development experience:
    *   **🦙 Groq LLMs:** Experience the unparalleled speed and efficiency of Groq's language models, including the powerful `llama-3.3-70b-versatile` and the cutting-edge `llama3-groq-70b-8192-tool-use-preview`. These models provide the intelligence behind your agents.
    *   **🔍 YFinanceTools:** Gain seamless access to a wealth of financial data from Yahoo Finance, empowering your agents with real-time and historical market information.
    *   **🦆 DuckDuckGo API:** Integrate privacy-focused web search seamlessly into your agents, providing them with the ability to gather information from across the internet.
    *   **🐘 PgVector and PostgreSQL:** Leverage the power of vector databases for efficient knowledge retrieval and storage. This combination allows your agents to quickly find and process relevant information from large datasets.
*   **The Power of Collaboration: Agent Teams:** Witness the magic of agent teams! This repository demonstrates how you can combine the strengths of specialized agents, such as the Web Agent and Finance Agent, to tackle complex queries and achieve more comprehensive, insightful results.
*   **Interactive Playground:**  The `playground.py` file provides a user-friendly interface for interacting with your agents. Experiment with different prompts, tweak parameters, and witness the power of your AI assistants firsthand in a dynamic environment.
*   **Easy to Use, Customize, and Extend:** Built with modularity and ease of use in mind, this repository makes it simple to customize existing agents or create your own from scratch. Integrate new tools, models, and knowledge bases to tailor the agents to your specific needs and unlock new possibilities.

## 📂 Repository Structure: A Detailed Map of the Project

This repository is organized into distinct directories, each containing valuable resources and examples:

*   **`/Financial AI analys`:**
    *   **`financial_agent.py`:**  The core script for the multi-agent system. It defines the Web Search Agent, Finance Agent, and a coordinating agent that leverages their combined expertise.
    *   **`README.md`:**  A detailed explanation of the multi-agent system, its functionality, and how to use it.
    *   **Purpose:** This directory showcases a sophisticated multi-agent system that combines web search and financial analysis capabilities. It's perfect for users who need a comprehensive overview of market trends, company performance, and relevant news.

*   **`/Foresight_Fin_AI_Agent`:**
    *   **`1_simple_groq_agent.py`:** A basic example demonstrating interaction with the Groq language model.
    *   **`2_finance_agent.py`:**  A more advanced agent equipped with YFinanceTools for accessing and analyzing financial data.
    *   **`3_agent_teams_openai.py`:** Demonstrates the power of agent teams by combining a Web Agent (using DuckDuckGo) and a Finance Agent to provide comprehensive answers to complex financial queries.
    *   **`README.md`:**  A comprehensive guide to the different financial AI agents in this directory, their features, and how they work together.
    *   **Purpose:** This directory offers a range of financial AI agents, from a simple chatbot powered by Groq to a sophisticated team of agents that leverage `yfinance` and DuckDuckGo for in-depth financial analysis. It's a great starting point for users interested in building AI-powered financial tools.

*   **`/Pdfassistant`:**
    *   **`pdf_assistant.py`:** The core script for the PDF Assistant. It sets up the knowledge base, vector database, and assistant agent.
    *   **`README.md`:**  A detailed explanation of the PDF Assistant, its functionality, and how to adapt it to different domains.
    *   **Purpose:** This directory demonstrates how to build a custom AI assistant that can extract knowledge from PDF documents. The provided example focuses on Thai recipes, showcasing how you can create a domain-specific expert by leveraging Groq, PgVector2, and PostgreSQL.

*   **`playground.py`:**
    *   **Purpose:** This file provides an interactive playground for testing and experimenting with the pre-built agents (Finance Agent and Web Search Agent). It uses the `FastAPI` and `uvicorn` to serve a web application where users can interact with the agents in real-time.

*   **`.env.exmaple`:**
    *   **Purpose:** A template for your environment variables. You'll need to create a `.env` file based on this example and fill in your API keys for Groq, OpenAI, and Phi (if you're using the Phi API).

*   **`requirements.txt`:**
    *   **Purpose:** Lists all the Python packages required to run the project. This ensures you have all the necessary dependencies installed.

## 🚀 Getting Started: Your Journey into AI Agents Begins Here

Follow these simple steps to get started with the AI-Agents project:

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/nasserml/ai-agents.git
    cd ai-agents-main
    ```

    Replace `<repository_url>` with the actual URL of your repository.

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    This command installs all the necessary Python packages listed in `requirements.txt`.

3. **Set Up Your Environment:**

    *   Create a `.env` file in the root directory of the project by copying the provided `.env.example`:

        ```bash
        cp .env.exmaple .env
        ```

    *   Open the `.env` file and fill in your API keys for **Groq**, **OpenAI**, and **Phi** (if applicable). These keys are essential for the agents to function correctly.

4. **Explore the Examples:**

    *   Navigate to the different agent directories (e.g., `Foresight_Fin_AI_Agent`, `Pdfassistant`).
    *   Each directory contains example scripts that demonstrate how to use the agents. For instance, to run the finance agent, you would navigate to the `Foresight_Fin_AI_Agent` directory and execute:

        ```bash
        cd Foresight_Fin_AI_Agent
        python 2_finance_agent.py
        ```

    *   Similarly, explore the other examples to understand the capabilities of each agent.

5. **Experiment in the Playground:**

    ```bash
    uvicorn playground:app --reload
    ```
    *   This command starts a local web server that hosts the interactive playground. The `--reload` flag automatically restarts the server whenever you make changes to the code, making development easier.
    *   Open your web browser and go to `http://localhost:8000` (or the port specified by Uvicorn if it's different).
    *   Interact with the pre-built agents (Finance Agent and Web Search Agent) and see how they respond to your queries! Experiment with different prompts and explore their capabilities.

## 🤝 Contributing: Join Us in Shaping the Future of AI

We wholeheartedly welcome contributions to the AI-Agents project! Whether you want to improve existing agents, add exciting new features, or create entirely new agents to expand the capabilities of this toolkit, your input is valuable.

Please see our [CONTRIBUTING.md](CONTRIBUTING.md) (you'll need to create this file) for detailed guidelines on how to contribute. We encourage you to submit pull requests, report issues, and suggest enhancements to help us make this project even better.

## 📄 License: Understanding the Terms of Use

This project is licensed under the [MIT License](LICENSE) (you'll need to create this file) - see the LICENSE file for details. The MIT License is a permissive open-source license that allows you to use, modify, and distribute the code in this repository with minimal restrictions.

## 🙏 Acknowledgements: Giving Credit Where It's Due

We extend our sincere gratitude to the developers and maintainers of the following projects, which are integral to the AI-Agents repository:

*   **[Groq](https://www.groq.com/):** For their revolutionary language models that provide the intelligence behind our agents.
*   **[Yahoo Finance](https://finance.yahoo.com/):** For providing a wealth of financial data through their API.
*   **[DuckDuckGo](https://duckduckgo.com/):** For their privacy-focused search engine and API.
*   **[yfinance](https://github.com/ranaroussi/yfinance):** For making it easy to access financial data from Yahoo Finance in Python.
*   **[PgVector](https://github.com/pgvector/pgvector):** For enabling efficient vector storage and similarity search in PostgreSQL.

## 🌟 Star This Repo!

If you find this project helpful, interesting, or valuable in any way, please give it a star! ⭐ Your support helps others discover the project and motivates us to keep improving it.

**Let's embark on this exciting journey together and build the future of AI!** 🤖🤝