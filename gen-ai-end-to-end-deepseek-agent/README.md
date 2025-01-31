# 🧠 DeepSeek Code Companion: Your AI Pair Programmer 💻🤖

This project leverages the power of large language models (LLMs) 🧠 through Ollama 🦙 and LangChain 🦜 to create an interactive coding assistant Streamlit app 🎉. It's like having a super-smart coding buddy 🤓 right in your browser! 🚀

## ✨ Project Overview

The DeepSeek Code Companion 🤖 is designed to assist you with various coding tasks 👨‍💻👩‍💻, primarily focusing on Python 🐍. It utilizes the DeepSeek Coder models (1.5b and 14b parameter versions) to provide intelligent support for:

*   **🐍 Python Code Generation:** Generate Python code snippets 📃 based on your natural language descriptions 🗣️.
*   **🐞 Debugging:** Identify and fix errors ❌ in your code with helpful suggestions ✅.
*   **📝 Documentation:** Get assistance with writing clear and concise code documentation 📖.
*   **💡 Solution Design:** Explore different approaches and algorithms 💡 for your coding challenges 🤔.

## 🛠️ Project Structure

The project consists of two main components ✌️:

1. **`gen-ai-end-to-end-deepseek-agent/gen_ai_app_end_to_end_deepseek_agent.ipynb`:** A Jupyter Notebook 📓 that sets up the environment, installs dependencies, downloads the DeepSeek Coder models, and prepares the Streamlit application.
2. **`gen-ai-end-to-end-deepseek-agent/gen_ai_app_end_to_end_deepseek_agent.py`:** The Python script 🐍 that defines the Streamlit application, including the user interface, chat logic, and integration with the Ollama API through LangChain.

## 🚀 Getting Started

### Prerequisites

*   **Google Colab:** This project is designed to be run in a Google Colab environment 🤙 with a GPU 🎮.
*   **ngrok Account (Optional):** If you want to share your Streamlit app publicly 🌎, you'll need an ngrok account and an authentication token 🔑.

### Installation and Setup

The Jupyter Notebook 📓 handles the installation of all necessary dependencies:

1. **Install Packages:**
    ```bash
    !pip install streamlit 🎉
    !pip install langchain-core 🦜
    !pip install langchain-community 👥
    !pip install langchain_ollama 🦙
    !pip install pyngrok 🚇
    ```
    These commands install:
    *   **Streamlit:** For creating the web application interface 🌐.
    *   **LangChain:** To build the language model pipeline ⛓️.
    *   **Ollama:** To interact with the DeepSeek Coder models 🦙.
    *   **pyngrok:** For creating a public URL for your app (optional) 🚇.

2. **Verify GPU:**
    ```bash
    !nvidia-smi
    ```
    This command checks if a GPU is available ✅ and displays its status 🎮.

3. **Install Ollama:**
    ```bash
    !curl -fsSL https://ollama.com/install.sh | sh
    ```
    This script downloads ⬇️ and installs Ollama 🦙, a tool for running LLMs locally.

4. **Start Ollama Service:**
    ```bash
    !nohup ollama serve &
    ```
    This command starts the Ollama server in the background 🤫, making it accessible to the Streamlit app.

5. **Download DeepSeek Coder Models:**
    ```bash
    !ollama pull deepseek-coder:1.3b-instruct
    !ollama pull deepseek-coder:6.7b-instruct
    ```
    These commands download the DeepSeek Coder models 📥 (both 1.5b and 14b versions) that will be used by the application.

### The Streamlit Application (`app.py`)

The `app.py` file contains the core logic ❤️ for the Streamlit application. Here's a breakdown of its key components:

#### 1. Imports:

```python
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
```

*   Import necessary modules 📦 from Streamlit, LangChain, and Ollama.

#### 2. Custom CSS Styling:

```python
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }

    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }

    .stSelectbox svg {
        fill: white !important;
    }

    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }

    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
```

*   Adds custom CSS 🎨 to give the app a dark theme 🕶️ and style the select box ☑️.

#### 3. Title and Caption:

```python
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")
```

*   Sets the title and a catchy caption 😜 for the application.

#### 4. Sidebar Configuration:

```python
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-coder:1.3b-instruct", "deepseek-coder:6.7b-instruct"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) 🦙 | [LangChain](https://python.langchain.com/) 🦜")
```

*   Creates a sidebar with:
    *   A select box ☑️ to choose between the two DeepSeek Coder models.
    *   A description of the model's capabilities 💪.
    *   Links to the Ollama and LangChain websites 🔗.

#### 5. Initialize Chat Engine:

```python
llm_engine=ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)
```

*   Initializes the `ChatOllama` instance 🦙 with the selected model and sets the `temperature` 🌡️ for controlling the creativity 🎨 of the responses.

#### 6. System Prompt:

```python
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)
```

*   Defines the system prompt 💬 that sets the role and behavior of the AI assistant 🤖.

#### 7. Session State Management:

```python
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]
```

*   Initializes a `message_log` 📝 in the session state to store the conversation history 📚. The first message is a greeting from the AI 👋.

#### 8. Chat Container and Message Display:

```python
chat_container = st.container()

with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
```

*   Creates a container 📦 for the chat interface.
*   Iterates through the `message_log` 📝 and displays each message with the appropriate role (user 🧑‍💻 or AI 🤖).

#### 9. Chat Input and Processing:

```python
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    processing_pipeline=prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})

    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    st.session_state.message_log.append({"role": "ai", "content": ai_response})

    st.rerun()
```

*   Provides a chat input box 💬 for the user to type their queries.
*   Defines two functions:
    *   `generate_ai_response`: Sends the prompt chain to the LLM engine 🧠 and returns the response.
    *   `build_prompt_chain`: Constructs the prompt chain ⛓️ by combining the system prompt with the conversation history 📚.
*   When the user enters a query:
    *   Adds the user's message to the `message_log` ➕.
    *   Displays a spinner 🔄 while processing.
    *   Builds the prompt chain.
    *   Generates the AI response 🤖.
    *   Adds the AI response to the `message_log` ➕.
    *   Triggers a rerun of the Streamlit app to update the chat display 🔄.

## 🌐 Running the Application

1. **Execute the Jupyter Notebook:** Run all the cells in the `gen_ai_app_end_to_end_deepseek_agent.ipynb` notebook. This will install dependencies, download models, start the Ollama service, and create the `app.py` file.

2. **Run the Streamlit App (Two Options):**
    *   **Local Tunnel (ngrok):**
        ```bash
        !streamlit run app.py & npx localtunnel --port 8501
        ```
        This command starts the Streamlit app and uses `localtunnel` to create a temporary public URL. You'll need to have `npx` installed (usually comes with Node.js).

    ***OR***
    ```bash
    !streamlit run app.py &>/dev/null&
    ```
    ```python
    from pyngrok import ngrok

    # Setup a tunnel to port 8501
    public_url = ngrok.connect(8501)
    print('Public URL:', public_url)
    ```
    *   **Directly in Colab (Limited):**
        You can run the Streamlit app directly within the Colab environment using:
        ```bash
        !streamlit run app.py
        ```
        However, this method might have limitations in terms of interactivity and external access.

3. **Access the App:** Once the app is running, you'll see a URL in the output. Click on this URL to open the DeepSeek Code Companion in your browser.

## 🤝 Using the DeepSeek Code Companion

1. **Choose a Model:** In the sidebar, select either the "deepseek-coder:1.3b-instruct" or "deepseek-coder:6.7b-instruct" model. The 6.7b model is larger and potentially more powerful 💪 but might be slower.
2. **Start Chatting:** Type your coding questions ❓ or requests in the chat input box at the bottom of the app.
3. **Interact:** The AI will respond to your queries 🗣️, providing code snippets, explanations, or debugging suggestions. You can continue the conversation 💬, asking follow-up questions or refining your requests.

## 🌟 Example Interactions

Here are a few examples of how you can use the DeepSeek Code Companion:

*   **"Write a Python function 🐍 to calculate the factorial of a number."**
*   **"I'm getting a TypeError ❌ in this code, can you help me debug it? 🐞"**
*   **"How can I improve the efficiency ⏱️ of this sorting algorithm?"**
*   **"Generate documentation 📝 for this Python class."**
*   **"What are the different ways to implement a queue in Python? 🤔"**

## 🧰 Troubleshooting

*   **Ollama Service Not Running:** If you encounter errors related to the Ollama service 🦙, make sure you've run the `!nohup ollama serve &` command in the notebook. You can check if it's running with `!ps aux | grep ollama`.
*   **Model Download Issues:** If the model download 📥 fails or takes too long, check your internet connection 🌐 and try running the `!ollama pull` commands again.
*   **ngrok Authentication:** If you're using ngrok 🚇 and get an authentication error, make sure you've replaced `<YOUR_AUTH_TOKEN>` with your actual ngrok authentication token 🔑.
*   **Streamlit Rerun Issues:** If the chat interface doesn't update automatically after you send a message, try manually refreshing your browser 🔄.

## 🤖 Future Enhancements

*   **Support for More Languages:** Extend support beyond Python 🐍 to other programming languages like JavaScript, Java, C++, etc. 🌎
*   **Code Execution:** Integrate the ability to execute code snippets directly within the app and display the output ▶️.
*   **Context Awareness:** Improve the AI's ability to understand and maintain context 🧠 across multiple turns in a conversation.
*   **Personalization:** Allow users to customize the system prompt and other settings to tailor the AI's behavior to their specific needs 👨‍🔧👩‍🔧.
*   **Integration with IDEs:** Explore the possibility of integrating the DeepSeek Code Companion as a plugin for popular IDEs 🧩.

## 🙏 Acknowledgements

*   **Ollama:** For providing a platform to run LLMs locally 🦙.
*   **LangChain:** For the powerful framework for building language model applications 🦜.
*   **Streamlit:** For the easy-to-use library for creating interactive web apps 🎉.
*   **DeepSeek:** For the impressive DeepSeek Coder models 🧠.
*   **ngrok:** For providing an easy way to expose local servers to the internet 🚇.

## 👨‍💻👩‍💻 Get Involved!

This project is open source, and contributions are welcome! 🎉 If you have ideas for improvements, bug fixes, or new features, feel free to submit issues or pull requests on the project's GitHub repository (if you have one).

Let's make coding more fun and productive with the help of AI! 🥳💻🚀
