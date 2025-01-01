# 🍴 **PDF Assistant for Thai Recipes with Groq** 🌟

### 🧑‍💻 **Overview**
This Python application uses **Groq** to build a PDF Assistant that extracts knowledge from PDF files and provides interactive, AI-powered assistance. In this example, it processes a Thai Recipes PDF, making it perfect for developers who want to create custom assistants for various domains! 🚀

---

### 🛠️ **Key Features**
- 📘 **Knowledge Base:** Powered by `PDFUrlKnowledgeBase` to extract and manage data from a Thai recipes PDF.
- ⚡ **Vector Database:** Uses `PgVector2` for embedding and storing data in PostgreSQL for fast, vector-based queries.
- 🗣️ **AI Model:** Leverages `Groq (llama3-groq-70b)` for intelligent and context-aware responses.
- 🛒 **Persistent Storage:** Saves chat history using `PgAssistantStorage`, allowing users to continue sessions seamlessly.
- 🧰 **Command Line Interaction:** Built with **Typer**, enabling an intuitive CLI experience.

---

### 🏗️ **Code Structure**

#### 1️⃣ **Environment Setup**:
```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
```
- Loads environment variables (e.g., `GROQ_API_KEY`).
- Connects to the PostgreSQL database.

#### 2️⃣ **Knowledge Base**:
```python
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="recipes", db_url=db_url),
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview")
)
knowledge_base.load()
```
- Downloads the PDF and processes its content.
- Creates vector embeddings for fast querying.

#### 3️⃣ **Assistant Setup**:
```python
def pdf_assistant(new: bool = False, user: str = "user"):
    run_id = Optional[str] = None
    if not new:
        existing_run_ids = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]
    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )
    assistant.cli_app(markdown=True)
```
- Configures the assistant with optional session continuation.
- Enables features like chat history reading and knowledge base search.

---

### 🚀 **Running the Application**
1. Install dependencies:
   ```bash
   pip install typer dotenv phi
   ```
2. Start the assistant:
   ```bash
   python pdf_assistant.py
   ```

---

### 🌟 **Why This is Awesome**
- 🎯 Focused Knowledge: Extracts and uses knowledge from specific PDFs.
- 🔄 Session Continuity: Resume sessions without losing context.
- 🔍 AI-Powered Search: Easily retrieve data from loaded documents.
- ⚡ High Performance: Combines Groq, PostgreSQL, and vector databases for blazing-fast interaction.

---

### ✨ **Possible Extensions**
- 🌍 Add multilingual support for PDFs.
- 💬 Include chat-based feedback for user refinement.
- 📊 Visualize insights with interactive dashboards.

---

### 🔗 **Links**
- PDF: [Thai Recipes](https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf)
- Groq: [Groq LLM](https://www.groq.com/)
- Typer Docs: [Typer CLI](https://typer.tiangolo.com/)

🚀 *Unleash the power of custom AI assistants with this blueprint!* 🛠️
