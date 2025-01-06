# 🎥 Phidata Video AI Summarizer Agent

Welcome to the **Phidata Video AI Summarizer Agent**! This powerful tool leverages cutting-edge AI models to analyze videos and provide insightful summaries and answers to your queries. 🚀

---

## 🌟 Features

- **Upload & Analyze**: Easily upload video files (MP4, MOV, AVI) for AI analysis.
- **AI-Powered Insights**: Uses Google's **Gemini 2.0 Flash exp** model for advanced video content understanding.
- **Interactive Queries**: Ask specific questions about the video content and get detailed, actionable responses.
- **Web Research Integration**: Enhances video analysis with supplementary web research via DuckDuckGo. 🌐
- **User-Friendly Interface**: Built with Streamlit for a clean and interactive user experience.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/nasserml/ai-agents.git
cd "Video Summarizer"
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory and add your Google Generative AI API key:

```dotenv
GOOGLE_API_KEY=your_google_api_key_here
```

> **Note:** Ensure you have access to the Google Generative AI API and have your API key ready.

---

## 🚀 Running the Application

### For the Enhanced Version (`app-enhanced.py`)

```bash
streamlit run app-enhanced.py
```

### For the Basic Version (`app.py`)

```bash
streamlit run app.py
```

---

## 📖 Usage Guide

### 1. Upload a Video

- Click on the **"Browse files"** button or drag and drop your video file.
- Supported formats: **MP4**, **MOV**, **AVI**.

### 2. Ask a Question

- In the **text area**, type in any questions or insights you're seeking about the video content.
  - *Example:* "What are the key takeaways from this presentation?"
- Be as specific as possible to get the best results.

### 3. Analyze the Video

- Click on the **"Analyze Video"** button.
- The app will process your video and generate insights based on your query.
- **Please wait** while the AI processes the video. This may take a few minutes depending on the video length and complexity.

### 4. View Results

- The **Analysis Result** section will display the AI-generated response.
- The response is designed to be **detailed**, **user-friendly**, and **actionable**.

---

## 🧠 How It Works

1. **Video Upload & Processing**

   - The uploaded video is temporarily stored and sent to Google's Generative AI services for processing.
   - The app uses `google.generativeai` library to upload and manage video files.
   - The enhanced version uses asynchronous processing to handle large files efficiently. ⏱️

2. **AI Analysis**

   - An **Agent** is initialized with the **Gemini 2.0 Flash exp** model.
   - The agent is also equipped with the **DuckDuckGo** tool for supplementary web research.
   - The user's query is combined with a predefined prompt to guide the AI's response.

3. **Generating Insights**

   - The AI analyzes the video content in the context of the user's query.
   - It integrates any necessary external information via web search to provide a comprehensive answer.
   - The final response is formatted in Markdown for readability.

---

## 🛡️ Error Handling & Timeouts

- The enhanced app includes robust error handling to manage:
  - **Video Processing Timeouts**: If processing takes too long, the app will notify you.
  - **Invalid Inputs**: Warnings are displayed for missing video files or queries.
  - **Exceptions**: Any unexpected errors during analysis are caught and displayed.

---

## 📝 Code Overview

### `app-enhanced.py` Highlights

- **Asynchronous Video Processing**

  ```python
  async def process_video_async(video_path):
      # Asynchronously handles video processing with exponential backoff
  ```

- **Caching & Performance**

  - Uses `@st.cache_resource` and `@lru_cache` for efficient resource management.

- **Customizable Prompts**

  ```python
  def get_analysis_prompt(query):
      return f"""
      Analyze the uploaded video for content and context.
      Respond to the following query using video insights and supplementary web research:
      {query.strip()}
      
      Provide a detailed, user-friendly, and actionable format.
      """
  ```

### `app.py` Highlights

- **Synchronous Video Processing**

  - Simpler processing flow suitable for smaller applications or quick setups.

- **Direct Agent Interaction**

  ```python
  response = multimodel_Agent.run(analysis_prompt, videos=[processed_video])
  ```

---

## 🎨 Customizations

- **Adjust Text Area Height**

  ```markdown
  <style>
  .stTextArea textarea {
      min-height: 100px;
  }
  </style>
  ```

- **Page Configuration**

  ```python
  st.set_page_config(
      page_title="Video Summarizer",
      page_icon="🎥",
      layout="wide"
  )
  ```

---

## 💡 Tips for Best Results

- **Be Specific**: The more detailed your question, the better the AI can tailor its response.
- **Video Quality**: Higher quality videos with clear audio and visuals yield better analysis.
- **Patience is Key**: Processing may take time depending on the video size and complexity.

---

## 🤝 Contributing

We welcome contributions! Feel free to submit issues or pull requests.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📧 Contact

For any questions or support, please reach out to [mnasserone@gmail.com](mailto:mnasserone@gmail.com).

---

## 🚧 Disclaimer

- **Privacy**: Uploaded videos are processed by Google's Generative AI services. Ensure you have the right to upload and process the content.
- **Limits**: The AI's responses are as good as the underlying models and data. Always verify critical information independently.

---

Thank you for using the Phidata Video AI Summarizer Agent! 🎉

> *Empowering you with AI insights.* 🤖