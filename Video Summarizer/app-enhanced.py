import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
from google.generativeai import upload_file, get_file
import time
import tempfile
from dotenv import load_dotenv
import os
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables once at startup
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini API
if not API_KEY:
    raise ValueError("API key for Google Generative AI is missing. Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

# Constants
MAX_WAIT_TIME = 64
INITIAL_WAIT_TIME = 1
SUPPORTED_VIDEO_FORMATS = ["mp4", "mov", "avi"]

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True
    )

@lru_cache(maxsize=32)
def get_analysis_prompt(query):
    return f"""
    Analyze the uploaded video for content and context.
    Respond to the following query using video insights and supplementary web research:
    {query.strip()}
    
    Provide a detailed, user-friendly, and actionable format.
    """

async def process_video_async(video_path):
    try:
        processed_video = upload_file(video_path)
        wait_time = INITIAL_WAIT_TIME
        total_wait = 0

        while processed_video.state.name == "PROCESSING":
            await asyncio.sleep(wait_time)
            total_wait += wait_time
            
            if total_wait >= MAX_WAIT_TIME:
                return None
                
            processed_video = get_file(processed_video.name)
            wait_time = min(wait_time * 2, MAX_WAIT_TIME)
            
        return processed_video
    except Exception as e:
        st.error(f"Video processing error: {e}")
        return None

def process_video(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        video_path = temp_video.name
        
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        processed_video = loop.run_until_complete(process_video_async(video_path))
        return processed_video
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
        loop.close()

def main():
    st.title("Phidata Video AI Summarizer Agent 🎥")
    st.header("Powered by Gemini 2.0 Flash exp")
    
    multimodel_agent = initialize_agent()
    
    video_file = st.file_uploader(
        "Upload a video file",
        type=SUPPORTED_VIDEO_FORMATS,
        help="Upload a video for AI analysis"
    )
    
    if not video_file:
        st.info("Please upload a video file to analyze.")
        return
        
    st.video(video_file)
    
    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
        help="Provide specific questions or insights you want from the video."
    )
    
    if not st.button("Analyze Video", key="analyze_video_button"):
        return
        
    if not user_query.strip():
        st.warning("Please enter a question or insight to analyze the video.")
        return
        
    with st.spinner("Processing video and generating insights..."):
        try:
            with ThreadPoolExecutor() as executor:
                video_bytes = video_file.read()
                processed_video = executor.submit(process_video, video_bytes).result()
                
            if processed_video is None:
                st.error("Failed to process the video.")
                return
                
            analysis_prompt = get_analysis_prompt(user_query)
            response = multimodel_agent.run(analysis_prompt, videos=[processed_video])
            
            st.subheader("Analysis Result")
            st.markdown(response.content)
            
        except Exception as error:
            st.error(f"Analysis error: {error}")

if __name__ == "__main__":
    main()
    
    # Custom CSS
    st.markdown(
        """
        <style>
        .stTextArea textarea { min-height: 100px; }
        </style>
        """,
        unsafe_allow_html=True
    )