# ✨ Crawl4AI Examples: Unleash the Power of Web Crawling! 🕸️

Dive into these amazing examples demonstrating the capabilities of Crawl4AI! 🚀  We'll explore single page crawling, sequential processing, and lightning-fast parallel crawling. Get ready to be impressed! 🤩

### **1. `1-crawl_single_page.py`**  🔍📄

**Purpose:**  Crawls a single web page (a Google search results page in this case) and prints the extracted content in Markdown format.

**Code Breakdown:**

```python
import asyncio  # 🔄 For asynchronous programming
from crawl4ai import *  # 🕷️ Import the Crawl4AI library

async def main():
    async with AsyncWebCrawler() as crawler:  # 🕸️ Create a crawler instance
        result = await crawler.arun(
            url="https://www.google.com/search?q=20-20+voice+cancer+uk"  # 🎯 The URL to crawl
        )
        print(result.markdown)  # 🖨️ Print the Markdown output

if __name__ == "__main__":
    asyncio.run(main())  # ▶️ Run the main function
```

**Emojis in Action:**

*   🔍 **Search:** This script performs a search (on Google) and extracts information.
*   📄 **Single Page:** It focuses on crawling only one specific page.
*   🔄 **Asynchronous:** Uses `asyncio` for efficient, non-blocking operations.
*   🕷️ **Crawler:** The `crawl4ai` library is the star of the show, handling the web crawling.
*   🎯 **Target URL:** Specifies the exact web address to be processed.
*   🖨️ **Print Output:** Displays the result (Markdown content) on the console.
*   ▶️ **Run:** Executes the code.

### **2. `2-crawl_docs_sequential.py`** 📚🚶‍♂️

**Purpose:** Crawls multiple pages from a website (Pydantic AI documentation) sequentially, one after another, and prints the length of the extracted Markdown content for each page.

**Code Breakdown:**

```python
import asyncio  # 🔄 Asynchronous operations
from typing import List  # 📝 Type hinting
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig  # 🕸️ Crawl4AI components
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator  # ✍️ Markdown generator
import requests  # 🌐 For fetching the sitemap
from xml.etree import ElementTree  # 🌳 For parsing XML

async def crawl_sequential(urls: List[str]):
    print("\n=== Sequential Crawling with Session Reuse ===")  # 🚶‍♂️➡️🚶‍♂️ Sequence

    browser_config = BrowserConfig(
        headless=True,  # 👻 Run browser in the background
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],  # ⚙️ Optimization flags
    )

    crawl_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator()  # ✍️ Default Markdown style
    )

    crawler = AsyncWebCrawler(config=browser_config)  # 🕸️ Create the crawler
    await crawler.start()  # ▶️ Start the crawler

    try:
        session_id = "session1"  # 🪪 Reuse session for efficiency
        for url in urls:
            result = await crawler.arun(
                url=url,
                config=crawl_config,
                session_id=session_id
            )
            if result.success:
                print(f"Successfully crawled: {url}")  # ✅ Success message
                print(f"Markdown length: {len(result.markdown_v2.raw_markdown)}")  # 📏 Markdown length
            else:
                print(f"Failed: {url} - Error: {result.error_message}")  # ❌ Error message
    finally:
        await crawler.close()  # ⏹️ Close the crawler

# ... (rest of the code for fetching URLs from sitemap)
```

**Emojis in Action:**

*   📚 **Documentation:** This script targets a documentation website.
*   🚶‍♂️ **Sequential:** Processes pages one by one, in order.
*   🔄 **Asynchronous:** Still uses `asyncio`, but the crawling is sequential within the loop.
*   📝 **Type Hinting:** Uses `List[str]` for better code clarity.
*   🕸️ **Crawl4AI:**  Employs various components like `AsyncWebCrawler`, `BrowserConfig`, and `CrawlerRunConfig`.
*   ✍️ **Markdown Generator:** Specifies how to format the extracted content into Markdown.
*   🌐 **Requests:**  Uses the `requests` library to download the website's sitemap.
*   🌳 **XML Parsing:**  Parses the sitemap (which is in XML format) to extract the URLs.
*   🚶‍♂️➡️🚶‍♂️ **Sequence Indicator:**  Highlights the sequential nature of the crawling process.
*   👻 **Headless Browser:** Runs the browser without a visible window.
*   ⚙️ **Optimization:** Uses browser flags for improved performance, especially in restricted environments (like Docker).
*   🪪 **Session Reuse:** Reuses the browser session to avoid restarting it for each page, making the process faster.
*   ✅ **Success:** Indicates a successful crawl.
*   📏 **Markdown Length:** Reports the length of the generated Markdown for each page.
*   ❌ **Error:** Signals a failure during crawling.
*   ⏹️ **Close:**  Ensures the crawler is properly shut down after processing all URLs.

### **3. `3-crawl_docs_FAST.py`** 📚🚀

**Purpose:** Crawls multiple pages from a website (Pydantic AI documentation) in parallel (concurrently), significantly speeding up the process. It also includes memory usage monitoring.

**Code Breakdown:**

```python
# ... (imports and setup)

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    print("\n=== Parallel Crawling with Browser Reuse + Memory Check ===")  # 🏃‍♂️🏃‍♂️🏃‍♂️ Parallel

    peak_memory = 0  # 📈 Track peak memory usage
    process = psutil.Process(os.getpid())  # 💽 Get current process info

    def log_memory(prefix: str = ""):
        # ... (function to log current and peak memory)

    browser_config = BrowserConfig(
        # ... (minimal browser configuration)
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)  # 🔀 Bypass caching

    crawler = AsyncWebCrawler(config=browser_config)  # 🕸️ Create crawler
    await crawler.start()  # ▶️ Start crawler

    try:
        success_count = 0  # 👍 Successful crawls
        fail_count = 0  # 👎 Failed crawls
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            tasks = []

            for j, url in enumerate(batch):
                session_id = f"parallel_session_{i + j}"  # 🪪 Unique session per task
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append(task)

            log_memory(prefix=f"Before batch {i//max_concurrent + 1}: ")  # 📊 Log memory before

            results = await asyncio.gather(*tasks, return_exceptions=True)  # 🤝 Gather results

            log_memory(prefix=f"After batch {i//max_concurrent + 1}: ")  # 📊 Log memory after

            # ... (evaluate results and print summary)

    finally:
        # ... (close crawler and log final memory usage)

# ... (rest of the code for fetching URLs from sitemap and main function)
```

**Emojis in Action:**

*   📚 **Documentation:** Still targets a documentation website.
*   🚀 **Fast:** This is the key difference - it's designed for speed through parallel processing.
*   🏃‍♂️🏃‍♂️🏃‍♂️ **Parallel:** Processes multiple pages concurrently.
*   📈 **Peak Memory:** Keeps track of the highest memory usage during the process.
*   💽 **Process Info:**  Uses `psutil` to get information about the running process (for memory tracking).
*   📊 **Memory Logging:**  Logs memory usage before and after each batch of URLs.
*   🔀 **Bypass Cache:**  Disables caching to ensure fresh data is fetched each time.
*   👍 **Success Count:** Counts the number of successful crawls.
*   👎 **Fail Count:** Counts the number of failed crawls.
*   🪪 **Unique Session:**  Each concurrent task gets its own session ID.
*   🤝 **Gather Results:**  `asyncio.gather` collects the results from all the parallel tasks.

### **Summary Table**

| Feature         | `1-crawl_single_page.py` | `2-crawl_docs_sequential.py` | `3-crawl_docs_FAST.py` |
| :-------------- | :----------------------- | :--------------------------- | :--------------------- |
| **Purpose**     | Crawl a single page      | Crawl multiple pages sequentially | Crawl multiple pages in parallel |
| **Speed**       | Slow                     | Moderate                     | **Fast**               |
| **Concurrency** | None                     | None                         | **Yes (up to `max_concurrent`)** |
| **Session**     | Single                   | Reused                       | **Unique per task**      |
| **Memory Check**| No                       | No                           | **Yes**                |
| **Caching**     | Default                  | Default                      | **Bypass**             |
| **Emojis**      | 🔍📄🔄🕷️🎯🖨️▶️           | 📚🚶‍♂️🔄📝🕸️✍️🌐🌳🚶‍♂️➡️🚶‍♂️👻⚙️🪪✅📏❌⏹️ | 📚🚀🏃‍♂️🏃‍♂️🏃‍♂️📈💽📊🔀👍👎🪪🤝 |

