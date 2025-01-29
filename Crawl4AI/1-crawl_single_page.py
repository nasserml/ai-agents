import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.google.com/search?q=20-20+voice+cancer+uk"
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())