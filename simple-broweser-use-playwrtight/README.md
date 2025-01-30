# 🌐 Simple Browser Automation with Playwright 🤖

This documentation outlines a Jupyter Notebook (`broweser_use_playwright_agent.ipynb`) and its corresponding Python script (`broweser_use_playwright_agent.py`) that showcase a basic web automation agent powered by **Playwright** and a Large Language Model (LLM) from **Together AI**.

## 📝 Notebook: `broweser_use_playwright_agent.ipynb`

### Overview 📖

This notebook demonstrates a simple web interaction scenario using Playwright for browser automation and an LLM for decision-making. The agent navigates to Pinterest, clicks on elements, and captures screenshots, guided by instructions processed through an LLM.


### Cells Breakdown 🧮

#### Cell 1: Install Playwright 🛠️

```python
!pip install playwright
```

**Outputs:**

- Installs the Playwright library.
- Shows installation progress and success messages.

**Purpose:**
Installs the Playwright library which is essential for browser automation.

#### Cell 2: Install Chromium Dependencies ⚙️

```python
!playwright install --with-deps chromium
```

**Outputs:**

- Installs Chromium and its dependencies.
- Displays detailed logs of the installation process.

**Purpose:**
Installs Chromium (a compatible browser) and its dependencies, which Playwright will control.

#### Cell 3: Initialize Playwright and Browser 🚀

```python
from playwright.async_api import async_playwright

p = await async_playwright().start()
browser = await p.chromium.launch(headless=True)
context = await browser.new_context()
page = await context.new_page()
```

**Purpose:**

- Starts Playwright.
- Launches a Chromium browser instance in headless mode (no GUI).
- Creates a new browser context and page for interaction.

#### Cell 4: Navigate to Pinterest 📌

```python
await page.goto("https://www.pinterest.com")
```

**Outputs:**

- Navigates to `https://www.pinterest.com`.
- Returns the response object.

**Purpose:**
Opens the Pinterest homepage in the controlled browser.

#### Cell 5: Capture and Display Screenshot 📸

```python
from io import BytesIO
from PIL import Image as PILImage
from IPython.display import display

screenshot = await page.screenshot()
img = PILImage.open(BytesIO(screenshot))
width, height = img.size
new_width = int(width * 0.5)
new_height = int(height * 0.5)
img = img.resize((new_width, new_height))

display(img)
```

**Outputs:**

- Displays a resized screenshot of the Pinterest homepage.

**Purpose:**

- Captures a screenshot of the current page.
- Resizes the image for display within the notebook.

#### Cell 6: Identify and List Clickable Elements 🖱️

```python
from pprint import pprint

clickable_elements = await page.query_selector_all('a, button, [role="button"], [onclick]')
labeled_elements = dict()
for index, element in enumerate(clickable_elements):
    text = await element.inner_text()
    cleaned_text = " ".join(text.split())
    if text and await element.is_visible():
        labeled_elements[index] = cleaned_text

pprint(labeled_elements)
```

**Outputs:**

- Prints a dictionary of clickable elements with their associated text labels.

**Purpose:**
- Finds all clickable elements (links, buttons) on the page.
- Extracts and cleans their text content for identification.
- Lists the elements that are visible and have text.

#### Cell 7: Click on a Specific Element 🖱️

```python
await clickable_elements[2].click()
```

**Purpose:**
Clicks on the third clickable element (index 2) identified in the previous step (which should be "Explore" on Pinterest).

#### Cell 8: Capture and Display Screenshot After Click 📸

```python
screenshot = await page.screenshot()
img = PILImage.open(BytesIO(screenshot))
width, height = img.size
new_width = int(width * 0.5)
new_height = int(height * 0.5)
img = img.resize((new_width, new_height))

display(img)
```

**Outputs:**
- Displays a resized screenshot of the page after the click action.

**Purpose:**
- Captures and displays a screenshot to show the result of the click action.

#### Cell 9: Fetch Together API Key 🔑

```python
from google.colab import userdata

print(userdata.get('TOGETHER_API_KEY_NEW'))
```

**Outputs:**
- Prints the Together API key from the environment.

**Purpose:**
- Fetches and prints the API key for Together AI, which is used for authentication.

#### Cell 10: Initialize OpenAI Client and Define Tools 🤖

```python
from openai import OpenAI
from google.colab import userdata

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=userdata.get('TOGETHER_API_KEY_NEW')
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "load_page",
            "description": "Go to a webpage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                    }
                },
                "required": [
                    "url"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "click_element",
            "description": "Click on an element by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_id": {
                        "type": "number",
                    }
                },
                "required": [
                    "element_id"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

chat_history = [
    {"role": "user", "content": "Go to Pinterest and find some interesting images"}
]
```

**Purpose:**
- Initializes the OpenAI client with the Together API base URL and API key.
- Defines the tools (`load_page` and `click_element`) that the agent can use.
- Sets up the initial chat history with a user instruction.

#### Cell 11: Interact with LLM and Perform Actions 🧠

```python
completion = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=chat_history,
    tools=tools
)

print(completion.choices[0].message.content)
print(completion.choices[0].message.tool_calls)
```

**Outputs:**

- Prints the LLM's response content.
- Prints the tool calls suggested by the LLM.

**Purpose:**
- Sends the chat history and available tools to the LLM.
- Receives the LLM's response, including suggested actions (tool calls).

#### Cell 12: Define Functions for Interaction 🖱️

```python
import json

clickable_elements = []

async def get_clickable_elements():
    global clickable_elements
    await page.wait_for_load_state()
    clickable_elements = await page.query_selector_all('a, button, [role="button"], [onclick]')
    labeled_elements = dict()

    for index, element in enumerate(clickable_elements):
        text = await element.inner_text()
        cleaned_text = " ".join(text.split())
        if text and await element.is_visible():
            labeled_elements[index] = cleaned_text

    return "The page has loaded and the following element IDs can be clicked " + json.dumps(labeled_elements)

async def load_page(url):
    await page.goto(url)
    return await get_clickable_elements()

async def click_element(element_id):
    await clickable_elements[element_id].click()
    return await get_clickable_elements()
```

**Purpose:**
- Defines asynchronous functions to:
  - `get_clickable_elements()`: Fetches and returns a list of clickable elements on the current page.
  - `load_page(url)`: Navigates to a given URL and then calls `get_clickable_elements()`.
  - `click_element(element_id)`: Clicks on an element identified by its index in the `clickable_elements` list.

#### Cell 13: Main Automation Loop 🔄

```python
import json
from playwright.async_api import async_playwright

p = await async_playwright().start()
browser = await p.chromium.launch(headless=True)
context = await browser.new_context()
page = await context.new_page()

chat_history = [{"role": "user", "content": "Go to Pinterest and find interesting images"}]

while True:
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=chat_history,
        tools=tools
    )

    print(completion.choices[0].message.content)
    print(completion.choices[0].message.tool_calls)
    tool_calls = completion.choices[0].message.tool_calls
    if tool_calls:
        chat_history.append(completion.choices[0].message)
        tool_call_name = tool_calls[0].function.name
        if tool_call_name == "load_page":
            url = json.loads(tool_calls[0].function.arguments)["url"]
            result = await load_page(url)
            chat_history.append({
                "role": "function",
                "name": tool_call_name,
                "content": result
            })
        print("==================Load page")
        pprint(result)
        if tool_call_name == "click_element":
            element_id = json.loads(tool_calls[0].function.arguments)["element_id"]
            result = await click_element(element_id)
            chat_history.append({
                "role": "function",
                "name": tool_call_name,
                "content": result
            })
            print("==================Click element")
            pprint(result)
    else:
        print("No more tools =========")
        break

pprint(chat_history)
```

**Outputs:**

- Prints the LLM's responses and tool calls in each iteration.
- Prints the results of `load_page` and `click_element` actions.
- Prints the complete chat history at the end.

**Purpose:**
- Contains the main loop that drives the agent:
  1. Sends the current chat history to the LLM.
  2. Receives the LLM's response and suggested actions.
  3. If the LLM suggests a tool call:
     - Appends the LLM's message to the chat history.
     - Executes the suggested action (`load_page` or `click_element`).
     - Appends the result of the action to the chat history.
     - Prints the results for debugging.
  4. Continues the loop until the LLM does not suggest any more actions.
- Prints the final chat history.

#### Cell 14: Capture Final Screenshot 📸

```python
screenshot = await page.screenshot()
img = PILImage.open(BytesIO(screenshot))
width, height = img.size
new_width = int(width * 0.5)
new_height = int(height * 0.5)
img = img.resize((new_width, new_height))

display(img)
```

**Outputs:**
- Displays a resized screenshot of the final page state.

**Purpose:**
- Captures and displays a screenshot of the page after all actions have been performed.

### Summary 🚀

The notebook demonstrates a basic browser automation agent that can follow instructions from an LLM to navigate web pages and interact with elements. It uses Playwright to control a Chromium browser, and the Together AI LLM to decide what actions to take based on user instructions. The agent can load pages, click on elements, and provide feedback on the current state of the page.

## 🐍 Python Script: `broweser_use_playwright_agent.py`

This script is essentially a direct translation of the Jupyter Notebook into a standalone Python file. It performs the same actions as the notebook:

1. **Installs Playwright and Chromium.**
2. **Initializes Playwright and launches a browser.**
3. **Navigates to Pinterest.**
4. **Captures and displays an initial screenshot.**
5. **Identifies clickable elements.**
6. **Clicks on a specific element.**
7. **Captures and displays a screenshot after the click.**
8. **Fetches the Together API key (commented out in the script).**
9. **Initializes the OpenAI client.**
10. **Defines the `load_page` and `click_element` tools.**
11. **Enters a loop to interact with the LLM and perform actions.**
12. **Captures and displays a final screenshot.**

### Usage 💻

To use this script, you would:

1. **Install the required libraries:**

    ```bash
    pip install playwright
    playwright install --with-deps chromium
    pip install python-dotenv  # If you are using a .env file for the API key
    pip install openai pillow
    ```
2. **Set up your Together API key:**
    -   Either set the `TOGETHER_API_KEY_NEW` environment variable.
    -   Or replace `userdata.get('TOGETHER_API_KEY_NEW')` in the script with your actual API key.
3. **Run the script:**

    ```bash
    python broweser_use_playwright_agent.py
    ```

### Notes 📒

-   The script assumes you are using Google Colab's `userdata` for managing secrets, but you can adapt it to use environment variables or other methods for storing your API key.
-   The script currently runs the browser in headless mode (`headless=True`). You can change this to `headless=False` to see the browser window during execution.
-   The main loop continues until the LLM does not suggest any more tool calls.
-   The interaction with Pinterest and the specific actions taken are determined by the initial user instruction and the LLM's responses.

