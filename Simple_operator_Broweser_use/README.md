# 🖥️ **OS-Copilot: Your AI-Powered Desktop Navigator** 🖥️

This project brings to life an AI agent capable of interacting with your computer's desktop environment. Leveraging advanced language models and computer vision, OS-Copilot can understand your instructions, view the desktop, and perform actions like clicking. 🖱️✨

## 🚀 **Project Overview**

OS-Copilot is built using state-of-the-art tools like `e2b-code-interpreter` for creating sandboxed environments and `Groq`'s API for accessing powerful language models. It demonstrates the potential of AI to understand and interact with graphical user interfaces.

## 🛠️ **Setup and Installation**

### **Prerequisites**

Before we dive in, make sure you have the following API keys set up as environment variables:

*   `E2B_API_KEY`: For accessing E2B's sandbox environment.
*   `GROQ_API_KEY`: For utilizing the Groq API.
*   `HuggingFace Token`

These keys will allow our AI agent to work its magic. 🔑

### **Installation**

First, let's get our tools ready. We'll start by installing the necessary Python packages:

```bash
!pip install e2b-code-interpreter
!pip install gradio_client
```

This command installs:

*   `e2b-code-interpreter`: Enables the creation of a secure, sandboxed environment for code execution.
* `gradio_client` : Enables the creation of a user interface.

## 🧠 **Integrating with Language Models**

### **E2B Sandbox Setup**

We initialize a sandboxed desktop environment using E2B Code Interpreter. This allows us to safely execute commands and interact with a virtual desktop.

```python
from e2b_code_interpreter import Sandbox
from google.colab import userdata

sbx = Sandbox(
    api_key=userdata.get('E2B_API_KEY'),
    template="desktop"
)
```

### **Groq API Integration**

We'll use Groq's API to access a powerful language model for processing instructions and understanding screenshots.

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=userdata.get("GROQ_API_KEY")
)
```

## 📸 **Capturing and Displaying Screenshots**

Here's how we capture the current state of the desktop and display it:

```python
screenshot_path = "screenshot.png"
sbx.commands.run(f"scrot --pointer {screenshot_path}")
screenshot = sbx.files.read(screenshot_path, format="bytes")
sbx.files.remove(screenshot_path)

from io import BytesIO
from PIL import Image as PILImage
from IPython.display import display

img = PILImage.open(BytesIO(screenshot))
width, height = img.size
new_width = int(width * 0.5)
new_height = int(height * 0.5)
small_img = img.resize((new_width, new_height))
display(small_img)
img.save("screenshot.png")
```

This code snippet captures a screenshot using `scrot`, reads the image file, and then displays a resized version of it.

## 🖱️ **Simulating Mouse Actions**

We can simulate mouse movements and clicks using `xdotool`:

```python
sbx.commands.run("xdotool mousemove --sync 550 750")
sbx.commands.run("xdotool click 1")
```

## 🤖 **AI Agent: The OS-Copilot**

### **Defining Tool Specifications**

We define the tools available to our AI agent. For now, it's just a simple mouse click.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "click_mouse",
        "description": "Click on an item on the screen.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A description of what you want to click on"
                }
            },
            "required": [
                "query"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]
```

### **Processing Instructions with the Language Model**

We send the screenshot and the user's objective to the language model and get back instructions.

```python
import base64
base64_screenshot = base64.b64encode(screenshot).decode('utf-8')
completion = client.chat.completions.create(
    model="llama-3.2-90b-vision-preview",
    messages=[
        {"role": "user", "content": [
            {"type":"image_url", "image_url":{"url": f"data:image/jpeg;base64,{base64_screenshot}"}},
            {"type":"text", "text":"Here are the contents of the screen."},
            {"type":"text", "text":"The objective is: Check the weather using the web browser?"},
        ]}
    ],
    tools=tools
)

print(completion.choices[0].message.content)
print(completion.choices[0].message.tool_calls)
```

### **Using Gradio for Interaction**

We integrate with Gradio to create a user interface for interaction. (Note: The code provided seems to have an issue with the `handle_file` function in the context of Google Colab. You might need to adapt this part based on how you intend to run the code.)

```python
from gradio_client import Client

OSATLAS_HUGGINGFACE_SOURCE = "maxiw/OS-ATLAS"
OSATLAS_HUGGINGFACE_MODEL = "OS-Copilot/OS-Atlas-Base-7B"
OSATLAS_HUGGINGFACE_API = "/run_example"
huggingface_client = Client(OSATLAS_HUGGINGFACE_SOURCE)
```

### **Main Loop: Capturing, Processing, and Acting**

This loop continuously captures the screen, processes instructions, and performs actions.

```python
import base64
import json
import time
from io import BytesIO
from PIL import Image as PILImage
from IPython.display import display

while True:
    # Getting screen shot
    screenshot_path = "screenshot.png"
    sbx.commands.run(f"scrot --pointer {screenshot_path}")
    screenshot = sbx.files.read(screenshot_path, format="bytes")
    sbx.files.remove(screenshot_path)

    # Display the screenshot (resized for convenience)
    img = PILImage.open(BytesIO(screenshot))
    width, height = img.size
    new_width = int(width * 0.5)
    new_height = int(height * 0.5)
    small_img = img.resize((new_width, new_height))
    display(small_img)
    img.save("screenshot.png")

    # Encode the screenshot to base64
    base64_screenshot = base64.b64encode(screenshot).decode('utf-8')

    # Send the screenshot and objective to the language model
    completion = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_screenshot}"}},
                {"type": "text", "text": "Here are the contents of the screen."},
                {"type": "text", "text": "The objective is: Check the weather using the web browser?"},
            ]}
        ],
        tools=tools
    )

    print(completion.choices[0].message.content)
    print(completion.choices[0].message.tool_calls)

    tool_calls = completion.choices[0].message.tool_calls

    # Act on the tool calls from the language model
    if tool_calls:
        tool_call = tool_calls[0]
        function_name = tool_call.function.name

        if function_name == "click_mouse":
            query = json.loads(tool_call.function.arguments)["query"]
            print("Query=========")
            print(query)
            result_x, result_y = get_coordinates(query)
            sbx.commands.run(f"xdotool mousemove --sync {result_x} {result_y}")
            sbx.commands.run("xdotool click 1")
            time.sleep(5)  # Wait for the action to take effect
        else:
            print("No tool called")
            break
```

## **Function: Get Coordinates for Clicking**

```python
import re

def get_coordinates(query):
    """
    Uses an external service (Gradio Client) to determine the coordinates
    to click on based on the given query and the current screenshot.
    """
    result = huggingface_client.predict(
        image=handle_file("screenshot.png"),
        text_input=query + "\nReturn the response in the form of a bbox",
        model_id=OSATLAS_HUGGINGFACE_MODEL,
        api_name=OSATLAS_HUGGINGFACE_API,
    )
    print(result)

    numbers = [float(number) for number in re.findall(r"\d+\.\d+", result[1])]
    # x1, y1, x2, y2

    result_x, result_y = (numbers[0] + numbers[2]) // 2, (numbers[1] + numbers[3]) // 2
    return result_x, result_y
```

## 🔮 **Future Enhancements**

*   **Improved Error Handling**: Implement more robust error handling and retries for API calls.
*   **More Actions**: Expand the range of actions the agent can perform (e.g., typing, scrolling).
*   **Context Awareness**: Enhance the agent's ability to maintain context across interactions.
*   **User Interface**: Develop a more interactive and user-friendly interface.

## 🌟 **Conclusion**

OS-Copilot represents a significant step towards creating AI agents that can interact with our digital environments in a more intuitive and natural way. By combining the power of language models with the flexibility of a sandboxed desktop environment, we're paving the way for more seamless human-computer interaction.

Let's keep exploring and pushing the boundaries of what's possible with AI! 🌌🚀
