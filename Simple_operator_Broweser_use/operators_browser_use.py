# -*- coding: utf-8 -*-
"""operators-browser-use.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Gdsh7n-QkX_s6IMKoqE-Akr8L1htpOeI
"""

!pip install e2b-code-interpreter

from google.colab import userdata
print(userdata.get('E2B_API_KEY'))

from e2b_code_interpreter import Sandbox
from google.colab import userdata

sbx = Sandbox(
    api_key=userdata.get('E2B_API_KEY'),
    template="desktop"
) # by default sandbox is  a live for 5 minutes

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

sbx.commands.run("xdotool mousemove --sync 550 750")

sbx.commands.run("xdotool click 1")

import base64
print(base64.b64encode(screenshot).decode('utf-8'))

print(userdata.get("GROQ_API_KEY"))

from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=userdata.get("GROQ_API_KEY")
)
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

!pip install gradio_client

from gradio_client import Client, handle_file

OSATLAS_HUGGINGFACE_SOURCE = "maxiw/OS-ATLAS"
OSATLAS_HUGGINGFACE_MODEL = "OS-Copilot/OS-Atlas-Base-7B"
OSATLAS_HUGGINGFACE_API = "/run_example"
huggingface_client = Client(OSATLAS_HUGGINGFACE_SOURCE)

result = huggingface_client.predict(
    image=handle_file("screenshot.png"),
    text_input="Click on the Web Broweser" + "\nReturn the response in the form of a box",
    model_id=OSATLAS_HUGGINGFACE_MODEL,
    api_name=OSATLAS_HUGGINGFACE_API
)
print(result)

result[1]

import re

numbers = [float(number) for number in  re.findall(r"\d+.\d+", result[1])]
# x1, y1, x2, y2

result_x, result_y = (numbers[0] + numbers[2])//2 , (numbers[1] + numbers[3])//2
result_x, result_y

sbx.commands.run(f"xdotool mousemove --sync {result_x} {result_y}")
sbx.commands.run("xdotool click 1")

import re

def get_coordinates(query):
  result = huggingface_client.predict(
      image=handle_file("screenshot.png"),
      text_input= "Web browser" + "\nReturn the response in the form of a bbox",
      model_id=OSATLAS_HUGGINGFACE_MODEL,
      api_name=OSATLAS_HUGGINGFACE_API,
  )
  print(result)

  numbers = [float(number) for number in re.findall(r"\d+\.\d+", result[1])]
  # x1, y1, x2, y2

  result_x, result_y = (numbers[0] + numbers[2]) // 2, (numbers[1] + numbers[3]) // 2
  return result_x, result_y

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



  img = PILImage.open(BytesIO(screenshot))
  width, height = img.size
  new_width = int(width * 0.5)
  new_height = int(height * 0.5)
  small_img = img.resize((new_width, new_height))
  display(small_img)
  img.save("screenshot.png")

  base64_screenshot = base64.b64encode(screenshot).decode('utf-8')

  # Asking the LLM
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

  tool_calls = completion.choices[0].message.tool_calls

  # running the tool
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
      time.sleep(5)
    else:
      print("No tool called")
      break