# -*- coding: utf-8 -*-
"""ReAct-ai-agent.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cwm8E5fx8g1yojh-SdMF9PpNIvoQ-z0p
"""

# Define the calculator tool

from sympy import sympify
def calculator(expression):
    try:
        result = sympify(expression)
        return result
    except Exception as e:
        return f"Error in calculation: {e}"

calculator("1*2*3*4*5")

import re

def extract_action_and_input(text):
  action = re.search(r"Action: (.*)", text)
  action_input = re.search(r"Action Input: (.*)", text)
  return action.group(1).strip() if action else None, action_input.group(1).strip() if action else None

extract_action_and_input("""
Thought: To calculate the square root of 144, I can use the math library in Python or a calculator.

Action: Calculator
Action Input: sqrt(144)
""")

from google.colab import userdata

from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=userdata.get('OPENROUTER_API_KEY'),
)

user_prompt = "Calculate the square root of 144"

chat_history = [
    {
      "role": "system",
      "content": """
      You have access to the following tools:
      Calculator: Use this when you want to do math. Use SymPy expressions, eg: 2 + 2

      Use the following format:

      Question: the input question you must answer
      Thought: you should always think about what to do
      Action: the action to take, should be one of [Calculator]
      Action Input: the input to the action
      Observation: the result of teh action
      ... (the Thought/Action/Observation can repeat any number of times)
      Thought: I now know the final answer!
      Final Answer: the answer to the original input question
      """
    },
    {
      "role": "user",
      "content": f"Question: {user_prompt}"
    }
  ]

import re

while True:
  completion = client.chat.completions.create(
    model="meta-llama/llama-3.2-90b-vision-instruct:free",
    messages=chat_history,
    stop=["Observation:"]
  )
  response_text = completion.choices[0].message.content
  print("==="*50)
  print(response_text)
  print("==="*50)

  action, action_input = extract_action_and_input(response_text)
  # We want to see if the LLM took an action
  if action == "Calculator":
    action_result = calculator(action_input)
    print(f"Observation: {action_result}")
    chat_history.extend([
      { "role": "assistant", "content": response_text },
      { "role": "user", "content": f"Observation: {action_result}" }
    ])
  else:
    break