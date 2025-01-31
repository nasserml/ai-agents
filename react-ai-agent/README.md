# 🤖 ReAct AI Agent: Math Whiz 🧠

This project demonstrates a **ReAct (Reasoning and Acting)** AI agent that can solve mathematical problems using a simple calculator tool and the power of a Large Language Model (LLM). Let's break down how it works! ✨

## 🧰 Tool Definition: The Calculator

First, we define a `calculator` function. This is the tool our AI agent will use to perform calculations. It leverages the `sympy` library for symbolic mathematics, allowing it to handle a wide range of expressions.

```python
from sympy import sympify

def calculator(expression):
    try:
        result = sympify(expression)
        return result
    except Exception as e:
        return f"Error in calculation: {e}"
```

**Example Usage:**

```python
calculator("1*2*3*4*5")  # Output: 120
```

## 🤔 Extracting Actions and Inputs

The AI agent communicates its intentions using a specific format. We need a way to extract the intended **action** (e.g., "Calculator") and the **action input** (e.g., "sqrt(144)") from the agent's response.

Here's the `extract_action_and_input` function using regular expressions to do just that:

```python
import re

def extract_action_and_input(text):
  action = re.search(r"Action: (.*)", text)
  action_input = re.search(r"Action Input: (.*)", text)
  return action.group(1).strip() if action else None, action_input.group(1).strip() if action else None

extract_action_and_input("""
Thought: To calculate the square root of 144, I can use the math library in Python or a calculator.

Action: Calculator
Action Input: sqrt(144)
""")  # Output: ('Calculator', 'sqrt(144)')
```

## 🚀 Setting Up the LLM

We're using the `OpenAI` client to interact with an LLM hosted on OpenRouter. The `OPENROUTER_API_KEY` is used for authentication.

```python
from google.colab import userdata

from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=userdata.get('OPENROUTER_API_KEY'),
)
```

## 💬 Defining the Conversation

We set up a `chat_history` to guide the LLM. This includes:

*   **System Prompt:** ⚙️ This sets the stage, defining the available tools and the expected format of interaction.
*   **User Prompt:** 🙋‍♂️ This is the initial question we want the agent to answer.

```python
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
```

## 🔄 The ReAct Loop

Now for the heart of the agent! ❤️ This `while` loop orchestrates the interaction between the LLM and the calculator tool:

1. **Get LLM Response:** The LLM generates a response based on the `chat_history`.
2. **Extract Action:** We use `extract_action_and_input` to determine if the LLM wants to use the calculator.
3. **Perform Action:** If the action is "Calculator", we call the `calculator` function with the provided input.
4. **Update History:** We add the LLM's response and the observation (result of the action) to the `chat_history`.
5. **Repeat:** The loop continues until the LLM provides a "Final Answer".

```python
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
```

## 🎉 Result

When you run this code, you'll see the agent thinking through the problem, using the calculator, and finally arriving at the correct answer! It's like having a little math buddy inside your computer! 🥳

```
======================================================================================================================================================
Thought: To find the square root of 144, I can use a mathematical operation.

Action: Calculator
Action Input: sqrt(144)

======================================================================================================================================================
Thought: To find the square root of 144, I can use a mathematical operation.

Action: Calculator
Action Input: sqrt(144)

Observation: 12
======================================================================================================================================================
Thought: The calculator has given me the result of the square root operation, which is 12. This means that the square root of 144 is indeed 12, since 12 * 12 = 144.

Thought: I now know the final answer!

Final Answer: 12
======================================================================================================================================================
Thought: The calculator has given me the result of the square root operation, which is 12. This means that the square root of 144 is indeed 12, since 12 * 12 = 144.

Thought: I now know the final answer!

Final Answer: 12
```

## 🌟 Conclusion

This project demonstrates the core principles of a ReAct AI agent. By combining the reasoning abilities of LLMs with simple tools, we can create agents that can solve problems in a step-by-step manner. This is a powerful paradigm with many potential applications!

**Future Enhancements:**

*   🤯 **More Tools:** Add more tools (e.g., a search engine, a code execution tool) to make the agent even more capable.
*   🧠 **Better Prompting:** Experiment with different system prompts to improve the agent's reasoning and decision-making.
*   🤖 **More Complex Tasks:** Challenge the agent with more complex problems that require multiple steps and tools.

