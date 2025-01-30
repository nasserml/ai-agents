# 🌟 **JSON-AI-Agent: Your AI Sidekick for JSON and Weather Data** 🌟


This code creates an AI agent that can generate JSON objects (like Pokémon cards!) and interact with external APIs (like a weather API) to provide useful information. It leverages the power of large language models (LLMs) from OpenAI (using the Together API for access) to understand natural language prompts and respond intelligently.

## ✨ **Key Features & How it Works** ✨

1. **JSON Generation** 🤖
    *   **Pokémon Power!** ⚡️: The code starts by defining a `schema` for a Pokémon card (name, type, HP).
    *   **LLM Magic**: It then uses an LLM (Meta-Llama-3.1-70B-Instruct-Turbo) to generate new Pokémon cards in JSON format based on this schema.
    *   **Example**:

        ```json
        {
          "name": "Embermaw",
          "type": "Fire/Dragon",
          "HP": 120
        }
        ```

2. **Tool Use: Expanding the Agent's Abilities** 🛠️
    *   **Function Calling**: The agent is equipped with "tools" – functions it can call to perform specific tasks.
    *   **Defined Tools**:
        *   `calculator`: For math problems (not fully implemented in this example, but the structure is there).
        *   `check_weather`: To fetch weather data.

3. **Weather Wizardry** ☀️🌧️
    *   **Weather API Integration**: The `get_weather` function uses the `weatherapi.com` API to retrieve current weather data for a given city.
    *   **API Key**: You'll need to sign up for a free API key from `weatherapi.com` and store it as `WEATHER_API_KEY` in your Colab environment using `google.colab.userdata`.
    *   **Example Output**:
        ```json
         {
           'location': {'name': 'Karachi', 'region': 'Sindh', 'country': 'Pakistan', ...},
           'current': {'last_updated': '2025-01-30 23:30', 'temp_c': 18.3, 'temp_f': 64.9, 'condition': {'text': 'Mist', ...}, ...}
         }
        ```

4. **Conversational AI: Putting it All Together** 🧠💬
    *   **Chat History**: The `chat_history` list stores the conversation between the user and the AI.
    *   **Interactive Loop**: The `while True` loop is the heart of the agent:
        1. **Prompt the LLM**: Sends the `chat_history` to the LLM (Llama-3.3-70B-Instruct-Turbo-Free) along with the available `tools`.
        2. **Tool Calls**: If the LLM decides to use a tool, it provides the function name and arguments in `tool_calls`.
        3. **Execute Tool**: The code extracts the arguments, calls the appropriate function (e.g., `get_weather`), and appends the result to the `chat_history`.
        4. **LLM Response**: If no tool is called, the LLM's response is directly printed.
    *   **Example Conversation**:
        *   **User**: "What's the weather in France today?"
        *   **Agent** (after calling `check_weather` for Paris): "The weather in Paris, France today is partly cloudy with a temperature of 6.3°C (43.3°F)..."

## 🚀 **Code Highlights with Emojis** 🚀

### **1. Setting up the LLM Client**

```python
from openai import OpenAI
from google.colab import userdata

client= OpenAI(
    api_key=userdata.get('TOGETHER_API_KEY_NEW'),  # 🔑 Your API key from Together
    base_url='https://api.together.xyz/v1'      # 🌐 Together API endpoint
)
```

### **2. Generating a Pokémon Card**

```python
completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", # 🧠 Powerful LLM
    messages=[
        {"role": "user",
         "content": "Generate a pokemon card. use JSON, and awlwys respones in this format: "}], # 📝 Prompt
  response_format={"type": "json_object", "schema": schema} #  ফরম্যাট

)

print(completion.choices[0].message.content) # 🖨️ Print the JSON
```

### **3. Defining Tools**

```python
tools = [{
    "type": "function",
    "function": {
        "name": "calculator", # 🧮 Name of the tool
        "description": "Useful for when you need to answer questions about math.", # ℹ️ Description
        "parameters": { # ⚙️ Parameters for the tool
            "type": "object",
            "properties": {
                "expression": { "type": "string" }
            }
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "check_weather", # 🌦️ Name of the tool
        "description": "Useful for when you need to check the weather!", # ℹ️ Description
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": { "type": "string", "description": "The name of a city NOT a country" } # 🏙️ City parameter
            }
        }
    }
}]
```

### **4. Getting Weather Data**

```python
import requests

def get_weather(city):
  api_key = userdata.get('WEATHER_API_KEY') # 🔑 Your Weather API key
  url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no" # 🌐 API URL
  response = requests.get(url) # 📡 Make the API request
  weather_data = response.json() # ➡️ JSON response
  return weather_data
```

### **5. The Main Loop**

```python
import json

chat_history=[
    {
        "role": "user",
        "content": "What's the weather in France today?"
    }
  ] # 💬 Initial conversation

while True: # 🔁 Keep going until break

  completion = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", # 🧠 LLM model
    messages=chat_history, # 📜 Send the conversation history
    tools=tools # 🛠️ Provide the tools
  )

  tool_calls = completion.choices[0].message.tool_calls # 📞 Check for tool calls

  if tool_calls:
    arguments = tool_calls[0].function.arguments # 📄 Get arguments
    print(arguments)
    name = tool_calls[0].function.name # 🧰 Get tool name
    print(name)

    chat_history.append(completion.choices[0].message) # ➕ Add LLM message to history

    if name == "check_weather": # 🌦️ If it's the weather tool
      city = json.loads(arguments)["city_name"] # 🏙️ Get the city
      weather_data = get_weather(city) # ➡️ Get weather data
      chat_history.append({ # ➕ Add weather data to history
          "role": "function",
          "name": name,
          "content": json.dumps(weather_data)
      })
  else:
    print(completion.choices[0].message.content) # 🖨️ Print the LLM's response
    break # 🛑 Exit the loop
```

## 🎉 **Conclusion** 🎉

This code demonstrates a powerful and flexible way to build AI agents that can interact with the real world through APIs and provide valuable information in a structured format like JSON. You can expand upon this example by adding more tools, refining the prompts, and even integrating it into a larger application. Have fun exploring the possibilities!
