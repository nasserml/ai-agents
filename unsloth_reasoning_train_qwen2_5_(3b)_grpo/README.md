# 🦥 Unsloth Reasoning Trainer: Qwen 2.5 (3B) with GRPO 🚀

Welcome to this exciting journey of training a reasoning model using Unsloth! This Markdown document is based on the Google Colab notebook `unsloth_reasoning_train_qwen2_5_(3b)_grpo/unsloth_reasoning_train_Qwen2_5_(3B)_GRPO.ipynb`. We're going to take the code and transform it into an awesome and educational guide.

## What are We Doing?

We're going to fine-tune the `Qwen/Qwen2.5-3B-Instruct` language model using Unsloth's `FastLanguageModel` and `PatchFastRL`, with a focus on **reasoning** capabilities. We will be leveraging the **GRPO (Generalized Reinforcement learning Policy Optimization)** algorithm.

## Getting Started

To execute this, you just need a **free** Tesla T4 GPU instance on Google Colab. Hit that "*Runtime*" button and select "*Run all*". Simple as that!

<div class="align-center">
<a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
<a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
<a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
</div>

Join us on **[Discord](https://discord.gg/unsloth)** if you need help, and don't forget to give us a ⭐ on **[GitHub](https://github.com/unslothai/unsloth)**! ⭐

## Table of Contents

1.  **[News](#news)**
2.  **[Installation](#installation)**
3.  **[Unsloth Setup](#unsloth-setup)**
4.  **[Model Loading](#model-loading)**
5.  **[Data Preparation](#data-preparation)**
6.  **[Training Configuration](#training-configuration)**
7.  **[Model Training](#model-training)**
8.  **[Inference (Testing the Model)](#inference)**
9. **[Saving](#saving)**
10. **[GGUF / llama.cpp Conversion](#gguf-llamaccpp-conversion)**
---
## News

**Check out our [blog post](https://unsloth.ai/blog/r1-reasoning) for an in-depth guide on training reasoning models!** 🧠

Also be sure to visit all of **[model uploads](https://docs.unsloth.ai/get-started/all-our-models)** and **[notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks)**.

## Installation

First things first, let's get Unsloth and its dependencies installed. The following code will handle the setup for us:

```python
# %%capture
# # Skip restarting message in Colab
# import sys; modules = list(sys.modules.keys())
# for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None
#
# !pip install unsloth vllm
# !pip install --upgrade pillow
# # If you are running this notebook on local, you need to install `diffusers` too
# # !pip install diffusers
# # Temporarily install a specific TRL nightly version
# !pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
```

**Explanation:**

*   `%%capture`: This is an IPython magic command that suppresses output. We use it here because the installation process can be a bit noisy.
*   `!pip install unsloth vllm`: Installs the `unsloth` and `vllm` packages. Unsloth is our star player, and `vllm` provides fast inference capabilities.
*   `!pip install --upgrade pillow`:pillow is an image processing library, and it needs to be updated.
*    `!pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b`: Installs a specific nightly version of the `trl` (Transformer Reinforcement Learning) library from its GitHub repository.

## Unsloth Setup

Now, let's set up Unsloth.
```python
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
```

**Explanation:**

*   `from unsloth import FastLanguageModel, PatchFastRL`: Imports the necessary components from the Unsloth library.
    *   `FastLanguageModel`: A class for quickly loading and fine-tuning language models.
    *   `PatchFastRL`: A utility to patch/modify existing functions for reinforcement learning algorithms.

*   `PatchFastRL("GRPO", FastLanguageModel)`: Patches Unsloth to use the **GRPO** algorithm. It's like giving Unsloth a superpower!

## Model Loading

Time to load our language model, `Qwen/Qwen2.5-3B-Instruct`.

```python
from unsloth import is_bfloat16_supported
import torch
max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)
```
**Explanation:**
*   `max_seq_length = 1024`: Sets the maximum sequence length the model can handle.
*   `lora_rank = 64`:  This controls the rank of the LoRA adapters. A larger rank means a potentially "smarter" model but slower training.
* **Loading the model:**
    *   `model_name = "Qwen/Qwen2.5-3B-Instruct"`: Specifies the pre-trained model we're using.
    *   `max_seq_length = max_seq_length`:  Applies the maximum sequence length.
    *  `load_in_4bit = True`: Loads the model in 4-bit precision, saving memory. Set to `False` for 16-bit LoRA.
    *   `fast_inference = True`: Enables vLLM for speed!
   *  `max_lora_rank = lora_rank`: Sets the maximum LoRA rank.
    *   `gpu_memory_utilization = 0.5`: Controls how much GPU memory Unsloth is allowed to use. Adjust this if you encounter out-of-memory errors.
    *   The function returns a `model` and a `tokenizer`.
        *   `model`:  The loaded language model.
        *   `tokenizer`:  The tokenizer associated with the model, used to convert text into a format the model understands.

Now Add LoRA Adapters:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)
```

**Explanation:**

*   `FastLanguageModel.get_peft_model(...)`:  This function adds LoRA (Low-Rank Adaptation) adapters to the model. LoRA is a technique that makes fine-tuning much more efficient by adding small, trainable "adapters" to the model instead of updating all the original model weights.

*   **Parameters:**

    *   `r = lora_rank`:  The rank of the LoRA adapters. This is a crucial hyperparameter.  Higher rank = more parameters = potentially more expressive power, but also more computationally expensive.
    *   `target_modules`: Specifies which parts of the model the LoRA adapters should be applied to. In this case, it targets the query, key, value, output, gate, up, and down projection layers, which are essential components of the transformer architecture. You can comment some of the, out if you are out of memory.
    *   `lora_alpha`: A scaling factor for the LoRA adapters. It's often set to be equal to the rank.
    *  `use_gradient_checkpointing = "unsloth"`: Enables gradient checkpointing. This trades compute for memory by recomputing some activations during the backward pass instead of storing them.  This is *crucial* for longer sequence lengths.  Unsloth's implementation is much faster than Hugging Face's!
    *  `random_state`: Sets the random state for reproducibility.

## Data Preparation

We'll use a modified version of the GSM8K dataset, which is a set of grade school math problems.

```python
import re
from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
```

**Explanation:**

*   **`SYSTEM_PROMPT`**: Defines the desired output format (reasoning followed by answer).
*   **`XML_COT_FORMAT`**: String template for formatting the output
*   **`extract_xml_answer` and `extract_hash_answer`**: These functions extract the answer from the model's output using different delimiters.
*   **`get_gsm8k_questions`**: Loads the GSM8K dataset, formats the questions into prompts with system and user roles, and extracts the correct answers.
*   **Reward Functions (Crucial for RL!)**:
    *   **`correctness_reward_func`**:  Gives a high reward (2.0) if the extracted answer matches the correct answer, 0.0 otherwise.
    *   **`int_reward_func`**: Gives a small reward (0.5) if the extracted answer is a digit.
    *   **`strict_format_reward_func`**: Gives a reward (0.5) if the output strictly follows the desired XML format with reasoning and answer tags.
    *   **`soft_format_reward_func`**:  A more lenient version of the above, giving a reward (0.5) if the output contains the reasoning and answer tags, even if the spacing isn't perfect.
    *    **`xmlcount_reward_func`**: Assigns rewards to the model when reasoning and answer tags are present in model output.

These reward functions will guide the training process, encouraging the model to produce correct, well-formatted answers with clear reasoning.

## Training Configuration

Now, we configure the `GRPOTrainer` with all the necessary settings:

```python
from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)
```

**Explanation:**

*   **`GRPOConfig`**:  This class holds all the configuration options for the `GRPOTrainer`.
*   **Key Parameters:**
    *   `use_vllm`: Enables vLLM (Very Large Language Model) inference.  This makes generation *much* faster, which is critical for RL training.
    *   `learning_rate`: Controls how quickly the model adapts during training.
    *   `adam_beta1`, `adam_beta2`: Parameters for the AdamW optimizer.
    *   `weight_decay`: Regularization to prevent overfitting.
    *   `warmup_ratio`: Gradually increases the learning rate at the beginning of training.
    *   `lr_scheduler_type`:  Specifies the learning rate schedule (cosine annealing in this case).
    *   `optim`: The optimizer to use (AdamW with 8-bit quantization).
    *   `logging_steps`: How often to print training progress.
    *   `bf16`: Use bfloat16 precision if supported by the hardware.
    *   `fp16`: Fallback to float16 if bfloat16 isn't supported.
    *   `per_device_train_batch_size`: Number of training examples processed per GPU per step.
    *   `gradient_accumulation_steps`: Accumulates gradients over multiple steps to effectively increase batch size.
    *   `num_generations`: Number of responses the model generates per prompt during each training step.
    *   `max_prompt_length`: Maximum length of the input prompt.
    *   `max_completion_length`: Maximum length of the generated response.
    *   `max_steps`:  Maximum number of training steps. We limit it to 250 in this case for demonstration purposes.
    *  `save_steps`: Number of steps after which model will be saved.
    *   `max_grad_norm`: Gradient clipping to prevent exploding gradients.
    *   `report_to`:  Specifies where to report training progress ("none" means no reporting).  You could use "wandb" for Weights & Biases integration.
    *   `output_dir`: Where to save the trained model.

## Model Training

Finally, the main event! Let's train the model:

```python
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()
```
**Explanation:**
*   **`GRPOTrainer`**: This is the core class from the `trl` library that handles the reinforcement learning training loop.
*   **Arguments:**
    *   `model`: The model we loaded and prepared with LoRA.
    *   `processing_class = tokenizer`: The tokenizer.
    *   `reward_funcs`:  A list of our reward functions.  The trainer will use these to evaluate the model's responses.
    *   `args`: The `GRPOConfig` object we created earlier, containing all the training hyperparameters.
    *   `train_dataset`: The prepared dataset.
* **`trainer.train()`**: This line starts the training process. The trainer will:
    1.  Generate responses from the model.
    2.  Evaluate those responses using the reward functions.
    3.  Update the model's parameters (specifically, the LoRA adapters) to improve the rewards.
    4.  Repeat steps 1-3 until the maximum number of steps is reached.

The output of `trainer.train()` will be a training log, which you can inspect to track the training progress. It will be a long list of dictionaries, and the most important metrics to monitor is a high value for the 'reward'.

## Inference

After training, let's see how well our model performs. We'll compare the output of the base model (without GRPO training) and the fine-tuned model.

**Base Model Inference (No Fine-tuning):**
```python
text = tokenizer.apply_chat_template([
    {"role" : "user", "content" : "How many r's are in strawberry?"},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None, # No LoRA for the base model!
)[0].outputs[0].text

print(output)

```

**Explanation:**

1.  **Prompt Formatting:**
    *   `tokenizer.apply_chat_template(...)`: This crucial step formats the input text into the specific chat template expected by the Qwen 2.5 model.  It handles adding the appropriate special tokens (like beginning-of-sequence, end-of-sequence, and role markers) that the model needs to understand the conversation structure.
    *   `tokenize = False`: We *don't* want to tokenize the text yet.  We want the raw string.
    *   `add_generation_prompt = True`: This adds a special token that tells the model to start generating text.

2.  **Sampling Parameters:**
    *   `SamplingParams(...)`:  This class configures how the model generates text.
        *   `temperature = 0.8`: Controls the randomness of the output. Higher values (closer to 1.0) make the output more diverse, while lower values (closer to 0.0) make it more deterministic.
        *   `top_p = 0.95`:  This is for "nucleus sampling."  It restricts the model to choose from the most likely tokens that cumulatively have a probability of at least 0.95.  This helps to avoid generating nonsensical text.
        *   `max_tokens = 1024`:  Sets the maximum length of the generated text.

3.  **Generation:**
    *   `model.fast_generate(...)`: This is Unsloth's optimized generation function (using vLLM).
        *   `[text]`:  The input prompt (as a list, since vLLM can handle batch generation).
        *   `sampling_params`: The sampling parameters we defined.
        *   `lora_request = None`:  Crucially, we set `lora_request` to `None` because we're using the *base* model, *not* the LoRA-adapted model.
    *   `[0].outputs[0].text`: Extracts the generated text from the output.  `fast_generate` returns a list of results (even if you only provide one input), and each result has an `outputs` field, which is a list of generated sequences.

**Fine-Tuned Model Inference (with LoRA):**
```python
model.save_lora("grpo_saved_lora")
text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : "How many r's are in strawberry?"},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"), # Load the LoRA!
)[0].outputs[0].text

print(output)

```

**Explanation**
* **Saving and loading:** First the trained LoRA adapters are saved to the directory "grpo_saved_lora". Then we provide this to be used during inferencing by using the model.load_lora method.

The rest of the code is identical to the base model inference, *except* for the `lora_request` parameter:
*   `lora_request = model.load_lora("grpo_saved_lora")`:  This line is *critical*.  It loads the LoRA adapters that we trained during the GRPO process. This tells the model to use the fine-tuned weights for generation.

By comparing the outputs of the base model and the fine-tuned model, you should see a noticeable improvement in the reasoning and formatting of the responses, thanks to our GRPO training.

## Saving

Unsloth supports saving models in multiple different ways:

```python
# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")
```

Explanation:
*  **Merged Saving**:
        *   `model.save_pretrained_merged(..., save_method="merged_16bit")`:  This saves the *entire* model (base model + LoRA adapters merged together) in float16 precision.  This is useful if you want to load the model later *without* needing Unsloth. You'd load it with the regular `AutoModelForCausalLM.from_pretrained` method.
        *   `model.save_pretrained_merged(..., save_method="merged_4bit")`:  Similar to above, but saves the merged model in 4-bit quantized format. This is even more memory-efficient, but might slightly reduce model quality.
        *   `model.push_to_hub_merged(...)`: This *uploads* the merged model to the Hugging Face Hub.  You'll need a Hugging Face account and a write token (which you can get from your Hugging Face settings).  Replace `"hf/model"` with your desired repository name (e.g., `"your_username/your_model_name"`).
*   **LoRA Adapter Saving:**
     *   `model.save_pretrained_merged(..., save_method="lora")`: This saves *only* the LoRA adapter weights. This is much smaller than saving the entire model.  To load it later, you'll need to load the base model *and* then load the LoRA adapters using `PeftModel`.
     * `model.push_to_hub_merged(..., save_method="lora", token = "")`:Upload to HF, same way as with merged models.

     It is generally recommended to upload just the LoRA parameters.

### GGUF / llama.cpp Conversion

Unsloth now supports conversion to the `GGUF` format, used by `llama.cpp` and other inference engines.
```python
# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",
    )

```
**Explanation:**

*   `model.save_pretrained_gguf(..., quantization_method=...)`:  Converts the model to GGUF format and saves it locally.
    *   `quantization_method`: Specifies the quantization method to use.  Common options include:
        *   `"q8_0"`:  8-bit quantization. A good balance between speed and quality.
        *   `"q4_k_m"`:  4-bit quantization (with some parts kept at higher precision).  A good choice for smaller file sizes.
        *   `"q5_k_m"`:  5-bit quantization.
        *   `"f16"`:  Saves in 16-bit floating-point format (no quantization).
*   `model.push_to_hub_gguf(...)`: Uploads the GGUF model to the Hugging Face Hub.  Again, you'll need a token.

You can then use the generated `.gguf` file with inference engines like `llama.cpp`, or UIs like Jan or Open WebUI.

