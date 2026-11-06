"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAPI_API_KEY")
    if not api_key:
        api_key = "mock-key"
    
    client = OpenAI(api_key=api_key)
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time - start_time
    response_text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return response_text, latency, usage
    # TODO: Import OpenAI, instantiate client, call chat.completions.create with parameters,
    #       measure execution start/end time, extract text and token usage, and return them.


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.
    Supports dual-import fallback (new google-genai and legacy google-genarativeai) to ensure zero-friction execution.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "mock-key"
    start_time = time.time
    try:
        #Option A (New Google GenAI SDK):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
            # Configure using types.GenerateContentConfig
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        latency = time.time() - start_time    
        #Option B (Legacy Google GenerativeAI SDK):
        response_text = response.text or ""
        usage = {
            "input_tokens": response.usage_metadata.prompt_token_count or 0,
            "output_tokens": response.usage_metadata.candidates_token_count or 0,
        }
        return response_text, latency, usage

    # TODO: Initialize Gemini client, set config parameters, call generate_content,
    #       measure latency, extract response text and usage metadata, and return the tuple.
    except ImportError:
        # Option B: Legacy Google GenerativeAI SDK (fallback)
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model_inst = genai.GenerativeModel(model)
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )
        response = model_inst.generate_content(
            prompt,
            generation_config=generation_config,
        )
        latency = time.time() - start_time

        response_text = response.text or ""
        usage = {
            "input_tokens": response.usage_metadata.prompt_token_count or 0,
            "output_tokens": response.usage_metadata.candidates_token_count or 0,
        }
        return response_text, latency, usage

# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY") or "mock-key"
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start_time = time.time
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature= temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time - start_time
    response_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    return response_text, latency, usage
    # TODO: Initialize Anthropic client, create message, measure latency,
    #       extract content text and usage statistics, and return the tuple.


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    
    # TODO: Call call_openai with default gpt-4o model
    gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(prompt, model= OPENAI_MODEL)
    gpt4o_cost = (gpt4o_usage["input_tokens"] *PRICING_1M_TOKENS["gpt-4o"]["input"] + gpt4o_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["output"]) / 1_000_000
    # TODO: Call call_openai with gpt-4o-mini model
    mini_text, mini_lat, mini_usage = call_openai(prompt, model="gpt-4o-mini")
    mini_cost = (
        mini_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["input"]+ mini_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000
    # TODO: Call call_gemini with default gemini-2.5-flash model
    gemini_text, gemini_lat, gemini_usage = call_gemini(prompt, model= GEMINI_MODEL)
    gemini_cost = (gemini_usage["input_tokens"] *PRICING_1M_TOKENS["gemini-2.5-flash"]["input"] + gemini_usage["output_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["output"]) / 1_000_000

    # TODO: Calculate costs exactly based on input and output token counts using PRICING_1M_TOKENS
    #       Formula: Cost = (input_tokens * input_rate_per_1M + output_tokens * output_rate_per_1M) / 1,000,000
    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_lat,
            "cost": mini_cost,
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_lat,
            "cost": gemini_cost,
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }
    # TODO: Assemble and return the comparison dictionary


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\033[93m[System Warning] GEMINI_API_KEY environment variable not set. Running in dummy mode.\033[0m")
        api_key = "mock-key"

    print("\n\033[94m==================================================")
    print("🤖 Vin Smart Future — Intelligent Chat Assistant")
    print("Powered by Google Gemini 2.5 Flash")
    print("Type 'quit' or 'exit' to end the session.")
    print("==================================================\033[0m\n")

    # Khởi tạo client (dual-import fallback giống call_gemini)
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(temperature=0.7, top_p=0.9)

        def stream_reply(contents):
            for chunk in client.models.generate_content_stream(
                model=GEMINI_MODEL, contents=contents, config=config
            ):
                if chunk.text:
                    yield chunk.text

    except ImportError:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model_inst = genai.GenerativeModel(GEMINI_MODEL)

        def stream_reply(contents):
            for chunk in model_inst.generate_content(contents, stream=True):
                if chunk.text:
                    yield chunk.text

    history = []   # mỗi turn = 1 tin nhắn user + 1 tin nhắn model
    MAX_TURNS = 3

    while True:
        try:
            user_input = input("\033[92mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        history.append({"role": "user", "parts": [{"text": user_input}]})

        print("\033[96mAssistant:\033[0m ", end="", flush=True)
        reply = ""
        try:
            if api_key == "mock-key":
                # Dummy mode: giả lập streaming, không gọi API thật
                for word in f"(dummy) You said: {user_input}".split():
                    print(word + " ", end="", flush=True)
                    reply += word + " "
                    time.sleep(0.05)
            else:
                for text in stream_reply(history):
                    print(text, end="", flush=True)
                    reply += text
        except Exception as e:
            print(f"\n\033[91m[Error] {e}\033[0m")
            history.pop()   # bỏ tin nhắn user bị lỗi để history không lệch
            continue

        print()
        history.append({"role": "model", "parts": [{"text": reply.strip()}]})

        # Giữ lại 3 turn gần nhất (= 6 tin nhắn)
        history = history[-MAX_TURNS * 2:]
    # TODO: Setup interactive session, prompt user for input, stream response, and update history.
    raise NotImplementedError("Implement streaming_chatbot")


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)

    raise last_exception
    # TODO: implement retry loop with exponential backoff


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    # TODO: iterate over prompts, call compare_models, and inject the original "prompt".
    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    # TODO: Build and return the formatted table string. Truncate response to 50 chars for clean display.
    def clean(text: str, limit: int = 50) -> str:
        # Bỏ xuống dòng và ký tự | để không làm vỡ bảng Markdown
        text = str(text).replace("\n", " ").replace("|", "\\|").strip()
        return text if len(text) <= limit else text[:limit] + "..."

    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]

    for result in results:
        prompt = clean(result.get("prompt", ""))
        for model_name, data in result.items():
            if model_name == "prompt" or not isinstance(data, dict):
                continue
            lines.append(
                f"| {prompt} "
                f"| {model_name} "
                f"| {clean(data.get('response', ''))} "
                f"| {data.get('latency', 0):.2f}s "
                f"| {data.get('input_tokens', 0)}/{data.get('output_tokens', 0)} "
                f"| ${data.get('cost', 0):.6f} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
