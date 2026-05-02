import json

from __init__ import initialize_openai_client
from guardrails import check_message
from weather_service import get_weather_info
from ask_drucker_service import ask_drucker
from smart_converter_service_agent import smart_converter
from prompts import PETER_ROUTER_PROMPT, PETER_FALLBACK_PROMPT


def peter_chat(message: str, history: list) -> str:
    if not message.strip():
        return "I'm here — ask me about the weather, Drucker, or unit conversions."

    # ── Guardrail check (before any routing) ─────────────────────────────
    refusal = check_message(message)
    if refusal:
        return refusal

    try:
        client = initialize_openai_client()

        # ── Step 1: Classify intent ───────────────────────────────────────
        routing_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PETER_ROUTER_PROMPT},
                *history,
                {"role": "user", "content": message},
            ],
            response_format={"type": "json_object"},
            max_tokens=60,
        )
        routing = json.loads(routing_response.choices[0].message.content)
        intent = routing.get("intent", "general")
        param = routing.get("param", message)

        # ── Step 2: Delegate to the right service ─────────────────────────
        if intent == "weather":
            return get_weather_info(param)

        if intent == "drucker":
            return ask_drucker(message, history)

        if intent == "converter":
            return smart_converter(param)

        # ── Step 3: General Peter response ────────────────────────────────
        general_response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=150,
            messages=[
                {"role": "system", "content": PETER_FALLBACK_PROMPT},
                *history,
                {"role": "user", "content": message},
            ],
        )
        return general_response.choices[0].message.content

    except ValueError as e:
        return f"Configuration error: {str(e)}\n\nPlease ensure your OPENAI_API_KEY is set."
    except Exception as e:
        return f"Something went wrong: {str(e)}"
