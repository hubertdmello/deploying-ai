import json

from __init__ import initialize_openai_client
from prompts import GUARDRAIL_PROMPT, GUARDRAIL_REFUSAL


def check_message(message: str) -> str | None:
    """
    Returns None if the message is safe to process.
    Returns a refusal string if the message violates guardrails.
    """
    try:
        client = initialize_openai_client()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=40,
            messages=[
                {"role": "system", "content": GUARDRAIL_PROMPT},
                {"role": "user", "content": message},
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        if not result.get("safe", True):
            return GUARDRAIL_REFUSAL

        return None

    except Exception:
        # Fail open — if the guardrail check itself errors, let the router handle it
        return None
