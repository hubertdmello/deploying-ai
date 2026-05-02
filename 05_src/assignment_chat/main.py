"""
Multi-Service Chat App — single conversational interface.
Peter routes each message to the appropriate back-end service:
  - Weather questions  → Open-Meteo API + OpenAI summary
  - Drucker questions  → ChromaDB RAG over 'Managing Oneself'
  - Unit conversions   → LangGraph ReAct agent
  - Anything else      → Peter responds directly
"""

import gradio as gr
from router import peter_chat


def create_app():
    with gr.Blocks(title="Ask Peter", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # Ask Peter

        Peter is an all-in-one scholarly companion.
        He can fetch the **weather** for any city, answer questions about
        Peter Drucker's *Managing Oneself*, and handle **unit conversions** —
        all in one flowing conversation.

        *Try: "What's it like in Lisbon today?" or "How do I find my strengths?"
        or "Convert 70 kg to pounds."*
        """)

        gr.ChatInterface(
            fn=peter_chat,
            type="messages",
        )

    return app
