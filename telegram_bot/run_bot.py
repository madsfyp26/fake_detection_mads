"""
Run the Telegram bot (same research orchestrator as Streamlit).

  PYTHONPATH=. python -m telegram_bot.run_bot

Requires: TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, and optional Serp/News keys.
"""

from __future__ import annotations

import asyncio
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


async def cmd_start(update, context) -> None:
    msg = update.effective_message
    if not msg:
        return
    await msg.reply_text(
        "Send a question about a claim, topic, or news story. "
        "I pull SerpAPI (web + Google Lens), News API, then summarize with Gemini. "
        "This is not legal advice; sources can be incomplete."
    )


async def cmd_help(update, context) -> None:
    msg = update.effective_message
    if not msg:
        return
    await msg.reply_text("Commands: /start /help — then send any text message.")


async def on_text(update, context) -> None:
    from integrations.research_chat.chat_orchestrator import run_research_turn

    msg = update.effective_message
    if not msg:
        return
    text = (msg.text or "").strip()
    if not text:
        return

    hist = context.user_data.get("history")
    if not isinstance(hist, list):
        hist = []

    loop = asyncio.get_running_loop()
    turn = await loop.run_in_executor(
        None,
        lambda: run_research_turn(text, detection_context=None, history=hist),
    )

    reply = (turn.text or turn.error or "No response.")[:4096]
    hist.append({"role": "user", "content": text})
    hist.append({"role": "assistant", "content": reply})
    context.user_data["history"] = hist[-24:]

    await msg.reply_text(reply)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")

    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    app = (
        Application.builder()
        .token(token)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    print("Telegram bot polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
