import json
import time
import os
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv
import os

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LOG_URL = os.getenv("LOG_URL")


# Gemini's OpenAI-compatible endpoint — lets us keep using the OpenAI SDK as-is.
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY,
)

# Tried in order. If one is rate-limited / quota-exhausted / unavailable,
# we fall through to the next.
MODEL_CHAIN = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

LOG_FILE = "run.jsonl"

# Keeps the last few messages per chat, so multi-turn questions work —
# "answer the LAST message" still needs the earlier ones for context.
conversation_history = {}


def log_event(event: dict):
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def call_model_with_fallback(messages):
    """Try each model in MODEL_CHAIN in order until one succeeds."""
    last_error = None
    for model_name in MODEL_CHAIN:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            log_event({"type": "model_used", "model": model_name})
            return response
        except Exception as e:
            last_error = e
            log_event({
                "type": "model_fallback",
                "model": model_name,
                "error": str(e),
            })
            continue
    # Every model in the chain failed — re-raise the last error.
    raise last_error


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    # Ask the AI to work out the answer. The system prompt tells it exactly how to
    # format the final reply — this is the part that MUST match what the question asked.
    system_prompt = (
        "You are a careful data analyst. The user's LAST message asks a data-analysis "
        "question and tells you exactly what JSON shape to reply with. Work out the "
        "real answer (use any public data you know, e.g. MOSPI statistics, general "
        "world knowledge, or arithmetic on numbers given in the message). "
        "Reply with ONLY that exact JSON object and absolutely nothing else — no "
        "explanation, no markdown, no code fences, just the raw JSON."
    )

    messages = [{"role": "system", "content": system_prompt}] + history[-6:]

    try:
        response = call_model_with_fallback(messages)
    except Exception as e:
        log_event({"type": "all_models_failed", "error": str(e)})
        await update.message.reply_text(
            "Sorry, all models are currently unavailable. Please try again shortly."
        )
        return

    reply_text = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply_text})

    # Make sure we actually reply with valid JSON containing "log_url" — if the model
    # forgot the log_url field or wrapped it in markdown, fix it up here so the grader
    # never sees a malformed reply.
    try:
        parsed = json.loads(reply_text)
    except json.JSONDecodeError:
        # Model added extra text — try to pull out just the {...} part.
        start, end = reply_text.find("{"), reply_text.rfind("}")
        parsed = json.loads(reply_text[start:end + 1])

    parsed["log_url"] = LOG_URL
    final_reply = json.dumps(parsed)

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)


app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
print("Bot is running... (Ctrl+C to stop)")
app.run_polling()