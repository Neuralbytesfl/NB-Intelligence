# 📚 Dictionary → Wiki Page (Flask + gpt-oss)

Generate clean, wiki-style Markdown pages for dictionary terms using a local or OpenAI-compatible LLM (default: **gpt-oss**).  
Features live streaming to the browser, strict “Markdown-only” output, and **auto-save** to SQLite so you don’t regenerate the same term twice.

---

## ✨ Features

- **Streaming generation** over WebSockets (Flask-SocketIO + eventlet)
- **Strict mode**: model outputs *only the article Markdown* (no fences, no chatter)
- **Autosave cache**: (term + language + level + sections) → **unique key**
- **Instant reuse**: load existing articles from the cache automatically
- **Saved list**: browse/search most recent entries
- **Copy / Download**: copy Markdown to clipboard or download `.md`
- **Configurable**: choose sections (pronunciation, POS, etymology, examples, etc.)

---

## 🧱 Tech Stack

- **Backend**: Flask, Flask-SocketIO, eventlet
- **LLM Client**: OpenAI Python SDK (pointable to any *OpenAI-compatible* server)
- **DB**: SQLite via SQLAlchemy (table: `wiki_entries`)
- **Frontend**: Socket.IO + Marked.js (GFM Markdown renderer)

---

## 📦 Requirements

```txt
Flask==3.0.3
Flask-SocketIO==5.3.6
eventlet==0.36.1
openai>=1.35.7
python-dotenv==1.0.1
SQLAlchemy==2.0.32
