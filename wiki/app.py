import os, json, datetime
from dataclasses import dataclass
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, String, Text, DateTime, select, UniqueConstraint
)
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

# ----- Flask & Sockets -----
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ----- AI config -----
AI_BASE_URL = os.environ.get("AI_BASE_URL", "http://localhost:11434/v1")
AI_API_KEY  = os.environ.get("AI_API_KEY", "ollama")
AI_MODEL    = os.environ.get("AI_MODEL", "gpt-oss")
def ai_client() -> OpenAI:
    return OpenAI(base_url=AI_BASE_URL, api_key=AI_API_KEY)

SYSTEM_PROMPT = (
    "You are a precise lexicographer and technical writer. "
    "Generate a clean, GitHub-flavored Markdown wiki article for a single term. "
    "STRICT MODE: Output ONLY the Markdown article—no preface, no follow-ups, no code fences."
)

# ----- DB (SQLite) -----
DB_URL = os.environ.get("WIKI_DB_URL", "sqlite:///wiki.db")
engine = create_engine(DB_URL, future=True)
metadata = MetaData()

wiki_entries = Table(
    "wiki_entries", metadata,
    Column("id", Integer, primary_key=True),
    Column("term", String(255), nullable=False),
    Column("lang", String(16), nullable=False),
    Column("level", String(32), nullable=False),
    Column("sections_key", String(255), nullable=False),   # canonical string of sections
    Column("content_md", Text, nullable=False),
    Column("created_at", DateTime, default=datetime.datetime.utcnow, nullable=False),
    UniqueConstraint("term", "lang", "level", "sections_key", name="ux_wiki_key"),
)

metadata.create_all(engine)

# ----- Helpers -----
def canon_sections(extras: dict) -> str:
    """Stable key from checkbox selections."""
    true_keys = sorted([k for k, v in (extras or {}).items() if v])
    return ",".join(true_keys)

def build_user_prompt(term: str, lang: str, level: str, extras: dict) -> str:
    sections = []
    if extras.get("pronunciation"): sections.append("Pronunciation (IPA)")
    if extras.get("pos"):           sections.append("Part of Speech & Morphology")
    if extras.get("defs"):          sections.append("Short Definition + Detailed Definition")
    if extras.get("etym"):          sections.append("Etymology")
    if extras.get("syn"):           sections.append("Synonyms & Antonyms (bullets)")
    if extras.get("usage"):         sections.append("Usage Notes / Common Pitfalls")
    if extras.get("examples"):      sections.append("Examples (varied difficulty)")
    if extras.get("translations"):  sections.append("Translations table (en, es, fr, de)")
    if extras.get("related"):       sections.append("Related Terms / See Also")
    section_text = "\n".join(f"- {s}" for s in sections) if sections else "- Short Definition\n- Examples"

    return f"""
Write a comprehensive wiki-style entry for the term: "{term}".
Language: {lang}. Audience reading level: {level}.

Structure:
# {term}
{section_text}

Formatting rules:
- Use GitHub-flavored Markdown. Headings (#, ##, ###), bullet lists, and a simple table for translations if requested.
- Do NOT fabricate citations. If unknown, write '—'.
- Keep it factual, concise, and neutral. No filler.
- Output ONLY the article content (no backticks, fences, or extra commentary).
""".strip()

# ----- Web -----
@app.route("/")
def index():
    return render_template("index.html")

@app.get("/saved")
def list_saved():
    """List saved terms (optional ?q= search)."""
    q = (request.args.get("q") or "").strip().lower()
    with engine.begin() as conn:
        stmt = select(
            wiki_entries.c.id,
            wiki_entries.c.term,
            wiki_entries.c.lang,
            wiki_entries.c.level,
            wiki_entries.c.sections_key,
            wiki_entries.c.created_at
        ).order_by(wiki_entries.c.created_at.desc()).limit(200)
        rows = conn.execute(stmt).fetchall()
    items = []
    for r in rows:
        row = dict(r._mapping)
        if q and q not in row["term"].lower():
            continue
        items.append(row)
    return jsonify({"ok": True, "items": items})

@app.get("/saved/<int:item_id>")
def get_saved(item_id: int):
    with engine.begin() as conn:
        row = conn.execute(
            select(wiki_entries).where(wiki_entries.c.id == item_id)
        ).fetchone()
    if not row:
        return jsonify({"ok": False, "error": "not found"}), 404
    m = dict(row._mapping)
    return jsonify({"ok": True, "item": {
        "id": m["id"], "term": m["term"], "lang": m["lang"],
        "level": m["level"], "sections_key": m["sections_key"],
        "content_md": m["content_md"], "created_at": m["created_at"].isoformat()
    }})

# ----- Socket: generate or reuse -----
@socketio.on("start_wiki")
def start_wiki(data):
    term    = (data or {}).get("term", "").strip() or "diode"
    lang    = (data or {}).get("lang", "en")
    level   = (data or {}).get("level", "general")
    extras  = (data or {}).get("extras", {}) or {}
    use_cache   = bool((data or {}).get("use_cache", True))
    auto_save   = bool((data or {}).get("auto_save", True))
    force_new   = bool((data or {}).get("force_new", False))

    skey = canon_sections(extras)

    # 1) Try cache (unless forced)
    if use_cache and not force_new:
        with engine.begin() as conn:
            row = conn.execute(
                select(wiki_entries.c.content_md)
                .where(wiki_entries.c.term == term)
                .where(wiki_entries.c.lang == lang)
                .where(wiki_entries.c.level == level)
                .where(wiki_entries.c.sections_key == skey)
            ).fetchone()
        if row:
            emit("wiki_chunk", {"text": row[0]})
            emit("wiki_done", {"ok": True, "cached": True})
            return

    # 2) Generate via LLM (stream)
    client = ai_client()
    user_prompt = build_user_prompt(term, lang, level, extras)

    try:
        stream = client.chat.completions.create(
            model=AI_MODEL,
            stream=True,
            temperature=0.4,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        buffer = []
        got_any = False
        for chunk in stream:
            try:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    buffer.append(delta)
                    emit("wiki_chunk", {"text": delta})
                    got_any = True
            except Exception:
                pass

        full_text = "".join(buffer) if got_any else ""

        # 3) Fallback non-stream if provider didn't stream content
        if not got_any:
            resp = client.chat.completions.create(
                model=AI_MODEL,
                stream=False,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            full_text = (resp.choices[0].message.content or "")
            if full_text:
                emit("wiki_chunk", {"text": full_text})

        # 4) Save
        if auto_save and full_text.strip():
            try:
                with engine.begin() as conn:
                    # Upsert-ish: try insert; if conflict, ignore
                    conn.execute(wiki_entries.insert().prefix_with("OR IGNORE").values(
                        term=term, lang=lang, level=level, sections_key=skey,
                        content_md=full_text.strip(),
                        created_at=datetime.datetime.utcnow(),
                    ))
            except SQLAlchemyError:
                pass

        emit("wiki_done", {"ok": True, "cached": False})

    except Exception as e:
        emit("wiki_done", {"ok": False, "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
