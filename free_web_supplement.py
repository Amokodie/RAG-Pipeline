"""
Optional free web supplements for Ask AI (no API keys).
Uses the English Wikipedia MediaWiki API (opensearch + page extracts).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

# Wikimedia asks for a descriptive User-Agent; replace contact if you fork.
_WIKI_UA = "Ass3-RAG-ConceptDemo/1.0 (https://github.com/streamlit; educational use)"


def fetch_wikipedia_intro(query: str, timeout: float = 10.0) -> tuple[str | None, str | None]:
    """
    Return (article_title, plain_text_intro_extract) or (None, None) on failure / no hit.
    """
    q = query.strip()
    if len(q) < 2:
        return None, None

    params = {
        "action": "opensearch",
        "search": q[:240],
        "limit": 1,
        "namespace": 0,
        "format": "json",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _WIKI_UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None, None

    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return None, None
    title = data[1][0]
    if not title:
        return None, None

    p2 = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": "true",
        "explaintext": "true",
        "titles": title,
    }
    url2 = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(p2)
    req2 = urllib.request.Request(url2, headers={"User-Agent": _WIKI_UA})
    try:
        with urllib.request.urlopen(req2, timeout=timeout) as resp2:
            j = json.loads(resp2.read().decode("utf-8"))
    except Exception:
        return title, None

    pages = j.get("query", {}).get("pages", {})
    for _pid, page in pages.items():
        ext = page.get("extract")
        if ext and str(ext).strip():
            text = str(ext).strip()
            if len(text) > 4000:
                text = text[:4000] + "…"
            return title, text
    return title, None


def format_wikipedia_supplement_markdown(title: str | None, extract: str | None) -> str:
    """Markdown block for the chat UI; empty if no extract."""
    if not title or not extract:
        return ""
    safe_title = title.replace(" ", "_")
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(safe_title, safe="()%")
    return (
        "---\n\n**Free supplement — English Wikipedia** "
        "(third-party; **not** your instructor’s official answer — verify for exams)\n\n"
        f"**[{title}]({url})** (lead section)\n\n"
        f"{extract}"
    )


def append_wikipedia_to_answer(answer: str, wiki_md: str) -> str:
    if not wiki_md.strip():
        return answer
    return answer.rstrip() + "\n\n" + wiki_md
