"""
semantic.py - Embedding-based semantic search over menu items.

This is the "retrieval" half of the pipeline. Instead of matching the letters
of a word (so "meat" misses "Beef Bulgogi"), we convert each menu item AND the
user's request into embedding vectors that capture *meaning*, then rank items by
cosine similarity. "meat" then surfaces bulgogi, souvlaki and pork; "something
light" surfaces salads and grain bowls - even when those exact words never
appear on the menu.

Embeddings are cached on disk keyed by the item text, so the same dish is only
ever embedded once (repeat lookups are free and instant).
"""

import json
import math
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EXPAND_MODEL = "gpt-4o-mini"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "embeddings_cache.json")
QUERY_CACHE_PATH = os.path.join(os.path.dirname(__file__), "query_cache.json")

_client = None
_cache = None
_qcache = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _load_cache():
    global _cache
    if _cache is None:
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        except (FileNotFoundError, ValueError):
            _cache = {}
    return _cache


def _save_cache():
    if _cache is not None:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_cache, f)


def embed_texts(texts):
    """
    Return an embedding vector for each text, using the disk cache for any we've
    seen before and making a single batched API call for the rest.
    """
    cache = _load_cache()
    missing = [t for t in texts if t not in cache]

    if missing:
        # De-dupe before sending; one batched request for all new items.
        unique_missing = list(dict.fromkeys(missing))
        resp = _get_client().embeddings.create(model=EMBED_MODEL, input=unique_missing)
        for text, item in zip(unique_missing, resp.data):
            cache[text] = item.embedding
        _save_cache()

    return [cache[t] for t in texts]


def _load_qcache():
    global _qcache
    if _qcache is None:
        try:
            with open(QUERY_CACHE_PATH, "r", encoding="utf-8") as f:
                _qcache = json.load(f)
        except (FileNotFoundError, ValueError):
            _qcache = {}
    return _qcache


def expand_query(query):
    """
    Expand a short/vague food request into a richer phrase of concrete example
    dishes and ingredients, so its embedding separates cleanly from unrelated food.

    "meat" -> "meat dishes such as beef, steak, chicken, pork, bacon, sausage,
    turkey, lamb, meatballs and other animal-protein entrees"

    This is a lightweight query-expansion step (a standard retrieval technique).
    Results are cached on disk so each distinct query is only expanded once.
    """
    query = query.strip()
    cache = _load_qcache()
    if query in cache:
        return cache[query]

    try:
        resp = _get_client().chat.completions.create(
            model=EXPAND_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": (
                    "You expand a diner's food request into a short search phrase for a "
                    "menu semantic search. List concrete example dishes and ingredients that "
                    "match the request. Reply with ONLY the phrase, no preamble, under 40 words."
                )},
                {"role": "user", "content": f"Food request: {query}"},
            ],
        )
        expanded = (resp.choices[0].message.content or "").strip() or query
    except Exception:
        expanded = query  # never fail the search over the expansion step

    cache[query] = expanded
    with open(QUERY_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    return expanded


def filter_relevant(query, item_names, dietary=None):
    """
    LLM re-ranking: given candidate dish names, return the SET that genuinely
    matches the request AND fits the diner's dietary needs. Embeddings confuse
    cooking style with ingredient ("BBQ tofu" vs "meat"); a strict model pass
    fixes that, and the same pass enforces vegetarian/vegan/halal (the menu data
    has no dietary tags, so this is the only place we can).

    The model sees EVERY dish (chunked in batches), so matches aren't missed just
    because a compound name (e.g. "Dijon Crusted Pork Loin") ranked low. Falls back
    to keeping everything if the model call/parse fails.
    """
    names = list(dict.fromkeys(item_names))  # dedupe, keep order
    if not names:
        return set()

    batch = 45
    if len(names) > batch:
        keep = set()
        for i in range(0, len(names), batch):
            keep |= _filter_relevant_batch(query, names[i:i + batch], dietary)
        return keep
    return _filter_relevant_batch(query, names, dietary)


def _filter_relevant_batch(query, names, dietary=None):
    diet_rule = ""
    if dietary:
        diet_rule = (
            f" The diner is {', '.join(dietary)}. ALSO exclude any dish that violates that "
            "(vegetarian excludes meat, poultry, fish, gelatin; vegan also excludes dairy and eggs; "
            "halal excludes pork and alcohol) - UNLESS the request explicitly asks for that food "
            "(e.g. asking for 'meat'), in which case honor the explicit request over the stored "
            "dietary preference."
        )

    numbered = "\n".join(f"{i}. {n}" for i, n in enumerate(names))
    try:
        resp = _get_client().chat.completions.create(
            model=EXPAND_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": (
                    "You decide which menu items truly match a diner's request. Be strict. "
                    "For example, for 'meat' do NOT include tofu, vegetables, salads, or "
                    "vegetarian dishes." + diet_rule + " Reply with ONLY the matching numbers, "
                    "comma-separated (e.g. '0, 3, 4'). If none match, reply 'none'."
                )},
                {"role": "user", "content": f"Request: {query}\n\nItems:\n{numbered}"},
            ],
        )
        text = (resp.choices[0].message.content or "").strip().lower()
        if text == "none":
            return set()
        keep = set()
        for tok in text.replace(".", ",").split(","):
            tok = tok.strip()
            if tok.isdigit() and int(tok) < len(names):
                keep.add(names[int(tok)])
        return keep or set(names)
    except Exception:
        return set(names)


def filter_dietary(item_names, dietary):
    """Return the SET of dishes a diner with these dietary needs can eat.

    Unlike filter_relevant this does NOT judge relevance - it only removes dishes
    that violate the diet, keeping everything else (used for the 'what's good'
    board where we want variety, not query-matching).
    """
    if not dietary:
        return set(item_names)
    names = list(dict.fromkeys(item_names))
    if not names:
        return set()

    numbered = "\n".join(f"{i}. {n}" for i, n in enumerate(names))
    try:
        resp = _get_client().chat.completions.create(
            model=EXPAND_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": (
                    f"A diner is {', '.join(dietary)}. From the list, reply with the numbers of "
                    "the dishes they CANNOT eat because it violates their diet (vegetarian excludes "
                    "meat, poultry, fish, gelatin; vegan also excludes dairy and eggs; halal excludes "
                    "pork and alcohol). Keep everything else. Reply ONLY those numbers comma-separated, "
                    "or 'none' if they can eat all of them."
                )},
                {"role": "user", "content": numbered},
            ],
        )
        text = (resp.choices[0].message.content or "").strip().lower()
        if text == "none":
            return set(names)
        excluded = set()
        for tok in text.replace(".", ",").split(","):
            tok = tok.strip()
            if tok.isdigit() and int(tok) < len(names):
                excluded.add(names[int(tok)])
        return set(names) - excluded
    except Exception:
        return set(names)


def curate_highlights(hall_candidates, dietary=None, per_hall=4):
    """Let the model curate the 'what's good' board: for each hall pick up to
    `per_hall` real MAIN dishes, dropping plain sides / sauces / toppings and
    balancing types (not all desserts). Returns {hall: [names]} or None on failure.
    """
    if not hall_candidates:
        return {}
    diet = ""
    if dietary:
        diet = f" The diner is {', '.join(dietary)}; exclude any dish that violates that."
    try:
        resp = _get_client().chat.completions.create(
            model=EXPAND_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": (
                    f"You curate a dining-hall 'what's good' board. For EACH hall, choose up to "
                    f"{per_hall} dishes that best show what's good to eat. Prefer real MAIN dishes. "
                    "EXCLUDE plain sides (steamed/roasted plain vegetables, plain rice, beans, corn, "
                    "bread/rolls/buns), sauces, condiments, toppings, and drinks. Include at most ONE "
                    "dessert per hall, and give a mix of types rather than all the same." + diet +
                    " Keep dish names EXACTLY as written. Return ONLY a JSON object mapping each hall "
                    "name to an array of chosen dish names."
                )},
                {"role": "user", "content": json.dumps(hall_candidates)},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if "{" in text:
            text = text[text.find("{"):text.rfind("}") + 1]
        data = json.loads(text)
        return {h: [str(x) for x in lst] for h, lst in data.items() if isinstance(lst, list)}
    except Exception:
        return None


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def rank_by_similarity(query, items, text_key="item"):
    """
    Given a natural-language query and a list of item dicts, attach a
    `semantic_score` (0-1, higher = more relevant) to each item based on cosine
    similarity between the query embedding and each item's embedding.

    Returns the same list of dicts, each with a new `semantic_score` key,
    sorted most-relevant first.
    """
    if not items:
        return []

    texts = [it[text_key] for it in items]
    query_vec = embed_texts([query])[0]
    item_vecs = embed_texts(texts)

    scored = []
    for it, vec in zip(items, item_vecs):
        enriched = dict(it)
        enriched["semantic_score"] = _cosine(query_vec, vec)
        scored.append(enriched)

    scored.sort(key=lambda x: x["semantic_score"], reverse=True)
    return scored
