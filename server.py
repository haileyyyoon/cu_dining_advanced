"""
server.py - Web UI for the Cornell Dining assistant.

A one-way "input -> tiles" interface (not a back-and-forth chat window):
  - First visit: button-based onboarding to pick where you eat each meal + diet/allergies.
  - After that: you type one request; the assistant replies in a short strip and
    fills the screen with a TILE per recommended dining hall, its menu inside.

It reuses the same brain as the CLI (chat.py's tools + recommend.find_food), but
returns STRUCTURED data so the front-end can draw tiles instead of plain text.
"""

import json

from flask import Flask, jsonify, render_template, request

import chat
import dining
import preferences
import recommend

app = Flask(__name__)

# One local user, so a single in-memory conversation is fine. Lets the model
# remember context ("what about dinner?", a time mentioned earlier) between inputs.
MESSAGES = []

WEB_STYLE = """
IMPORTANT (web mode): The recommended dishes are shown to the user as visual tiles,
grouped by dining hall. So keep your text reply to ONE short, friendly sentence that
introduces the picks (e.g. "Here are some meaty dinner options on West:"). Do NOT list
the individual dishes in your text - the tiles already show them.
"""


def _system_message():
    """Fresh system prompt each turn so newly-saved preferences take effect."""
    content = chat.SYSTEM_PROMPT.format(profile=preferences.summary()) + WEB_STYLE
    return {"role": "system", "content": content}


def _why_bits(best_pick, match_count):
    """A short, human 'why this pick' from the ranking signals (no extra API call)."""
    bits = []
    breakdown = best_pick.get("_breakdown", {})
    if breakdown.get("campus_match") == 1.0:
        bits.append("on your campus")
    mins = best_pick.get("minutes_until_close")
    if isinstance(mins, int):
        if mins <= 25:
            bits.append("closing soon")
        elif mins >= 60:
            bits.append("open a while")
    bits.append(f"{match_count} match" + ("es" if match_count != 1 else ""))
    return bits


def _tiles_from_find_food(result):
    """Group find_food's flat list of dishes into one tile per dining hall.

    Picks arrive already sorted best-first, so the first pick seen for a hall is
    that hall's strongest match - we use it to explain why the tile ranked here.
    """
    tiles = []
    by_hall = {}
    for pick in result.get("picks", []):
        hall = pick["hall"]
        if hall not in by_hall:
            by_hall[hall] = {
                "hall": hall,
                "campus_area": pick.get("campus_area"),
                "meal": pick.get("meal"),
                "starts_at": pick.get("starts_at"),
                "closes_at": pick.get("closes_at"),
                # Time context so clicking the tile can fetch this hall's full menu.
                "hour": result.get("hour"),
                "minute": result.get("minute", 0),
                "day_offset": result.get("day_offset", 0),
                "_best": pick,
                "items": [],
            }
            tiles.append(by_hall[hall])
        # Skip duplicate dishes within a hall (the API sometimes lists one twice).
        if any(i["item"].lower() == pick["item"].lower() for i in by_hall[hall]["items"]):
            continue
        by_hall[hall]["items"].append({
            "item": pick["item"],
            "category": pick.get("category"),
        })

    for tile in tiles:
        tile["why"] = _why_bits(tile.pop("_best"), len(tile["items"]))
    return tiles


def _sanitize_find_food_args(args, user_message):
    """Guard against the model narrowing the search on its own.

    Only honor a specific campus if the user actually named one in this message,
    and only honor 'anywhere' if they actually asked for all campuses. Otherwise
    strip those so find_food falls back to the user's saved campuses for the meal.
    """
    msg = user_message.lower()
    named_campus = any(c in msg for c in ("north", "west", "central"))
    if args.get("campus_area") and not named_campus:
        args.pop("campus_area", None)
    wants_all = any(p in msg for p in ("anywhere", "any campus", "all campus",
                                       "all of campus", "everywhere", "every campus"))
    if args.get("all_campuses") and not wants_all:
        args["all_campuses"] = False
    # Decide 'highlights' (the curated, staple-free, variety-balanced browse) from
    # the USER'S message, not the model's arguments - the model is inconsistent about
    # setting it. An open-ended ask => highlights ON with a clean query, so it always
    # takes the same curated path as the "suggested for you" board.
    open_ended = any(p in msg for p in (
        "what's good", "whats good", "what looks good", "what's looking good",
        "whats looking good", "what should i", "anything good", "something good",
        "surprise me", "i'm hungry", "im hungry", "recommend", "suggestion",
        "what do you", "good options", "what's for", "whats for",
    ))
    if open_ended:
        args["highlights"] = True
        args["query"] = "a good meal"
    elif args.get("highlights"):
        # A specific food was named but the model flagged a browse; keep highlights
        # only if the query really is a generic browse phrase (else "vegan chocolate
        # cake" would get dropped as a "cake" staple).
        generic = {"a good meal", "a good lunch", "a good dinner", "a good breakfast",
                   "good meal", "good food", "something good", "a satisfying meal",
                   "a satisfying breakfast", "a satisfying lunch", "a satisfying dinner"}
        if (args.get("query") or "").strip().lower() not in generic:
            args["highlights"] = False
    return args


def run_turn(user_message):
    """Run one assistant turn; return (reply_text, tiles, context)."""
    if not MESSAGES:
        MESSAGES.append(_system_message())
    else:
        MESSAGES[0] = _system_message()

    MESSAGES.append({"role": "user", "content": user_message})

    find_food_results = []
    reply = ""

    # Tool-calling loop (allow a few rounds in case the model chains calls).
    for _ in range(5):
        response = chat.client.chat.completions.create(
            model=chat.MODEL, messages=MESSAGES, tools=chat.TOOLS,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            MESSAGES.append(message)
            reply = message.content or ""
            break

        MESSAGES.append(message)
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "find_food":
                args = _sanitize_find_food_args(args, user_message)
            result = chat.call_tool(tool_call.function.name, args)
            if tool_call.function.name == "find_food":
                find_food_results.append(result)  # keep ALL, so multi-food asks merge
            MESSAGES.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    # A find_food call means this was a real food search - remember it so the
    # "suggested for you" board can lean toward the user's tastes.
    if find_food_results:
        preferences.record_search(user_message)

    merged = _merge_find_food(find_food_results)
    tiles = _tiles_from_find_food(merged) if merged else []
    context = {}
    if merged:
        context = {"meal": merged.get("meal"), "region": merged.get("region")}
        if not tiles and merged.get("note"):
            context["note"] = merged["note"]
    return reply, tiles, context


def _merge_find_food(results):
    """Combine one or more find_food results (the user may ask for several foods
    in one message, e.g. 'broccoli and meat') into a single set of picks, deduped
    by hall+dish, so every requested food is represented in the tiles.
    """
    results = [r for r in results if r]
    if not results:
        return None

    seen, picks, notes = set(), [], []
    for r in results:
        for p in r.get("picks", []):
            key = (p.get("hall"), p.get("item"))
            if key not in seen:
                seen.add(key)
                picks.append(p)
        if r.get("note"):
            notes.append(r["note"])

    first = results[0]
    return {
        "meal": first.get("meal"),
        "region": first.get("region"),
        "hour": first.get("hour"),
        "minute": first.get("minute", 0),
        "day_offset": first.get("day_offset", 0),
        "picks": picks,
        "note": notes[0] if (not picks and notes) else None,
    }


@app.route("/")
def index():
    # Don't let the browser cache the page, so code changes always show up.
    resp = app.make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/profile")
def api_profile():
    prefs = preferences.load()
    return jsonify({
        "needs_onboarding": preferences.needs_onboarding(prefs),
        "summary": preferences.summary(prefs),
        "meal_locations": prefs["meal_locations"],
        "dietary": prefs["dietary"],
        "allergies": prefs["allergies"],
    })


@app.route("/api/onboarding", methods=["POST"])
def api_onboarding():
    data = request.get_json(force=True) or {}
    # The Preferences screen sends the full desired state, so replace (not merge)
    # -> deselecting a campus/diet/allergy actually clears it.
    preferences.replace(
        breakfast=data.get("breakfast"),
        lunch=data.get("lunch"),
        dinner=data.get("dinner"),
        dietary=data.get("dietary"),
        allergies=data.get("allergies"),
    )
    # New profile -> reset the conversation so the system prompt is rebuilt.
    MESSAGES.clear()
    return jsonify({"ok": True, "summary": preferences.summary()})


@app.route("/api/suggest")
def api_suggest():
    """Default 'suggested for you' board (no search needed)."""
    try:
        result = recommend.suggest()
    except dining.DiningDataError as exc:
        return jsonify({"error": str(exc)}), 503
    tiles = _tiles_from_find_food(result)
    meal = result.get("meal")
    region = result.get("region")
    if tiles:
        reply = f"Here's what looks good for {meal}" + (f" on {region}" if region else "") + ":"
    else:
        reply = "Nothing's open on your usual campus for the rest of today — try a search or ask for anywhere."
    return jsonify({
        "reply": reply, "tiles": tiles,
        "context": {"meal": meal, "region": region, "suggested": True},
    })


@app.route("/api/menu", methods=["POST"])
def api_menu():
    """Full menu for one hall at the same time the tiles were generated."""
    data = request.get_json(force=True) or {}
    hall = (data.get("hall") or "").strip()
    if not hall:
        return jsonify({"error": "no hall"}), 400
    hour = data.get("hour")
    minute = data.get("minute", 0)
    day_offset = data.get("day_offset", 0)
    try:
        menu = dining.get_menu(hall, hour, minute, day_offset=day_offset)
    except dining.DiningDataError as exc:
        return jsonify({"error": str(exc)}), 503
    if not menu:
        return jsonify({"hall": hall, "categories": []})
    categories = [
        {"category": c.get("category"), "items": [i.get("item") for i in c.get("items", [])]}
        for c in menu.get("menu", [])
    ]
    return jsonify({
        "hall": menu.get("name", hall),
        "meal": menu.get("meal"),
        "campus_area": menu.get("campus_area"),
        "categories": categories,
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400
    try:
        reply, tiles, context = run_turn(message)
    except Exception as exc:  # keep the UI alive on any backend hiccup
        return jsonify({"error": str(exc)}), 500
    return jsonify({"reply": reply, "tiles": tiles, "context": context})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
