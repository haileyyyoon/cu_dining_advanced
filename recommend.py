"""
recommend.py - The ranking pipeline that ties retrieval + signals together.

Flow:
  1. Retrieve   -> collect every menu item from halls open at the target time.
  2. Rank (semantic) -> embed & score each item by how well it matches the request.
  3. Rank (algorithm) -> blend the semantic score with real-world signals:
        - campus match   (is it where the user lives?)
        - closing soon   (don't send someone to a hall about to close)
        - healthy flag   (small nudge when the dining API marks an item healthy)
  4. Return the top picks for the chat model to explain, respecting allergies.

The final score is a weighted sum, so it's easy to explain in an application:
    score = 0.70*semantic + 0.20*campus_match + 0.10*time_ok (+ small healthy bonus)
"""

import dining
import preferences
import semantic

# Weights for the ranking algorithm (sum of the main three = 1.0).
W_SEMANTIC = 0.70
W_CAMPUS = 0.20
W_TIME = 0.10
HEALTHY_BONUS = 0.03

# Halls closing within this many minutes get penalized (not enough time to get there/eat).
CLOSING_SOON_MIN = 20

# Relevance gate for what shows up as tiles: a dish must match the query's MEANING
# at least this well to be shown, so campus/time bonuses can't pull in off-topic
# items (e.g. tofu or miso mayo when the user asked for "meat").
REL_RELEVANCE = 0.60   # keep dishes within 60% of the best semantic score...
ABS_RELEVANCE = 0.20   # ...but never below this absolute cosine similarity.


def _time_score(minutes_until_close):
    """1.0 if there's comfortable time, ramping to 0 as the hall is about to close."""
    if minutes_until_close is None:
        return 0.5
    if minutes_until_close >= 45:
        return 1.0
    if minutes_until_close <= CLOSING_SOON_MIN:
        return max(0.0, minutes_until_close / (CLOSING_SOON_MIN * 2))
    # Linear ramp between the two thresholds.
    return CLOSING_SOON_MIN / 45 + (minutes_until_close - CLOSING_SOON_MIN) / 45


def _violates_allergy(item_text, allergies):
    """Conservative substring check so anything that even mentions an allergen is dropped."""
    text = item_text.lower()
    return any(a and a.lower() in text for a in allergies)


# Categories treated as fillers - shown only if a hall doesn't have enough mains.
_FILLER_CATEGORY_WORDS = ("side", "soup", "dessert", "salad")

# Rough food-type buckets so one tile isn't all sweets or all eggs.
_SWEET_WORDS = ("cake", "cookie", "pie", "pudding", "brownie", "dessert", "chocolate",
                "cinnamon roll", "donut", "doughnut", "pastry", "cobbler", "custard",
                "muffin", "ice cream", "sorbet", "sundae", "parfait", "whipped",
                "danish", "churro", "crisp", "tart", "sweet roll")
_EGG_WORDS = ("egg", "omelet", "omelette", "frittata", "quiche", "scrambled", "benedict")
# Max of each capped type allowed per tile (savory mains are uncapped).
_TYPE_CAPS = {"sweet": 1, "egg": 2}


def _is_filler(item):
    return any(w in (item.get("category") or "").lower() for w in _FILLER_CATEGORY_WORDS)


def _food_type(item):
    low = item["item"].lower()
    if any(w in low for w in _EGG_WORDS):
        return "egg"
    if any(w in low for w in _SWEET_WORDS):
        return "sweet"
    return "savory"


def _pick_for_tile(items, per_hall):
    """Choose up to per_hall dishes from one hall, capping sweets/eggs so a tile is
    a mix. Relaxes the caps only if needed to reach a reasonable count."""
    chosen, counts = [], {}
    for it in items:
        t = _food_type(it)
        if t in _TYPE_CAPS and counts.get(t, 0) >= _TYPE_CAPS[t]:
            continue
        chosen.append(it)
        counts[t] = counts.get(t, 0) + 1
        if len(chosen) >= per_hall:
            return chosen
    # Under-filled because of the caps -> top up ignoring them.
    for it in items:
        if it not in chosen:
            chosen.append(it)
            if len(chosen) >= per_hall:
                break
    return chosen


def _build_highlights(safe_items, area, dietary, per_hall=5, min_per_hall=2, max_halls=6,
                      hall_pool=10):
    """Build the general 'what's good' board directly from what each hall offers.

    Prefers the day's specials over everyday staples; fills each hall's tile with
    up to `per_hall` dishes, MAIN dishes first so soup/sides only appear when a
    hall doesn't have enough mains; balances across the requested campuses; and
    enforces dietary needs once (the menu data has no veg/vegan/halal tags).
    """
    # Prefer specials; fall back to staples only if a hall has literally nothing else.
    specials = [it for it in safe_items if not dining.is_staple(it["item"])]
    pool = specials if specials else safe_items

    by_hall, hall_campus = {}, {}
    for it in pool:
        by_hall.setdefault(it["hall"], []).append(it)
        hall_campus.setdefault(it["hall"], it["campus_area"])

    # Mains before fillers (soup/sides/etc.), keeping menu order within each group.
    for lst in by_hall.values():
        lst.sort(key=lambda x: 1 if _is_filler(x) else 0)

    # Let the model curate each hall's picks (mains only, balanced types, dietary-safe).
    # Fall back to the keyword heuristic if the call fails.
    candidates = {h: [it["item"] for it in lst[:12]] for h, lst in by_hall.items()}
    curated = semantic.curate_highlights(candidates, dietary=dietary, per_hall=per_hall)

    if curated is not None:
        for h, lst in list(by_hall.items()):
            index = {it["item"].lower(): it for it in lst}
            chosen = [index[n.lower()] for n in curated.get(h, []) if n.lower() in index]
            by_hall[h] = chosen[:per_hall]
        min_per_hall = 1  # a single genuine main is fine; no keyword padding needed
    else:
        if dietary:
            allowed = semantic.filter_dietary(
                [it["item"] for lst in by_hall.values() for it in lst[:hall_pool]], dietary)
            for h in list(by_hall):
                by_hall[h] = [it for it in by_hall[h] if it["item"] in allowed]
        for h in list(by_hall):
            by_hall[h] = _pick_for_tile(by_hall[h], per_hall)

    hall_campus = {h: by_hall[h][0]["campus_area"] for h in by_hall if by_hall[h]}
    halls = [h for h in by_hall if len(by_hall[h]) >= min_per_hall] or [h for h in by_hall if by_hall[h]]
    halls.sort(key=lambda h: len(by_hall[h]), reverse=True)  # richer halls first
    if area and len(area) > 1:
        halls = _roundrobin_halls(halls, hall_campus, area)

    home = {a.lower() for a in area} if area else None
    picks = []
    for h in halls[:max_halls]:
        for it in by_hall[h][:per_hall]:
            it = dict(it)
            cm = 1.0 if home and it["campus_area"].lower() in home else (0.4 if home else 0.6)
            it["score"] = 0.0
            it["_breakdown"] = {"semantic": 0, "campus_match": cm, "time_ok": 0}
            picks.append(it)
    return {"picks": picks}


def _roundrobin_halls(halls, hall_campus, area):
    """Reorder halls so campuses alternate (keeps both campuses represented)."""
    buckets = {a.lower(): [] for a in area}
    for h in halls:
        buckets.setdefault(hall_campus[h].lower(), []).append(h)
    out, i = [], 0
    while any(i < len(v) for v in buckets.values()):
        for a in area:
            b = buckets.get(a.lower(), [])
            if i < len(b):
                out.append(b[i])
        i += 1
    return out


def find_food(query, hour=None, minute=0, campus_area=None,
              allergies=None, all_campuses=False, day_offset=0, highlights=False,
              hall=None, top_k=8):
    """
    Return the top_k ranked dishes matching `query` from halls open at the given
    time. Items mentioning a listed allergen are removed outright (safety first).

    Region selection (the "answers for that region unless specified" behavior):
      - all_campuses=True     -> search every campus.
      - campus_area given     -> that campus (user explicitly asked).
      - otherwise             -> the campus the user normally eats at for the
                                 MEAL implied by the time (breakfast/lunch/dinner).

    Each returned dish includes a `score` and its component breakdown, so the
    chat model (and a curious reviewer) can see *why* it was ranked where it is.
    """
    prefs = preferences.load()
    # Always apply saved allergies for safety, even if the caller forgot to pass them.
    allergies = list(allergies or []) + list(prefs.get("allergies") or [])
    dislikes = list(prefs.get("dislikes") or [])
    dietary = list(prefs.get("dietary") or [])

    hl = None
    if hall:
        hl = hall.strip().lower()
        for alias, canonical in dining.ALIASES.items():
            if alias in hl:
                hl = canonical
                break

    def area_for(m):
        # A hall or "anywhere" request isn't limited to the usual campus.
        if hall or all_campuses:
            return None
        if campus_area:
            return [campus_area] if isinstance(campus_area, str) else list(campus_area)
        return preferences.location_for_meal(m, prefs) or None  # usual campus for the meal

    def collect_at(h, mnt, m):
        got = dining.collect_open_items(h, mnt, campus_area=area_for(m), day_offset=day_offset)
        return [it for it in got if hl in it["hall"].lower()] if hall else got

    meal = dining.meal_for_hour(hour)
    search_area = area_for(meal)
    items = collect_at(hour, minute, meal)

    # No time given and nothing open right now (e.g. mid-afternoon between meals):
    # assume they mean the rest of TODAY, so roll forward to the next meal that has food.
    if not items and hour is None and not day_offset:
        now_hour = dining.datetime.now(dining.EASTERN).hour
        for h_try in sorted((9, 12, 18), key=lambda t: (t < now_hour, t)):
            m_try = dining.meal_for_hour(h_try)
            alt = collect_at(h_try, 0, m_try)
            if alt:
                items, hour, minute, meal, search_area = alt, h_try, 0, m_try, area_for(m_try)
                break

    if not items:
        where = f"at {hall}" if hall else ("on " + " or ".join(search_area) if search_area else "anywhere")
        when = "tomorrow" if day_offset == 1 else ("that day" if day_offset else "that time")
        return {
            "meal": meal,
            "region": (hall if hall else (" & ".join(search_area) if search_area else None)),
            "day_offset": day_offset,
            "picks": [],
            "note": f"Nothing's open {where} for {meal} {when}.",
        }

    # For region/scoring, use the hall's actual campus when a hall was requested.
    area = [items[0]["campus_area"]] if hall else search_area

    # Drop unsafe (allergies), unwanted (dislikes), and non-dish components
    # (whipped cream, pie filling...) before ranking.
    avoid = allergies + dislikes
    safe_items = [it for it in items
                  if not _violates_allergy(it["item"], avoid) and not dining.is_component(it["item"])]
    if not safe_items:
        return {"picks": [], "note": "Everything open right now conflicts with your allergies/dislikes."}

    # For general highlights ("what's good"), hide everyday staples (salad bar,
    # ice cream, fries...) so the special dishes stand out. Fall back to keeping
    # them only if that would leave too little.
    if highlights:
        # General "what's good": build a per-hall board directly (no query to rank
        # against), filling each tile with up to a few of that hall's own specials.
        result = _build_highlights(safe_items, area, dietary)
        result.update({
            "meal": meal, "region": (" & ".join(area) if area else "all campuses"),
            "hour": hour, "minute": minute, "day_offset": day_offset,
        })
        return result

    # Expand the query ("meat" -> "beef, chicken, pork, bacon...") so the embedding
    # separates matching dishes from unrelated ones, then rank by semantic similarity.
    search_text = semantic.expand_query(query)
    ranked = semantic.rank_by_similarity(search_text, safe_items, text_key="item")

    # Have the model strictly filter EVERY open dish (chunked) so only true matches
    # remain (removes e.g. tofu for "meat") and dietary is enforced. Filtering all
    # items - not just the top-ranked ones - means a match isn't missed because a
    # compound name ("Dijon Crusted Pork Loin") had a weak embedding.
    keep = semantic.filter_relevant(query, [c["item"] for c in ranked], dietary=dietary)
    filtered = [c for c in ranked if c["item"] in keep]
    # Fall back to top candidates only when there's no dietary rule to honor, so we
    # never reintroduce a meat dish for a vegetarian just to avoid an empty board.
    ranked = filtered or ([] if dietary else ranked[:4])

    home = {a.lower() for a in area} if area else None
    for it in ranked:
        campus_match = 1.0 if home and it["campus_area"].lower() in home else (0.4 if home else 0.6)
        time_ok = _time_score(it.get("minutes_until_close"))
        score = (
            W_SEMANTIC * it["semantic_score"]
            + W_CAMPUS * campus_match
            + W_TIME * time_ok
            + (HEALTHY_BONUS if it.get("healthy") else 0.0)
        )
        it["score"] = round(score, 4)
        it["_breakdown"] = {
            "semantic": round(it["semantic_score"], 3),
            "campus_match": campus_match,
            "time_ok": round(time_ok, 3),
        }

    ranked.sort(key=lambda x: x["score"], reverse=True)
    # Show ALL matching dishes (every hall that satisfies the request), just
    # interleaved across campuses for balance - no top-k truncation.
    picks = _balance_by_campus(ranked, area, len(ranked))
    region = " & ".join(area) if area else "all campuses"
    return {
        "meal": meal, "region": region,
        "hour": hour, "minute": minute, "day_offset": day_offset,  # so the UI can fetch a hall's full menu at this time
        "picks": picks,
    }


def _balance_by_campus(ranked, area, top_k):
    """Pick the top_k dishes, but when several campuses were requested, interleave
    them (round-robin) so a content-rich campus can't crowd the others out entirely.

    Within each campus the best matches still come first; we just guarantee every
    requested campus that HAS a match gets represented before filling remaining
    slots by score.
    """
    if not area or len(area) <= 1:
        return ranked[:top_k]

    by_campus = {a.lower(): [] for a in area}
    for it in ranked:
        by_campus.setdefault(it["campus_area"].lower(), []).append(it)

    picks, idx = [], 0
    # Round-robin across campuses (in the order the user listed them) until full.
    while len(picks) < top_k and any(idx < len(v) for v in by_campus.values()):
        for a in area:
            bucket = by_campus.get(a.lower(), [])
            if idx < len(bucket):
                picks.append(bucket[idx])
                if len(picks) >= top_k:
                    break
        idx += 1
    return picks


def suggest(hour=None, minute=0):
    """A personalized 'suggested for you' board for when the user hasn't searched.

    Uses the current meal + the campus they usually eat it on, leans toward what
    they've searched before, and respects their dietary needs / dislikes.
    """
    prefs = preferences.load()
    meal = dining.meal_for_hour(hour)

    theme_parts = []
    recent = prefs.get("recent_searches") or []
    if recent:
        theme_parts.append(recent[-1])          # their most recent craving
    if prefs.get("dietary"):
        theme_parts.append(" ".join(prefs["dietary"]))
    theme = " ".join(theme_parts) if theme_parts else f"a satisfying {meal}"

    # The 'suggested for you' board is a highlights view -> hide everyday staples.
    result = find_food(theme, hour=hour, minute=minute, highlights=True)
    result["suggested"] = True
    result["theme"] = theme
    return result


if __name__ == "__main__":
    import json
    result = find_food("meat", campus_area="North")
    print(json.dumps(result, indent=2)[:2000])
