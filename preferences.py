"""
preferences.py - Persistent user profile for the dining chatbot.

Stores the user's campus, dietary restrictions, allergies, and dislikes in a
small JSON file so the bot remembers them ACROSS sessions - not just within a
single conversation. This is what lets the assistant say "I remembered you're
vegetarian and allergic to peanuts" the next time you launch it.
"""

import json
import os

PREFS_PATH = os.path.join(os.path.dirname(__file__), "preferences.json")

_EMPTY = {
    "campus_area": None,        # "North" / "West" / "Central" (general home base)
    "meal_locations": {         # campuses the user normally eats each meal (a LIST each)
        "breakfast": [],
        "lunch": [],
        "dinner": [],
    },
    "dietary": [],              # e.g. ["vegetarian"], ["vegan"], ["halal"]
    "allergies": [],            # e.g. ["peanuts", "shellfish"]
    "dislikes": [],             # foods to avoid, e.g. ["mushrooms"]
    "recent_searches": [],      # last few things the user searched, newest last
}

MAX_RECENT = 12

MEALS = ("breakfast", "lunch", "dinner")

# Fields that are lists we merge into (deduped), vs. scalars we overwrite.
_LIST_FIELDS = ("dietary", "allergies", "dislikes")


def load():
    """Return the saved preferences (a fresh empty profile if none exist yet)."""
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError):
        return dict(_EMPTY)
    # Fill in any missing keys so callers can rely on the shape.
    prefs = {
        "campus_area": None,
        "meal_locations": {"breakfast": [], "lunch": [], "dinner": []},
        "dietary": [],
        "allergies": [],
        "dislikes": [],
        "recent_searches": [],
    }
    prefs.update({k: v for k, v in data.items() if k in prefs})
    # Normalize each meal's campuses to a clean list (older files stored a single
    # string, so accept both str and list here for backward compatibility).
    saved_meals = data.get("meal_locations") or {}
    prefs["meal_locations"] = {m: _as_area_list(saved_meals.get(m)) for m in MEALS}
    return prefs


def save(prefs):
    """Write the full preferences dict to disk."""
    with open(PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)
    return prefs


_VALID_AREAS = {"north", "west", "central"}


def _norm_area(value):
    """Normalize a campus name like 'north campus' -> 'North'. Returns None if unclear."""
    if not value:
        return None
    word = str(value).strip().lower().replace("campus", "").strip()
    return word.title() if word in _VALID_AREAS else str(value).strip().title()


def _as_area_list(value):
    """Coerce a str, list, or None into a clean, deduped list of campus names."""
    if not value:
        return []
    values = [value] if isinstance(value, str) else list(value)
    out = []
    for v in values:
        a = _norm_area(v)
        if a and a not in out:
            out.append(a)
    return out


def set_meal_locations(breakfast=None, lunch=None, dinner=None):
    """Set/replace the campuses the user normally eats each meal.

    Each argument may be a single campus or a list of campuses; only meals given
    a non-empty value are changed.
    """
    prefs = load()
    for meal, val in (("breakfast", breakfast), ("lunch", lunch), ("dinner", dinner)):
        areas = _as_area_list(val)
        if areas:
            prefs["meal_locations"][meal] = areas
    return save(prefs)


def needs_onboarding(prefs=None):
    """True when the user has not set any of their per-meal locations yet."""
    prefs = prefs or load()
    return not any(prefs["meal_locations"].get(m) for m in MEALS)


def location_for_meal(meal, prefs=None):
    """The campuses the user normally eats a given meal (list), falling back to
    their general home campus. Returns a list (possibly empty)."""
    prefs = prefs or load()
    areas = prefs["meal_locations"].get(meal) or []
    if areas:
        return list(areas)
    return [prefs["campus_area"]] if prefs.get("campus_area") else []


def _clean_list(values):
    """Lowercase, strip, and dedupe a list of tags while preserving order."""
    seen, out = set(), []
    for v in values or []:
        v = str(v).strip().lower()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def remember(campus_area=None, dietary=None, allergies=None, dislikes=None,
             breakfast_area=None, lunch_area=None, dinner_area=None):
    """
    Merge new preferences into the saved profile and persist it.

    List fields (dietary/allergies/dislikes) are ADDED to what's already there;
    campus_area and per-meal locations overwrite. Returns the updated profile.
    """
    prefs = load()

    if campus_area:
        prefs["campus_area"] = _norm_area(campus_area)

    for meal, val in (("breakfast", breakfast_area), ("lunch", lunch_area), ("dinner", dinner_area)):
        areas = _as_area_list(val)
        if areas:
            prefs["meal_locations"][meal] = areas

    for field, incoming in (
        ("dietary", dietary),
        ("allergies", allergies),
        ("dislikes", dislikes),
    ):
        if incoming:
            merged = prefs.get(field, []) + _clean_list(incoming)
            prefs[field] = _clean_list(merged)

    return save(prefs)


def record_search(query):
    """Remember a search so 'suggested for you' can lean toward the user's tastes."""
    query = (query or "").strip()
    if not query:
        return load()
    prefs = load()
    recent = [q for q in prefs.get("recent_searches", []) if q.lower() != query.lower()]
    recent.append(query)
    prefs["recent_searches"] = recent[-MAX_RECENT:]
    return save(prefs)


def replace(breakfast=None, lunch=None, dinner=None, dietary=None, allergies=None):
    """Overwrite meal locations, dietary and allergies with exactly these values.

    Used by the Preferences screen, which shows the FULL desired state - so
    deselecting something actually removes it (unlike remember(), which adds).
    Keeps dislikes and recent_searches untouched.
    """
    prefs = load()
    prefs["meal_locations"] = {
        "breakfast": _as_area_list(breakfast),
        "lunch": _as_area_list(lunch),
        "dinner": _as_area_list(dinner),
    }
    prefs["dietary"] = _clean_list(dietary)
    prefs["allergies"] = _clean_list(allergies)
    return save(prefs)


def summary(prefs=None):
    """One-line human-readable description of the profile for the system prompt."""
    prefs = prefs or load()
    parts = []
    if prefs.get("campus_area"):
        parts.append(f"lives on {prefs['campus_area']} campus")
    meal_bits = [f"{m} on {' & '.join(prefs['meal_locations'][m])}"
                 for m in MEALS if prefs["meal_locations"].get(m)]
    if meal_bits:
        parts.append("usually eats " + ", ".join(meal_bits))
    if prefs.get("dietary"):
        parts.append("dietary: " + ", ".join(prefs["dietary"]))
    if prefs.get("allergies"):
        parts.append("ALLERGIC to: " + ", ".join(prefs["allergies"]))
    if prefs.get("dislikes"):
        parts.append("dislikes: " + ", ".join(prefs["dislikes"]))
    return "; ".join(parts) if parts else "no saved preferences yet"


if __name__ == "__main__":
    print("Current profile:", summary())
