"""
dining.py - Data layer for Cornell Dining API

Fetches live eatery/hours/menu data from Cornell's public Dining Now API
and provides helper functions the chatbot's function-calling layer will call.
"""

import requests
import time
from datetime import datetime, timedelta

try:
    # Python 3.9+ standard library. Real IANA timezone => correct EST/EDT
    # automatically, instead of a hardcoded summer-only UTC-4 offset.
    from zoneinfo import ZoneInfo
    EASTERN = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - fallback if tzdata missing
    from datetime import timezone, timedelta
    EASTERN = timezone(timedelta(hours=-5))  # EST fallback


class DiningDataError(RuntimeError):
    """Raised when the live Cornell Dining API can't be reached and we have no cache."""


# Common nicknames -> a substring of the eatery's official name.
# e.g. students call North Star Dining Room "Appel" (it's in Appel Commons).
ALIASES = {
    "appel commons": "north star",
    "appel": "north star",
}

API_URL = "https://admin-now.dining.cornell.edu/api/1.0/dining/eateries.json"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# Simple in-memory cache so we don't hit the API on every single chat message
_cache = {"data": None, "fetched_at": 0}
CACHE_TTL_SECONDS = 15 * 60  # refresh every 15 minutes


def _fetch_eateries():
    """Fetch fresh eatery data from the Cornell Dining API."""
    response = requests.get(API_URL, headers=HEADERS, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data["data"]["eateries"]


def get_eateries():
    """Return cached eatery data, refreshing if the cache is stale.

    If the API is unreachable we fall back to the last good cache; only if we
    have never fetched successfully do we raise DiningDataError so callers can
    show the user a friendly message instead of crashing.
    """
    now = time.time()
    if _cache["data"] is None or (now - _cache["fetched_at"]) > CACHE_TTL_SECONDS:
        try:
            _cache["data"] = _fetch_eateries()
            _cache["fetched_at"] = now
        except (requests.RequestException, ValueError, KeyError) as exc:
            if _cache["data"] is not None:
                # Serve slightly stale data rather than failing the whole request.
                return _cache["data"]
            raise DiningDataError(
                "Couldn't reach Cornell Dining right now. Please try again in a minute."
            ) from exc
    return _cache["data"]


def _today_events(eatery):
    """Return today's list of meal events (breakfast/lunch/dinner) for an eatery, or []."""
    now = time.time()
    for day in eatery.get("operatingHours") or []:
        for event in day.get("events") or []:
            # Only consider events happening today by timestamp proximity
            if event.get("startTimestamp") and event.get("endTimestamp"):
                if event["startTimestamp"] - 86400 < now < event["endTimestamp"] + 86400:
                    yield event


def get_open_now():
    """
    Return a list of eateries currently open, each with minutes_until_close,
    sorted soonest-closing first.
    """
    now = time.time()
    open_halls = []

    for eatery in get_eateries():
        for event in _today_events(eatery):
            if event["startTimestamp"] <= now <= event["endTimestamp"]:
                minutes_left = int((event["endTimestamp"] - now) / 60)
                open_halls.append({
                    "name": eatery["name"],
                    "meal": event.get("descr"),
                    "closes_at": event.get("end"),
                    "minutes_until_close": minutes_left,
                    "location": eatery.get("location"),
                    "campus_area": eatery.get("campusArea", {}).get("descrshort", "Unknown"),
                })
                break  # only need the current event for this eatery

    open_halls.sort(key=lambda h: h["minutes_until_close"])
    return open_halls


def get_open_at(hour, minute=0):
    """
    Return a list of eateries that will be open at a specific time today.
    hour: 0-23 (24hr format), minute: 0-59.
    Useful for "what's open at 5:30pm?" type questions.
    """
    # Build target timestamp for today at the given time (real Eastern Time:
    # handles EST/EDT automatically).
    now_eastern = datetime.now(EASTERN)
    target = now_eastern.replace(hour=hour, minute=minute, second=0, microsecond=0)
    target_ts = target.timestamp()

    open_halls = []
    for eatery in get_eateries():
        for event in _today_events(eatery):
            if event["startTimestamp"] <= target_ts <= event["endTimestamp"]:
                minutes_left = int((event["endTimestamp"] - target_ts) / 60)
                open_halls.append({
                    "name": eatery["name"],
                    "meal": event.get("descr"),
                    "opens_at": event.get("start"),
                    "closes_at": event.get("end"),
                    "minutes_until_close": minutes_left,
                    "location": eatery.get("location"),
                    "campus_area": eatery.get("campusArea", {}).get("descrshort", "Unknown"),
                    "menu": event.get("menu", []),
                })
                break

    open_halls.sort(key=lambda h: h["minutes_until_close"])
    return open_halls


def _target_timestamp(hour=None, minute=0, day_offset=0):
    """Convert an optional hour/minute (+ a day offset) to a Unix timestamp in
    Eastern Time. day_offset=1 means tomorrow, etc. Returns current time if hour
    is None and no offset is given."""
    if hour is None and not day_offset:
        return time.time()
    base = datetime.now(EASTERN) + timedelta(days=day_offset)
    h = base.hour if hour is None else hour
    m = 0 if hour is None else minute
    target = base.replace(hour=h, minute=m, second=0, microsecond=0)
    return target.timestamp()


def _events_covering(eatery, target_ts):
    """Yield this eatery's meal events that are being served at target_ts, scanning
    ALL days in operatingHours (so future days like 'tomorrow' work), matched
    exactly by timestamp rather than by proximity to now."""
    for day in eatery.get("operatingHours") or []:
        for event in day.get("events") or []:
            start, end = event.get("startTimestamp"), event.get("endTimestamp")
            if start and end and start <= target_ts <= end:
                yield event


def get_menu(hall_name, hour=None, minute=0, day_offset=0):
    """
    Return the menu for a specific eatery at the target time. If hour is provided,
    checks that time; day_offset=1 checks tomorrow, etc.
    """
    target_ts = _target_timestamp(hour, minute, day_offset)
    hall_name_lower = hall_name.lower()

    # Resolve nicknames (e.g. "Appel" -> "north star") before matching.
    for alias, canonical in ALIASES.items():
        if alias in hall_name_lower:
            hall_name_lower = canonical
            break

    for eatery in get_eateries():
        if hall_name_lower in eatery["name"].lower() or hall_name_lower in eatery["slug"].lower():
            for event in _events_covering(eatery, target_ts):
                return {
                    "name": eatery["name"],
                    "meal": event.get("descr"),
                    "campus_area": eatery.get("campusArea", {}).get("descrshort", "Unknown"),
                    "menu": event.get("menu", []),
                }
            return {"name": eatery["name"], "meal": None, "menu": [], "note": "Not serving at that time."}

    return None


def search_menu_by_food(query, hour=None, minute=0, campus_area=None):
    """
    Search eateries' menus for items matching a food query.
    If hour is provided, searches menus at that future time instead of right now.
    If campus_area is provided (e.g. 'North'), only searches that campus.
    """
    target_ts = _target_timestamp(hour, minute)
    query_lower = query.lower()
    campus_lower = campus_area.lower() if campus_area else None
    matches = []

    for eatery in get_eateries():
        area = eatery.get("campusArea", {}).get("descrshort", "")
        if campus_lower and area.lower() != campus_lower:
            continue
        for event in _today_events(eatery):
            if event["startTimestamp"] <= target_ts <= event["endTimestamp"]:
                for category in event.get("menu", []):
                    for item in category.get("items", []):
                        if query_lower in item["item"].lower():
                            matches.append({
                                "hall": eatery["name"],
                                "item": item["item"],
                                "category": category["category"],
                                "campus_area": area,
                            })
    return matches


# Cache of "everyday staple" info, so we can hide it from general highlights.
# Recomputed whenever the eatery data refreshes.
_staples_cache = {"fetched_at": None, "value": None}

# Generic always-available stations. A dish whose name contains one of these is an
# everyday staple even when the exact name rotates (e.g. pizza toppings change daily
# but there's always A pizza). Kept only if the live data confirms it's near-daily.
# Deliberately EXCLUDES ambiguous words like "waffle" (Chicken & Waffles is a special).
_STATION_KEYWORDS = [
    "pizza", "salad", "fries", "ice cream", "cookie", "fruit", "yogurt", "oatmeal",
    "cereal", "bagel", "muffin", "dessert", "smoothie", "granola", "cake", "pastr",
    "oats", "sorbet", "beverage",
]


def _compute_staples(min_fraction=0.8):
    """Return {'names', 'keywords'} describing the everyday staples.

    names: exact dish names present on >= min_fraction of the days we have data for.
    keywords: station words (pizza, salad...) the live data confirms are near-daily.
    """
    from collections import defaultdict
    day_sets = defaultdict(set)
    kw_days = defaultdict(set)
    all_days = set()
    for eatery in get_eateries():
        for day in eatery.get("operatingHours") or []:
            for event in day.get("events") or []:
                ts = event.get("startTimestamp")
                if not ts:
                    continue
                date = datetime.fromtimestamp(ts, EASTERN).date()
                all_days.add(date)
                for category in event.get("menu") or []:
                    for item in category.get("items") or []:
                        name = (item.get("item") or "").strip().lower()
                        if not name:
                            continue
                        day_sets[name].add(date)
                        for kw in _STATION_KEYWORDS:
                            if kw in name:
                                kw_days[kw].add(date)
    n_days = len(all_days)
    if n_days < 3:  # not enough history to tell staples from specials
        return {"names": set(), "keywords": []}
    threshold = max(2, int(min_fraction * n_days))
    kw_threshold = max(2, n_days - 1)  # a station keyword must be nearly every day
    names = {name for name, dates in day_sets.items() if len(dates) >= threshold}
    keywords = [kw for kw in _STATION_KEYWORDS if len(kw_days[kw]) >= kw_threshold]
    return {"names": names, "keywords": keywords}


def get_staples():
    """Cached {'names', 'keywords'} of everyday-staple info."""
    if _staples_cache["value"] is None or _staples_cache["fetched_at"] != _cache["fetched_at"]:
        _staples_cache["value"] = _compute_staples()
        _staples_cache["fetched_at"] = _cache["fetched_at"]
    return _staples_cache["value"]


# Ingredient / topping / condiment / bread line items that are not dishes on their own.
_COMPONENT_CONTAINS = ("whipped cream", "pie filling", "au jus", "cool whip",
                       "sour cream", "heavy cream", "whipped topping", "lettuce",
                       "pickle", "hamburger bun", "hot dog bun", "slider bun",
                       "hamburger roll", "hoagie roll", "hot dog roll", "dinner roll",
                       "kaiser roll", "sub roll", "flour tortilla", "corn tortilla",
                       "nacho chip", "shredded cheese", "sliced cheese")
_COMPONENT_TAIL = ("filling", "topping", "toppings", "dressing", "syrup", "glaze",
                   "vinaigrette", "frosting", "icing", "jam", "jelly", "garnish",
                   "sprinkles", "marinade", "batter", "drizzle", "mayo", "ketchup",
                   "mustard", "relish", "gravy", "bun", "buns")


def is_component(item_name):
    """True for ingredient/topping/condiment line items (whipped cream, pie
    filling, ranch dressing, maple syrup...) that shouldn't be shown as dishes."""
    low = (item_name or "").strip().lower()
    if any(c in low for c in _COMPONENT_CONTAINS):
        return True
    words = low.split()
    return bool(words) and words[-1] in _COMPONENT_TAIL


def is_staple(item_name):
    """True if a dish is an everyday staple (exact name or a generic station word)."""
    staples = get_staples()
    low = (item_name or "").strip().lower()
    if low in staples["names"]:
        return True
    return any(kw in low for kw in staples["keywords"])


def meal_for_hour(hour=None):
    """Map an hour (24hr) to a meal period: breakfast / lunch / dinner.

    If hour is None, uses the current Eastern-time hour. Boundaries: before 11am
    is breakfast, 11am-4pm is lunch, 4pm onward is dinner.
    """
    if hour is None:
        hour = datetime.now(EASTERN).hour
    if hour < 11:
        return "breakfast"
    if hour < 16:
        return "lunch"
    return "dinner"


def collect_open_items(hour=None, minute=0, campus_area=None, day_offset=0):
    """
    Flatten every menu item served by halls open at the target time into one list.

    This is the retrieval corpus for the semantic search / ranking pipeline.
    Each item carries the metadata the ranking algorithm needs (campus, how soon
    the hall closes, etc.). day_offset=1 searches tomorrow, etc.
    """
    target_ts = _target_timestamp(hour, minute, day_offset)
    # campus_area may be None, a single name, or a list of names.
    if campus_area is None:
        allowed = None
    elif isinstance(campus_area, str):
        allowed = {campus_area.lower()}
    else:
        allowed = {a.lower() for a in campus_area}
    items = []

    for eatery in get_eateries():
        area = eatery.get("campusArea", {}).get("descrshort", "") or "Unknown"
        if allowed is not None and area.lower() not in allowed:
            continue
        for event in _events_covering(eatery, target_ts):
            minutes_left = int((event["endTimestamp"] - target_ts) / 60)
            for category in event.get("menu", []):
                for item in category.get("items", []):
                    name = item.get("item")
                    if not name:
                        continue
                    items.append({
                        "hall": eatery["name"],
                        "campus_area": area,
                        "location": eatery.get("location"),
                        "meal": event.get("descr"),
                        "category": category.get("category"),
                        "item": name,
                        "starts_at": event.get("start"),
                        "closes_at": event.get("end"),
                        "minutes_until_close": minutes_left,
                        "healthy": bool(item.get("healthy")),
                    })
            break  # one active event per eatery is enough
    return items


if __name__ == "__main__":
    # Quick manual test
    print("=== Open now ===")
    for hall in get_open_now():
        print(f"{hall['name']}: {hall['meal']} until {hall['closes_at']} "
              f"({hall['minutes_until_close']} min left)")

    print("\n=== Searching for 'chicken' ===")
    for match in search_menu_by_food("chicken")[:10]:
        print(f"{match['hall']} - {match['category']}: {match['item']}")

    print("\n=== Menu item counts for open halls (debug) ===")
    for eatery in get_eateries():
        for event in _today_events(eatery):
            if event["startTimestamp"] <= time.time() <= event["endTimestamp"]:
                print(f"{eatery['name']}: {len(event.get('menu', []))} menu categories")

    print("\n=== Direct check: 104West menu ===")
    print(get_menu("104West"))

    print("\n=== Campus areas ===")
    for eatery in get_eateries():
        print(f"{eatery['name']}: {eatery.get('campusArea')}")