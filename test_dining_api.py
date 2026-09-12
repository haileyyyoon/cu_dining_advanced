"""
dining.py - Data layer for Cornell Dining API

Fetches live eatery/hours/menu data from Cornell's public Dining Now API
and provides helper functions the chatbot's function-calling layer will call.
"""

import requests
import time

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
    """Return cached eatery data, refreshing if the cache is stale."""
    now = time.time()
    if _cache["data"] is None or (now - _cache["fetched_at"]) > CACHE_TTL_SECONDS:
        _cache["data"] = _fetch_eateries()
        _cache["fetched_at"] = now
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
                })
                break  # only need the current event for this eatery

    open_halls.sort(key=lambda h: h["minutes_until_close"])
    return open_halls


def get_menu(hall_name):
    """
    Return today's current menu (categories + items) for a specific eatery,
    matched by name (case-insensitive, partial match allowed).
    """
    now = time.time()
    hall_name_lower = hall_name.lower()

    for eatery in get_eateries():
        if hall_name_lower in eatery["name"].lower() or hall_name_lower in eatery["slug"].lower():
            for event in _today_events(eatery):
                if event["startTimestamp"] <= now <= event["endTimestamp"]:
                    return {
                        "name": eatery["name"],
                        "meal": event.get("descr"),
                        "menu": event.get("menu", []),
                    }
            return {"name": eatery["name"], "meal": None, "menu": [], "note": "Not currently serving."}

    return None


def search_menu_by_food(query):
    """
    Search all currently-open eateries' menus for items matching a food query.
    Returns a list of {hall, item, category} matches.
    """
    now = time.time()
    query_lower = query.lower()
    matches = []

    for eatery in get_eateries():
        for event in _today_events(eatery):
            if event["startTimestamp"] <= now <= event["endTimestamp"]:
                for category in event.get("menu", []):
                    for item in category.get("items", []):
                        if query_lower in item["item"].lower():
                            matches.append({
                                "hall": eatery["name"],
                                "item": item["item"],
                                "category": category["category"],
                            })
    return matches


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
