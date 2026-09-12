"""
chat.py - Cornell Dining chatbot

Uses OpenAI function calling to answer natural-language dining questions,
backed by live data from dining.py.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

import dining
import preferences
import recommend

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a helpful Cornell Dining assistant. You help students figure out
where to eat on campus, what's on the menu, and which dining halls are closing soon.

Each eatery has a campus_area: North, West, or Central.
When a user says they live on a specific campus (e.g. "I live on north"), remember that
for the ENTIRE conversation and ALWAYS pass it as the campus_area parameter in your tool
calls. Only show results from other campuses if the user explicitly asks.

Similarly, if the user mentioned a specific time (e.g. "at 5:30pm"), remember that and
ALWAYS pass hour/minute in follow-up tool calls unless they mention a different time.

FINDING FOOD: For ANY request to recommend, suggest, or find something to eat, use find_food.
This includes a KIND of food or a vibe ("meat", "something healthy", "spicy", "vegetarian")
AND open-ended asks ("what looks good", "what should I eat", "anything good?", "surprise me",
"I'm hungry"). For an open-ended ask with no specific food, use a general query like
"a good meal" or the user's usual tastes, and set highlights=true so everyday staples (salad
bar, ice cream, fries) are hidden and the day's SPECIAL dishes show. Make a SINGLE find_food
call for an open-ended ask - do not split it into several. find_food runs semantic
search + ranking over live menus and is the ONLY tool that produces the visual TILES the user
sees - so ALWAYS use it for recommendations. Never answer a food recommendation from
get_open_at or get_menu alone.
Use get_open_at only for pure "what's open / hours" questions, not for recommending food.
Lead with the top pick and briefly say why it's a good fit.

SPECIFIC HALL: If the user asks what's at / what a specific dining hall has ("what's at
Morrison", "does Keeton have anything good", "what's Okenshields serving"), call find_food with
the hall parameter set to that hall's name. If they didn't name a specific food, use
query "a good meal" and highlights=true; if they did (e.g. "what meat is at Morrison"), put
that food as the query and leave highlights off.

MULTIPLE FOODS: If the user asks for several DISTINCT foods, call find_food separately for
each one so every food is represented. Treat "or" as alternatives inside ONE request, and
"and" as separating distinct requests. Example: "broccoli or cauliflower and meat" -> TWO
calls: find_food("broccoli or cauliflower") and find_food("meat"). Never merge distinct
foods into a single query like "broccoli and meat".

MEAL-BASED REGION (important): The user has told you where they normally eat each meal
(breakfast/lunch/dinner) - see their profile below. find_food automatically figures out the
meal from the time and answers for THAT meal's usual campus. So:
- Do NOT pass campus_area unless the user explicitly names a campus in THIS message.
  Never choose a campus yourself from their saved profile - if you omit campus_area,
  find_food already searches ALL of their usual campuses for that meal (they may have
  several, e.g. dinner on West & North, and both should be shown).
- Only if the user explicitly names a campus in this message, pass that campus_area.
- If the user says "anywhere" / "any campus" / "all of campus", pass all_campuses=true.
- A meal word (breakfast/lunch/dinner) is NOT a campus - never pass it as campus_area.
  Campuses are only North, West, or Central.
- Always pass hour/minute when the user mentions a clock time (e.g. "5:30pm" -> hour=17, minute=30).
- If the user names a MEAL but no clock time (e.g. "for dinner"), pass a representative hour so
  the right meal period is searched: breakfast -> hour=9, lunch -> hour=12, dinner -> hour=18.
  Otherwise the current time is used.
- For a future DAY, pass day_offset: today=0 (default), tomorrow=1, day after=2, etc.
  (Menus are available about a week ahead.)
The result includes which meal and region it used - mention them naturally
(e.g. "for dinner on West...").

PREFERENCES & SAFETY: The user's saved profile is below. ALWAYS respect it:
- Never recommend a dish that conflicts with their allergies. If unsure, leave it out.
- Prefer options matching their dietary needs (vegetarian/vegan/halal) and avoid dislikes.
When the user states a NEW lasting preference, allergy, dietary restriction, or their home
campus, call remember_preference so it's saved for next time. Confirm briefly what you saved.

NICKNAMES: "Appel" or "Appel Commons" refers to North Star Dining Room. Treat them as
the same place.

Only use information returned by your tools - never guess menu items or hours.
If no dining halls are open, say so honestly.
Keep answers short and conversational, like you're texting a friend.

--- SAVED USER PROFILE ---
{profile}"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_open_now",
            "description": "Get a list of all Cornell dining locations currently open, sorted by soonest-closing first.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_open_at",
            "description": "Get a list of Cornell dining locations that will be open at a specific time today, including their menus. Use this when the user asks about a future time (e.g. 'what's open at 5:30pm', 'where can I eat dinner tonight').",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {
                        "type": "integer",
                        "description": "Hour in 24hr format (0-23). e.g. 17 for 5pm, 12 for noon.",
                    },
                    "minute": {
                        "type": "integer",
                        "description": "Minute (0-59). Defaults to 0.",
                    },
                },
                "required": ["hour"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_food",
            "description": "PREFERRED way to recommend food by kind or vibe (e.g. 'meat', 'something healthy', 'spicy', 'vegetarian'). Runs semantic search + ranking over live menus of open halls and returns the best-matching dishes with scores. Pass campus_area if the user said where they live, hour/minute if they mentioned a time, and allergies from the saved profile so unsafe dishes are excluded.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What the user wants, in natural language, e.g. 'meat', 'light and healthy', 'comfort food'.",
                    },
                    "hour": {
                        "type": "integer",
                        "description": "Optional. Hour in 24hr format (0-23) to check menus at a future time today.",
                    },
                    "minute": {
                        "type": "integer",
                        "description": "Optional. Minute (0-59). Defaults to 0.",
                    },
                    "campus_area": {
                        "type": "string",
                        "description": "Only pass if the user explicitly names a campus ('North', 'West', 'Central') in this request. Otherwise leave it out - find_food picks the user's usual campus for the meal automatically.",
                    },
                    "all_campuses": {
                        "type": "boolean",
                        "description": "Set true only if the user asks for options across ALL campuses ('anywhere', 'any campus').",
                    },
                    "day_offset": {
                        "type": "integer",
                        "description": "Which day: 0 = today (default), 1 = tomorrow, 2 = day after, etc. Set when the user asks about a future day.",
                    },
                    "highlights": {
                        "type": "boolean",
                        "description": "Set true ONLY for open-ended asks with NO specific food ('what's good', 'what should I eat', 'surprise me'). It hides everyday staples to surface the day's specials. NEVER set it when the user names a specific food or dish (e.g. 'vegan chocolate cake', 'meat').",
                    },
                    "hall": {
                        "type": "string",
                        "description": "Limit results to one dining hall (e.g. 'Morrison', 'Keeton', 'Okenshields'). Set when the user asks what's at / what a specific hall has. A hall request is NOT limited to their usual campus.",
                    },
                    "allergies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional. Extra allergens to exclude. Saved allergies are always applied automatically.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_preference",
            "description": "Save a lasting user preference so it persists across sessions. Call when the user states their home campus, where they usually eat a given meal, a dietary restriction, an allergy, or a food they dislike.",
            "parameters": {
                "type": "object",
                "properties": {
                    "campus_area": {
                        "type": "string",
                        "description": "General home campus: 'North', 'West', or 'Central'.",
                    },
                    "breakfast_area": {
                        "type": "string",
                        "description": "Where the user usually eats breakfast: 'North', 'West', or 'Central'.",
                    },
                    "lunch_area": {
                        "type": "string",
                        "description": "Where the user usually eats lunch: 'North', 'West', or 'Central'.",
                    },
                    "dinner_area": {
                        "type": "string",
                        "description": "Where the user usually eats dinner: 'North', 'West', or 'Central'.",
                    },
                    "dietary": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Dietary needs to add, e.g. ['vegetarian'], ['vegan'], ['halal'].",
                    },
                    "allergies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Allergies to add, e.g. ['peanuts', 'shellfish'].",
                    },
                    "dislikes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Foods the user wants to avoid, e.g. ['mushrooms'].",
                    },
                },
                "required": [],
            },
        },
    },
]


def call_tool(name, args):
    try:
        if name == "get_open_now":
            return dining.get_open_now()
        if name == "get_open_at":
            return dining.get_open_at(args["hour"], args.get("minute", 0))
        if name == "get_menu":
            return dining.get_menu(args["hall_name"], args.get("hour"), args.get("minute", 0))
        if name == "search_menu_by_food":
            return dining.search_menu_by_food(args["query"], args.get("hour"), args.get("minute", 0), args.get("campus_area"))
        if name == "find_food":
            return recommend.find_food(
                args["query"],
                hour=args.get("hour"),
                minute=args.get("minute", 0),
                campus_area=args.get("campus_area"),
                all_campuses=args.get("all_campuses", False),
                day_offset=args.get("day_offset", 0),
                highlights=args.get("highlights", False),
                hall=args.get("hall"),
                allergies=args.get("allergies"),
            )
        if name == "remember_preference":
            return preferences.remember(
                campus_area=args.get("campus_area"),
                breakfast_area=args.get("breakfast_area"),
                lunch_area=args.get("lunch_area"),
                dinner_area=args.get("dinner_area"),
                dietary=args.get("dietary"),
                allergies=args.get("allergies"),
                dislikes=args.get("dislikes"),
            )
    except dining.DiningDataError as exc:
        return {"error": str(exc)}
    return {"error": f"Unknown tool {name}"}


def chat(messages):
    """Run one turn: send messages to the model, execute any tool calls, return final reply."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )
    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = call_tool(tool_call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })
        # Get the model's final natural-language response after seeing tool results
        response = client.chat.completions.create(model=MODEL, messages=messages)
        message = response.choices[0].message
        messages.append(message)
    else:
        messages.append(message)

    return message.content, messages


def run_onboarding():
    """First-time setup: ask where the user normally eats each meal."""
    prefs = preferences.load()
    if not preferences.needs_onboarding(prefs):
        return prefs

    print("Welcome! Let's set up where you usually eat, so I can tailor recommendations.")
    print("Campus options: North, West, Central, East. (Press Enter to skip a meal.)\n")
    answers = {}
    for meal in preferences.MEALS:
        ans = input(f"  Where do you usually eat {meal}? ").strip()
        if ans:
            answers[f"{meal}_area"] = ans

    if answers:
        prefs = preferences.remember(**answers)
        print(f"\nGreat - saved: {preferences.summary(prefs)}\n")
    else:
        print("\nNo problem, you can tell me anytime.\n")
    return prefs


if __name__ == "__main__":
    run_onboarding()
    profile = preferences.summary()
    conversation = [{"role": "system", "content": SYSTEM_PROMPT.format(profile=profile)}]
    print("Cornell Dining Assistant (type 'quit' to exit)")
    print(f"(remembered: {profile})\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        conversation.append({"role": "user", "content": user_input})
        reply, conversation = chat(conversation)
        print(f"\nBot: {reply}\n")