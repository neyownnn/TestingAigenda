"""AI Scheduler: rule-based cold-start heuristic, optionally re-scored by the Random Forest."""
from typing import Callable, Optional

PEAK = {"morning": (8, 12), "afternoon": (13, 17), "evening": (18, 22)}


def _place_meal(blocks: dict, prefs: list[int], label: str, lo: int, hi: int) -> None:
    """Put a meal on the first preferred hour that is still free (lo < hour < hi)."""
    for hr in prefs:
        if lo < hr < hi and hr not in blocks:
            blocks[hr] = (label, "rest")
            return


def plan_day(h: dict, subjects: list[dict], hobbies: list[str],
             scorer: Optional[Callable[[int, int, int], float]] = None,
             classes: Optional[list[tuple[int, int, str]]] = None,
             vacant: Optional[list[tuple[int, int]]] = None) -> tuple[list[dict], list[str]]:
    """Plan ONE day.

    h: wake, sleep, cs, ce, chrono.  subjects: [{name, rank}] to study today.
    classes: [(start, end, label)] class hours for this day (default: the habit profile's cs..ce).
    vacant:  [(start, end)] hours the student says are free; study/hobby blocks are only placed inside them.
             None = every hour between wake-up and bedtime that is not busy.
    Returns (blocks, unplaced) where unplaced = subjects that got no study block because the day is full.
    """
    ps, pe = PEAK[h["chrono"]]
    wake, sleep = h["wake"], h["sleep"]
    legacy = classes is None
    if legacy:
        classes = [(h["cs"], h["ce"], "Classes")]

    blocks: dict[int, tuple[str, str]] = {wake: ("Wake · Breakfast", "rest")}
    for s, e, label in classes:
        for i in range(s, e):
            if wake <= i < sleep:
                blocks[i] = (label, "class")
    _place_meal(blocks, [h["ce"], 12, 13] if legacy else [12, 13, 11, 14], "Lunch", wake, sleep - 1)
    _place_meal(blocks, [18, 19, 17], "Dinner", wake, sleep - 1)
    blocks[sleep - 1] = ("Wind down", "rest")

    free = [i for i in range(wake + 1, sleep - 1) if i not in blocks]
    if vacant is not None:
        free = [i for i in free if any(s <= i < e for s, e in vacant)]

    subs = sorted(subjects, key=lambda s: s["rank"])
    n = len(subs)

    # Keep one slot for a hobby, but only once every subject can still get its own block.
    reserve = 1 if hobbies and len(free) > n else 0
    budget = max(0, len(free) - reserve)

    # Random Forest P(complete) for every (free hour, subject rank), fetched in a single batched call when possible.
    rf: dict = {}
    if scorer and free and subs:
        combos = [(i, int(ps <= i < pe), r) for i in free for r in {s["rank"] for s in subs}]
        rf = scorer.batch(combos) if hasattr(scorer, "batch") else {c: scorer(*c) for c in combos}

    def score(hour: int, rank: int) -> float:
        in_peak = int(ps <= hour < pe)
        base = in_peak + 0.01 * (24 - hour)                 # rule-based: peak first, earlier is better
        if hour >= sleep - 2:
            base -= 0.5                                      # avoid the last hours before bed unless nothing else is left
        return base + rf.get((hour, in_peak, rank), 0.0)     # + Random Forest P(complete)

    meta: dict[int, int] = {}                                # hour -> subject rank (study blocks)
    placed: set[str] = set()

    def put(s: dict) -> bool:
        cand = [i for i in free if i not in blocks]
        if not cand or len(meta) >= budget:
            return False
        best = max(cand, key=lambda i: score(i, s["rank"]))
        blocks[best] = (s["name"], "study")
        meta[best] = s["rank"]
        placed.add(s["name"])
        return True

    # Pass 1: EVERY subject gets one block, highest rank first (this is what used to run out after ~4 subjects).
    for s in subs:
        put(s)
    # Pass 2: spare hours go to a second block for higher-ranked subjects (all but the lowest-ranked one).
    for s in subs[:-1]:
        put(s)

    rest = [i for i in free if i not in blocks]
    for j, i in enumerate(rest[-len(hobbies):] if hobbies else []):
        blocks[i] = (hobbies[j % len(hobbies)], "hobby")
    for i in free:
        blocks.setdefault(i, ("Free time · Rest", "rest"))

    out = [{"hour": i, "label": blocks[i][0], "kind": blocks[i][1], "rank": meta.get(i),
            "in_peak": int(ps <= i < pe)} for i in sorted(blocks)]
    return out, [s["name"] for s in subs if s["name"] not in placed]


def generate(h: dict, subjects: list[dict], hobbies: list[str],
             scorer: Optional[Callable[[int, int, int], float]] = None) -> list[dict]:
    """Backwards-compatible wrapper: one day from the habit profile. Returns [{hour,label,kind}]."""
    return plan_day(h, subjects, hobbies, scorer)[0]


def n_study_blocks(n_subjects: int) -> int:
    """Rough planned load before a day is built (used only as a model input estimate, capped at a realistic day)."""
    return min(8, sum(max(1, min(2, n_subjects - i)) for i in range(n_subjects)))


def tips(h: dict, subjects: list[dict], hobbies: list[str], progress: Optional[dict] = None) -> list[str]:
    """Personalized tips. If the student is consistently on track (>= 80% this week), encouragement replaces corrective tips."""
    ps, pe = PEAK[h["chrono"]]
    out = []
    if progress and progress.get("enough_data") and progress.get("week_pct", 0) >= 80:
        out.append(f"You completed {progress['week_pct']}% of your activities this week. Keep your routine exactly as it is.")
        if progress.get("streak", 0) >= 2:
            out.append(f"{progress['streak']}-day streak! Protect it by finishing one small task early tomorrow.")
    elif progress and progress.get("enough_data") and progress.get("week_pct", 100) < 50:
        out.append(f"Only {progress['week_pct']}% of activities were completed this week. Try fewer, shorter blocks "
                   "and put the hardest subject at the start of your focus window.")
    if subjects:
        top = min(subjects, key=lambda s: s["rank"])["name"]
        out.append(f"Your best focus is {ps}:00–{pe}:00. Keep {top} there and move "
                   f"{hobbies[0] if hobbies else 'hobbies'} to the evening.")
    out.append(f"You are awake {(h['sleep'] - h['wake']) % 24} hours a day, so short breaks between study blocks will help.")
    out.append("Finishing one small task early builds the streak that keeps you consistent.")
    return out


GENERAL_TIPS = ["Study in 45–50 minute blocks with a 10-minute break.",
                "Review notes within 24 hours of class; it halves what you forget.",
                "Keep a steady wake time, even on weekends."]
RESOURCES = {  # educational links only (UC-11 non-functional requirement)
    "Focus": [("Pomodoro technique", "https://en.wikipedia.org/wiki/Pomodoro_Technique")],
    "Procrastination": [("Procrastination", "https://en.wikipedia.org/wiki/Procrastination")],
    "Sleep": [("Sleep and learning", "https://en.wikipedia.org/wiki/Sleep_and_learning")],
    "Planning": [("Time management", "https://en.wikipedia.org/wiki/Time_management")],
}
ADVICE = {
    "Focus": "Work in 50-minute blocks inside your focus window, then take a 10-minute break. Keep your phone in another room.",
    "Procrastination": "Start with a 5-minute version of the task. Starting is the hard part, and momentum does the rest.",
    "Sleep": "Hold a steady wake time. Stress and low sleep hurt concentration, so protect your wind-down hour.",
    "Planning": "Plan tomorrow tonight: pick your top 3 tasks and attach each to a time slot.",
}


def support(h: dict, subjects: list[dict], progress: Optional[dict], ask: Optional[str] = None) -> dict:
    """UC-11: study tips + resources + time-management advice. Falls back to general tips with too little data."""
    topic = next((k for k in ADVICE if ask and (k.lower() in ask.lower() or
                  any(w in ask.lower() for w in {"Focus": ["focus", "distract", "phone"], "Procrastination": ["procrast", "start", "delay"],
                                                 "Sleep": ["sleep", "tired", "stress"], "Planning": ["plan", "schedule", "time"]}[k]))), None)
    enough = bool(progress and progress.get("enough_data")) and bool(subjects)
    tp = tips(h, subjects, [], progress) if enough else GENERAL_TIPS
    topics = [topic] if topic else list(ADVICE)
    return {"personalized": enough, "topic": topic, "tips": tp[:3],
            "resources": [{"title": t, "url": u} for k in topics for t, u in RESOURCES[k]],
            "advice": [ADVICE[k] for k in topics][:2 if not topic else 1],
            "note": None if enough else "Not enough activity data yet, so these are general tips. Use your schedule for a few days to personalize them."}
