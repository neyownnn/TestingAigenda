"""AI Scheduler: rule-based cold-start heuristic, optionally re-scored by the Random Forest."""
from typing import Callable, Optional

PEAK = {"morning": (8, 12), "afternoon": (13, 17), "evening": (18, 22)}
SLOTS = 48                                                   # a day is 48 half-hour slots, so 10:30 is a valid time


def _slot(x: float) -> int:
    """Hour as a number (10.5 = 10:30) -> half-hour slot index 0..47."""
    return int(round(float(x) * 2)) % SLOTS


def _hour(slot: int) -> float:
    return slot / 2


def _window(start: float, end: float) -> list[int]:
    """Slots covered by [start, end), wrapping past midnight (22 to 2 covers 22:00..01:30). end may be 24."""
    s, e = _slot(start), int(round(float(end) * 2))
    n = e - s if e > s else (e - s + SLOTS if e < s else 0)
    return [(s + k) % SLOTS for k in range(n)]


def plan_day(h: dict, subjects: list[dict], hobbies: list[str],
             scorer: Optional[Callable[[float, int, int], float]] = None,
             classes: Optional[list[tuple[float, float, str]]] = None,
             vacant: Optional[list[tuple[float, float]]] = None,
             optional: Optional[list[dict]] = None) -> tuple[list[dict], list[str]]:
    """Plan ONE day on a 30-minute grid.

    h: wake, sleep, cs, ce, chrono.  subjects: [{name, rank}] that MUST get a study block today (their class is today).
    classes: [(start, end, label)] class times, e.g. (10.5, 11.5) = 10:30-11:30 (default: the habit profile's cs..ce).
    vacant:  [(start, end)] hours the student says are free; study/hobby blocks are only placed inside them.
    optional: [{name, rank}] extra subjects (e.g. ones with no class today) that get a study block only if time is spare.
    Study blocks are one hour; a 30-minute block is used only when a subject would otherwise get nothing.
    Returns (blocks, unplaced). Each block is {hour, dur, label, kind, rank, in_peak}; unplaced = required subjects with no block.
    """
    ps, pe = PEAK[h["chrono"]]
    wake, sleep = _slot(h["wake"]), _slot(h["sleep"])
    span = (sleep - wake) % SLOTS                            # awake slots; wraps past midnight (e.g. 7 to 3)
    seq = [(wake + k) % SLOTS for k in range(span)]          # wake slot first, wind-down last
    order = {sl: k for k, sl in enumerate(seq)}
    legacy = classes is None
    if legacy:
        classes = [(h["cs"], h["ce"], "Classes")]

    taken: set[int] = set()
    items: list[tuple[int, int, str, str, Optional[int]]] = []      # (start slot, n slots, label, kind, rank)

    def emit(slots: list[int], label: str, kind: str, rank: Optional[int] = None, max_n: int = SLOTS) -> None:
        """Mark free slots as one block per run of consecutive slots (each at most max_n slots)."""
        run: list[int] = []
        for sl in sorted((x for x in slots if x in order and x not in taken), key=order.get):
            if run and (order[sl] != order[run[-1]] + 1 or len(run) == max_n):
                items.append((run[0], len(run), label, kind, rank)); run = []
            run.append(sl)
            taken.add(sl)
        if run:
            items.append((run[0], len(run), label, kind, rank))

    for s, e, label in classes:
        emit(_window(s, e), label, "class")
    emit(seq[:2], "Wake · Breakfast", "rest", max_n=2)
    inner = seq[2:-2]                                        # slots that can hold meals and study
    inner_set = set(inner)

    def pair_free(a: int) -> bool:
        b = (a + 1) % SLOTS
        return a in inner_set and b in inner_set and a not in taken and b not in taken and order[b] == order[a] + 1

    def meal(prefs: list[float], label: str) -> None:
        """One-hour meal on the first preferred time that is free (on the hour, then on the half hour), else any free hour,
        else any free half hour. A meal is never dropped."""
        for hr in prefs:
            for off in (0, 1):
                a = (_slot(hr) + off) % SLOTS
                if pair_free(a):
                    emit([a, (a + 1) % SLOTS], label, "rest"); return
        for a in inner:
            if pair_free(a):
                emit([a, (a + 1) % SLOTS], label, "rest"); return
        for a in inner:
            if a not in taken:
                emit([a], label, "rest"); return

    meal([h["ce"], 12, 13] if legacy else [12, 13, 11, 14], "Lunch")
    meal([18, 19, 17], "Dinner")
    emit(seq[-2:], "Wind down", "rest", max_n=2)             # a class in the last hour is kept, not replaced

    free = [sl for sl in inner if sl not in taken]
    if vacant is not None:
        vac = {sl for s, e in vacant for sl in _window(s, e)}
        free = [sl for sl in free if sl in vac]
    free_set = set(free)

    subs = sorted(subjects, key=lambda s: s["rank"])
    extra = [s for s in (optional or []) if s["name"].lower() not in {x["name"].lower() for x in subs}]
    n = len(subs)

    # Keep one hour for a hobby, but only once every required subject can still get its own block.
    reserve = 1 if hobbies and len(free) > 2 * n else 0
    budget = max(0, len(free) - 2 * reserve)                 # study slots allowed today
    used = 0

    # Random Forest P(complete) for every (free start time, subject rank), fetched in a single batched call when possible.
    rf: dict = {}
    if scorer and free and (subs or extra):
        combos = [(_hour(sl), int(ps <= _hour(sl) < pe), r) for sl in free for r in {s["rank"] for s in subs + extra}]
        rf = scorer.batch(combos) if hasattr(scorer, "batch") else {c: scorer(*c) for c in combos}

    def score(sl: int, rank: int) -> float:
        hour = _hour(sl)
        in_peak = int(ps <= hour < pe)
        base = in_peak + 0.01 * (24 - hour)                  # rule-based: peak first, earlier is better
        if order[sl] >= span - 4:
            base -= 0.5                                      # avoid the last two hours before bed unless nothing else is left
        return base + rf.get((hour, in_peak, rank), 0.0)     # + Random Forest P(complete)

    placed: set[str] = set()

    def put(s: dict, allow_half: bool = False) -> bool:
        nonlocal used
        room = budget - used
        if room <= 0:
            return False
        cand, size = [], 0
        if room >= 2:
            cand, size = [sl for sl in free if pair_free(sl) and sl in free_set and (sl + 1) % SLOTS in free_set], 2
        if not cand and allow_half:
            cand, size = [sl for sl in free if sl not in taken], 1
        if not cand:
            return False
        best = max(cand, key=lambda sl: score(sl, s["rank"]))
        emit([(best + k) % SLOTS for k in range(size)], s["name"], "study", s["rank"], max_n=size)
        used += size
        placed.add(s["name"])
        return True

    # Pass 1: EVERY required subject gets one block, highest rank first.
    for s in subs:
        put(s, allow_half=True)
    # Pass 2: spare time goes to a second block for higher-ranked subjects (all but the lowest-ranked one).
    for s in subs[:-1]:
        put(s)
    # Pass 3: subjects with no class today (soonest next class first) fill what is still spare.
    for s in extra:
        put(s)

    left = [sl for sl in free if sl not in taken]
    hob: list[int] = []                                      # hobbies take the LATEST free hours, one hour per hobby
    for sl in sorted((x for x in left if pair_free(x) and (x + 1) % SLOTS in set(left)), key=order.get, reverse=True):
        if len(hob) >= len(hobbies) or sl in taken or (sl + 1) % SLOTS in taken:
            continue
        hob.append(sl)
        emit([sl, (sl + 1) % SLOTS], hobbies[len(hob) - 1], "hobby", max_n=2)
    emit([sl for sl in free if sl not in taken], "Free time · Rest", "rest", max_n=2)

    out = [{"hour": _hour(st), "dur": n_ / 2, "label": label, "kind": kind, "rank": rank,
            "in_peak": int(ps <= _hour(st) < pe)} for st, n_, label, kind, rank in sorted(items, key=lambda x: order[x[0]])]
    return out, [s["name"] for s in subs if s["name"] not in placed]


def generate(h: dict, subjects: list[dict], hobbies: list[str],
             scorer: Optional[Callable[[float, int, int], float]] = None) -> list[dict]:
    """Backwards-compatible wrapper: one day from the habit profile. Returns [{hour,dur,label,kind}]."""
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
