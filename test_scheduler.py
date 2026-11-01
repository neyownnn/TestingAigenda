"""Regression tests for the scheduler (no database needed):  python test_scheduler.py"""
import scheduler

H = {"wake": 6, "sleep": 23, "cs": 8, "ce": 12, "chrono": "morning"}


def subjects(n): return [{"name": f"S{i}", "rank": i} for i in range(1, n + 1)]


# Bug: with more than 4 subjects the later ones got no study block at all.
for n in (1, 2, 4, 5, 6, 8):
    blocks, unplaced = scheduler.plan_day(H, subjects(n), ["Gaming", "Music"])
    got = {b["label"] for b in blocks if b["kind"] == "study"}
    assert got == {f"S{i}" for i in range(1, n + 1)} and not unplaced, (n, got, unplaced)

# A genuinely full day reports what didn't fit instead of silently dropping it.
blocks, unplaced = scheduler.plan_day(H, subjects(12), [])
assert unplaced, "a 12-subject day cannot fit and must say so"
assert unplaced == [f"S{i}" for i in range(13 - len(unplaced), 13)], "lowest-ranked subjects are the ones left out"

# Vacant time: study and hobby blocks stay inside it; class hours are blocked.
blocks, unplaced = scheduler.plan_day(H, subjects(3), ["Gaming"], classes=[(9, 11, "Class · Math")], vacant=[(13, 16), (19, 21)])
assert not unplaced
for b in blocks:
    if b["kind"] in ("study", "hobby"):
        assert 13 <= b["hour"] < 16 or 19 <= b["hour"] < 21, b
cls = [b for b in blocks if b["kind"] == "class"]
assert [(b["hour"], b["dur"]) for b in cls] == [(9, 2)], cls      # one class block: 9:00, 2 hours


def no_overlap(blocks):
    seen = set()
    for b in blocks:
        for k in range(int(b["dur"] * 2)):
            slot = (int(b["hour"] * 2) + k) % 48
            assert slot not in seen, ("overlap", b)
            seen.add(slot)


# Half-hour class times (10:30) are respected exactly and nothing overlaps them.
blocks, unplaced = scheduler.plan_day(H, subjects(3), ["Gaming"],
                                      classes=[(9, 10.5, "Class · Math"), (10.5, 11.5, "Class · Sci")])
no_overlap(blocks)
cls = {b["label"]: (b["hour"], b["dur"]) for b in blocks if b["kind"] == "class"}
assert cls == {"Class · Math": (9, 1.5), "Class · Sci": (10.5, 1)}, cls
assert not unplaced
for b in blocks:                                                   # study/hobby blocks are on the half-hour grid
    assert b["hour"] * 2 == int(b["hour"] * 2) and b["dur"] in (0.5, 1, 1.5, 2, 3, 4)

# Vacant windows may also use half hours.
blocks, unplaced = scheduler.plan_day(H, subjects(2), ["Gaming"], classes=[], vacant=[(13.5, 16.5)])
no_overlap(blocks)
for b in blocks:
    if b["kind"] in ("study", "hobby"):
        assert 13.5 <= b["hour"] and b["hour"] + b["dur"] <= 16.5, b

# A day with no class still gets study time from the week's other subjects (optional), and never reports them as unplaced.
blocks, unplaced = scheduler.plan_day(H, [], ["Gaming"], classes=[], optional=subjects(4))
assert {b["label"] for b in blocks if b["kind"] == "study"} == {"S1", "S2", "S3", "S4"} and not unplaced
# Required subjects come before optional ones when time is short.
blocks, unplaced = scheduler.plan_day(H, subjects(1), [], classes=[(7, 18, "Class")], optional=subjects(3)[1:])
got = {b["label"] for b in blocks if b["kind"] == "study"}       # dinner leaves 19:00-21:00 = two study hours
assert got == {"S1", "S2"} and not unplaced, got
print("scheduler OK")
