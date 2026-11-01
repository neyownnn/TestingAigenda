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
assert [b["hour"] for b in blocks if b["kind"] == "class"] == [9, 10]
print("scheduler OK")
