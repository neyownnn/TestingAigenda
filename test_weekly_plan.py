"""End-to-end check of the weekly plan with half-hour times against a real MySQL/MariaDB (throwaway DB `aigenda_test2`).
Run:  pip install httpx && python test_weekly_plan.py   (MySQL must be running)"""
import os
os.environ.update(DB_NAME="aigenda_test2", AIGENDA_MAINTENANCE="0")
import main
from datetime import date, timedelta
from fastapi.testclient import TestClient

boot = main.connect(with_db=False); boot.cursor().execute("DROP DATABASE IF EXISTS aigenda_test2"); boot.close()


def week(subjects_by_weekday):
    """Next 7 days starting today; subjects_by_weekday maps Python weekday (Mon=0) -> [(name, start, end)]."""
    t = date.today()
    days = []
    for i in range(7):
        d = t + timedelta(days=i)
        days.append({"day": str(d), "vacant": [],
                     "subjects": [{"name": n, "start": s, "end": e} for n, s, e in subjects_by_weekday.get(d.weekday(), [])]})
    return {"days": days}


with TestClient(main.app) as c:
    r = c.post("/api/register", json={"name": "Half Hour", "email": "h@test.local", "password": "secret1"})
    H = {"Authorization": "Bearer " + r.json()["token"]}
    assert c.put("/api/habits", headers=H, json={"wake": 6, "sleep": 23, "cs": 8, "ce": 12, "chrono": "morning",
                                                  "hobbies": ["Gaming"]}).status_code == 200

    # "Math 10:30-11:30 Monday, Wednesday", "Physics 13:00-14:30 Monday, Tuesday"
    wd = date.today().weekday()                      # first class day is today, so today's schedule can be checked
    plan = week({wd: [("Math", 10.5, 11.5), ("Physics", 13, 14.5)], (wd + 1) % 7: [("Physics", 13, 14.5)],
                 (wd + 2) % 7: [("Math", 10.5, 11.5)]})
    res = c.put("/api/weekly/plan", headers=H, json=plan)
    assert res.status_code == 200, res.text
    rows = c.get("/api/schedule/week", headers=H).json()
    assert rows, "schedule was generated"

    def day_rows(d): return [x for x in rows if x["day"] == str(d)]

    for day in plan["days"]:
        dr = day_rows(day["day"])
        assert dr, f"every day gets a schedule, including {day['day']} (no class)"
        # nothing overlaps, everything stays on the half-hour grid
        spans = sorted((x["hour"], x["hour"] + x["dur"], x["label"]) for x in dr)
        for (s1, e1, l1), (s2, e2, l2) in zip(spans, spans[1:]):
            assert e1 <= s2 + 1e-9, (day["day"], l1, l2, spans)
        assert all(abs(x["hour"] * 2 - round(x["hour"] * 2)) < 1e-9 for x in dr)
        # each class sits at exactly the entered time
        for s in day["subjects"]:
            cls = [x for x in dr if x["kind"] == "class" and x["label"] == "Class · " + s["name"]]
            assert cls and cls[0]["hour"] == s["start"] and cls[0]["dur"] == s["end"] - s["start"], (s, cls)
        # a subject with a class today gets study time today
        studied = {x["label"] for x in dr if x["kind"] == "study"}
        assert {s["name"] for s in day["subjects"]} <= studied, (day["day"], studied)
        # study never lands on top of a class
        classes = [(x["hour"], x["hour"] + x["dur"]) for x in dr if x["kind"] == "class"]
        for x in dr:
            if x["kind"] == "study":
                assert not any(x["hour"] < e and x["hour"] + x["dur"] > s for s, e in classes), x
    free_day = next(d for d in plan["days"] if not d["subjects"])
    assert {x["label"] for x in day_rows(free_day["day"]) if x["kind"] == "study"} <= {"Math", "Physics"}
    assert any(x["kind"] == "study" for x in day_rows(free_day["day"])), "free day still reviews subjects"
    print("day with classes:", [(x["hour"], x["dur"], x["label"]) for x in day_rows(plan["days"][0]["day"]) if x["kind"] in ("class", "study")])

    # validation
    bad = week({}); bad["days"][0]["subjects"] = [{"name": "X", "start": 10.25, "end": 11}]
    assert c.put("/api/weekly/plan", headers=H, json=bad).status_code == 422, "quarter hours rejected"
    bad["days"][0]["subjects"] = [{"name": "A", "start": 10, "end": 11.5}, {"name": "B", "start": 11, "end": 12}]
    assert c.put("/api/weekly/plan", headers=H, json=bad).status_code == 422, "overlapping classes rejected"
    bad["days"][0]["subjects"] = [{"name": "A", "start": 11, "end": 10.5}]
    assert c.put("/api/weekly/plan", headers=H, json=bad).status_code == 422, "end before start rejected"

    # move a study block to a half-hour time, and detect conflicts by overlap
    assert c.put("/api/weekly/plan", headers=H, json=plan).status_code == 200
    today_rows = c.get("/api/schedule", headers=H).json()
    study = [x for x in today_rows if x["kind"] == "study"]
    cls = next(x for x in today_rows if x["kind"] == "class")
    clash = c.patch(f"/api/schedule/{study[0]['id']}", headers=H, json={"hour": cls["hour"] + 0.5})
    assert clash.status_code == 409, "moving onto the middle of a class is a conflict"
    assert c.patch(f"/api/schedule/{study[0]['id']}", headers=H, json={"hour": 20.25}).status_code == 422
    me = study[0]
    others = [x for x in today_rows if x["id"] != me["id"]]
    target = next(t / 2 for t in range(0, 47) if t / 2 != me["hour"] and t / 2 + me["dur"] <= 24 and t % 2 == 1
                  and not any(x["hour"] < t / 2 + me["dur"] and x["hour"] + x["dur"] > t / 2 for x in others))
    ok = c.patch(f"/api/schedule/{me['id']}", headers=H, json={"hour": target})
    assert ok.status_code == 200 and any(x["id"] == me["id"] and x["hour"] == target for x in ok.json()), (target, ok.text)
    print("moved a study block to", target)
    sm = c.get("/api/schedule/summary", headers=H).json()
    assert sm["study_hours"] > 0 and sm["predicted_completion"] is not None, sm
    print("summary:", sm)
    print("ALL WEEKLY PLAN CHECKS PASSED")
