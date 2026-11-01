"""AIgenda REST API (FastAPI + MySQL/MariaDB). Run: uvicorn main:app --reload

Learning loop: students fill in habits/subjects -> the scheduler places study blocks -> students tick them off ->
once a day has passed, every study block's outcome (done OR skipped) is copied into activity_log ->
when enough new real rows exist, the Random Forest is retrained on them (blended with the synthetic base)."""
import hashlib, json, logging, os, re, secrets, threading, time
from datetime import date, timedelta
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

import pymysql
import pymysql.cursors
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import ml, scheduler

HERE = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("aigenda")

# XAMPP defaults: MySQL on 127.0.0.1:3306, user root, empty password. Override with environment variables.
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "blah")
DB_NAME = os.environ.get("DB_NAME", "aigenda")
RETRAIN_MIN_ROWS = int(os.environ.get("RETRAIN_MIN_ROWS", "200"))   # real rows needed before the first retrain
RETRAIN_EVERY = int(os.environ.get("RETRAIN_EVERY", "50"))          # then retrain after this many new real rows
MAINTENANCE_SECONDS = int(os.environ.get("MAINTENANCE_SECONDS", "3600"))
MAX_SUBJECTS, MAX_GOALS = 15, 10                                      # limits from UC-03 and UC-10
PLAN_DAYS = 7                                                         # the student is asked for a new plan every 7 days
TEST_EMAIL_DOMAIN = "@test.local"                                   # accounts with this email are excluded from training


# ---------- database ----------
class Conn:
    """Thin helper around a PyMySQL cursor."""
    def __init__(self, cur): self.cur = cur
    def one(self, sql, a=()): self.cur.execute(sql, a); return self.cur.fetchone()
    def all(self, sql, a=()): self.cur.execute(sql, a); return list(self.cur.fetchall())
    def run(self, sql, a=()): self.cur.execute(sql, a); return self.cur.lastrowid
    def many(self, sql, rows): self.cur.executemany(sql, rows)


def connect(with_db=True):
    return pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
                           database=DB_NAME if with_db else None, charset="utf8mb4",
                           cursorclass=pymysql.cursors.DictCursor, autocommit=False)


@contextmanager
def db():
    con = connect()
    try:
        yield Conn(con.cursor())
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db():
    boot = connect(with_db=False)
    boot.cursor().execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4")
    boot.close()
    sql = re.sub(r"--.*", "", open(os.path.join(HERE, "schema.sql"), encoding="utf8").read())
    with db() as c:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            c.run(stmt)


MODEL = None
MODEL_LOCK = threading.Lock()


# ---------- auth ----------
def hash_pw(pw: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    return f"{salt}${hashlib.pbkdf2_hmac('sha256', pw.encode(), bytes.fromhex(salt), 200_000).hex()}"


def check_pw(pw: str, stored: str) -> bool:
    return secrets.compare_digest(hash_pw(pw, stored.split("$")[0]), stored)


def current_user(authorization: str = Header(default="")) -> int:
    token = authorization.removeprefix("Bearer ").strip()
    with db() as c:
        r = c.one("SELECT user_id FROM sessions WHERE token=%s", (token,))
    if not r:
        raise HTTPException(401, "Not logged in")
    return r["user_id"]


class Register(BaseModel):
    name: str = Field(min_length=1)
    program: str = "Student"
    email: str = Field(min_length=3)
    password: str = Field(min_length=6)


class Login(BaseModel):
    email: str
    password: str


def new_session(c, uid: int) -> str:
    t = secrets.token_urlsafe(32)
    c.run("INSERT INTO sessions(token,user_id) VALUES(%s,%s)", (t, uid))
    return t


@asynccontextmanager
async def lifespan(_app):
    global MODEL
    init_db()
    MODEL = ml.load()
    if os.environ.get("AIGENDA_MAINTENANCE", "1") != "0":
        threading.Thread(target=maintenance_loop, daemon=True).start()
    yield


app = FastAPI(title="AIgenda API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/api/register")
def register(b: Register):
    email = b.email.strip().lower()
    with db() as c:
        if c.one("SELECT 1 x FROM users WHERE email=%s", (email,)):
            raise HTTPException(409, "Email already registered. Log in instead.")
        uid = c.run("INSERT INTO users(name,program,email,password_hash,is_test) VALUES(%s,%s,%s,%s,%s)",
                    (b.name, b.program, email, hash_pw(b.password), int(email.endswith(TEST_EMAIL_DOMAIN))))
        c.run("INSERT INTO habit_profiles(user_id) VALUES(%s)", (uid,))
        return {"token": new_session(c, uid), "name": b.name, "program": b.program}


@app.post("/api/login")
def login(b: Login):
    with db() as c:
        u = c.one("SELECT * FROM users WHERE email=%s", (b.email.strip().lower(),))
        if not u or not check_pw(b.password, u["password_hash"]):
            raise HTTPException(401, "Wrong email or password")
        return {"token": new_session(c, u["id"]), "name": u["name"], "program": u["program"]}


@app.post("/api/logout")
def logout(authorization: str = Header(default="")):
    with db() as c:
        c.run("DELETE FROM sessions WHERE token=%s", (authorization.removeprefix("Bearer ").strip(),))
    return {"ok": True}


@app.get("/api/me")
def me(uid: int = Depends(current_user)):
    with db() as c:
        u = c.one("SELECT name,program FROM users WHERE id=%s", (uid,))
    return {"name": u["name"], "program": u["program"]}


# ---------- habits & hobbies (UC-02) ----------
class Habits(BaseModel):
    wake: int = Field(ge=0, le=23)
    sleep: int = Field(ge=0, le=23)
    cs: int = Field(ge=0, le=23)
    ce: int = Field(ge=0, le=23)
    chrono: str = Field(pattern="^(morning|afternoon|evening)$")
    hobbies: list[str] = []
    adherence: Optional[int] = Field(default=None, ge=1, le=5)      # "I follow my study schedule consistently"
    distraction: Optional[int] = Field(default=None, ge=1, le=5)    # "I get distracted by social media"
    stress_sleep: Optional[int] = Field(default=None, ge=1, le=5)   # "Stress/lack of sleep affects my focus"


def load_habits(c, uid):
    p = c.one("SELECT * FROM habit_profiles WHERE user_id=%s", (uid,))
    hob = [r["name"] for r in c.all("SELECT name FROM hobbies WHERE user_id=%s ORDER BY id", (uid,))]
    return {"wake": p["wake_hour"], "sleep": p["sleep_hour"], "cs": p["class_start"], "ce": p["class_end"],
            "chrono": p["chronotype"], "hobbies": hob, "adherence": p["adherence"],
            "distraction": p["distraction"], "stress_sleep": p["stress_sleep"]}


@app.get("/api/habits")
def get_habits(uid: int = Depends(current_user)):
    with db() as c:
        return load_habits(c, uid)


@app.put("/api/habits")
def put_habits(b: Habits, uid: int = Depends(current_user)):
    if b.sleep <= b.wake:
        raise HTTPException(422, "Sleep time must be after wake time")
    with db() as c:
        c.run("UPDATE habit_profiles SET wake_hour=%s,sleep_hour=%s,class_start=%s,class_end=%s,chronotype=%s,"
              "adherence=%s,distraction=%s,stress_sleep=%s WHERE user_id=%s",
              (b.wake, b.sleep, b.cs, b.ce, b.chrono, b.adherence, b.distraction, b.stress_sleep, uid))
        c.run("DELETE FROM hobbies WHERE user_id=%s", (uid,))
        c.many("INSERT IGNORE INTO hobbies(user_id,name) VALUES(%s,%s)", [(uid, h) for h in b.hobbies])
        return load_habits(c, uid)


# ---------- subjects (UC-03) ----------
class Subject(BaseModel):
    name: str = Field(min_length=1)


def list_subjects(uid):
    with db() as c:
        return c.all("SELECT id,name,subject_rank AS `rank` FROM subjects WHERE user_id=%s ORDER BY subject_rank", (uid,))


@app.get("/api/subjects")
def get_subjects(uid: int = Depends(current_user)):
    return list_subjects(uid)


@app.post("/api/subjects")
def add_subject(b: Subject, uid: int = Depends(current_user)):
    with db() as c:
        n = c.one("SELECT COUNT(*) n FROM subjects WHERE user_id=%s", (uid,))["n"]
        if n >= MAX_SUBJECTS:
            raise HTTPException(422, f"You can list at most {MAX_SUBJECTS} subjects")
        try:
            c.run("INSERT INTO subjects(user_id,name,subject_rank) VALUES(%s,%s,%s)", (uid, b.name.strip(), n + 1))
        except pymysql.err.IntegrityError:
            raise HTTPException(409, "Subject already listed")
    return list_subjects(uid)


@app.put("/api/subjects/order")
def reorder(ids: list[int], uid: int = Depends(current_user)):
    """Body: subject ids from highest to lowest priority."""
    with db() as c:
        for rank, sid in enumerate(ids, 1):
            c.run("UPDATE subjects SET subject_rank=%s WHERE id=%s AND user_id=%s", (rank, sid, uid))
    return list_subjects(uid)


@app.delete("/api/subjects/{sid}")
def del_subject(sid: int, uid: int = Depends(current_user)):
    with db() as c:
        c.run("DELETE FROM subjects WHERE id=%s AND user_id=%s", (sid, uid))
        for rank, r in enumerate(c.all("SELECT id FROM subjects WHERE user_id=%s ORDER BY subject_rank", (uid,)), 1):
            c.run("UPDATE subjects SET subject_rank=%s WHERE id=%s", (rank, r["id"]))
    return list_subjects(uid)


# ---------- weekly plan: subjects + vacant time per day, re-asked every 7 days ----------
class ClassSlot(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    start: Optional[int] = Field(default=None, ge=0, le=23)   # optional class time; blank = just a subject to study
    end: Optional[int] = Field(default=None, ge=1, le=24)


class Window(BaseModel):
    start: int = Field(ge=0, le=23)
    end: int = Field(ge=1, le=24)


class DayPlan(BaseModel):
    day: date
    subjects: list[ClassSlot] = Field(default=[], max_length=MAX_SUBJECTS)
    vacant: list[Window] = Field(default=[], max_length=8)


class WeekPlan(BaseModel):
    days: list[DayPlan] = Field(min_length=1, max_length=PLAN_DAYS)


def today_of(c) -> date:
    return c.one("SELECT CURDATE() d")["d"]


def load_plan(c, uid, day) -> Optional[dict]:
    r = c.one("SELECT data FROM weekly_plans WHERE user_id=%s AND day=%s", (uid, day))
    return json.loads(r["data"]) if r else None


def plan_status(c, uid) -> dict:
    """needs_prompt is True when no plan exists or the last one has run out (7 days since it was submitted)."""
    today = today_of(c)
    last = c.one("SELECT MAX(created_on) d FROM weekly_plans WHERE user_id=%s", (uid,))["d"]
    next_due = last + timedelta(days=PLAN_DAYS) if last else today
    rows = c.all("SELECT day,data FROM weekly_plans WHERE user_id=%s AND day >= %s ORDER BY day LIMIT 14", (uid, today))
    prev = c.all("SELECT day,data FROM weekly_plans WHERE user_id=%s AND created_on=%s ORDER BY day", (uid, last)) if last else []
    return {"today": str(today), "has_plan": last is not None,
            "previous": [{"day": str(r["day"]), **json.loads(r["data"])} for r in prev], "submitted_on": str(last) if last else None,
            "next_prompt_on": str(next_due), "needs_prompt": today >= next_due,
            "days_left": max(0, (next_due - today).days),
            "plan": [{"day": str(r["day"]), **json.loads(r["data"])} for r in rows]}


def clean_day(d: DayPlan) -> dict:
    subs, seen = [], set()
    for s in d.subjects:
        name = s.name.strip()
        if not name or name.lower() in seen:
            continue
        if (s.start is None) != (s.end is None):
            raise HTTPException(422, f"{name}: give both a start and an end time, or leave both blank")
        if s.start is not None and s.end <= s.start:
            raise HTTPException(422, f"{name} on {d.day}: end time must be after start time")
        seen.add(name.lower())
        subs.append({"name": name, "start": s.start, "end": s.end})
    for w in d.vacant:
        if w.end <= w.start:
            raise HTTPException(422, f"Vacant time on {d.day}: end must be after start")
    return {"subjects": subs, "vacant": [{"start": w.start, "end": w.end} for w in d.vacant]}


def build_day(c, uid, day: date, h: dict, global_subs: list[dict], plan: Optional[dict]) -> tuple[list[dict], list[str]]:
    """Generate and store one day's schedule. Returns (study subject list, unplaced subjects)."""
    classes = vacant = None
    if plan is not None:                                         # that day's own subjects and vacant time win
        rank_of = {s["name"].lower(): s["rank"] for s in global_subs}
        top = max(rank_of.values(), default=0)
        subs = []
        for i, s in enumerate(plan["subjects"]):                  # ranked subjects keep their priority, new ones follow in order
            subs.append({"name": s["name"], "rank": rank_of.get(s["name"].lower(), top + i + 1)})
        classes = [(s["start"], s["end"], "Class · " + s["name"]) for s in plan["subjects"] if s["start"] is not None]
        vacant = [(w["start"], w["end"]) for w in plan["vacant"]] or None
        if not subs:
            subs = list(global_subs)                              # nothing entered for this day: fall back to the ranked list
    else:
        subs = list(global_subs)
    # Estimate the day's load for the model, then place blocks (the final count is what gets logged later).
    est = scheduler.n_study_blocks(len(subs))
    scorer = ml.scorer_from(MODEL, model_profile(h, est))
    result, unplaced = scheduler.plan_day(h, subs, h["hobbies"], scorer, classes, vacant)
    rank_of = {s["name"]: s["rank"] for s in subs}
    c.run("DELETE FROM schedule_items WHERE user_id=%s AND day=%s", (uid, day))
    c.many("INSERT INTO schedule_items(user_id,day,hour,label,kind,subject_rank) VALUES(%s,%s,%s,%s,%s,%s)",
           [(uid, day, x["hour"], x["label"], x["kind"], rank_of.get(x["label"]) if x["kind"] == "study" else None)
            for x in result])
    return subs, unplaced


@app.get("/api/weekly/status")
def weekly_status(uid: int = Depends(current_user)):
    """Frontend calls this on login: needs_prompt tells it to ask for the new week's subjects and vacant time."""
    with db() as c:
        return plan_status(c, uid)


@app.put("/api/weekly/plan")
def save_weekly_plan(b: WeekPlan, uid: int = Depends(current_user)):
    """Save the 7-day plan and generate a schedule for each day from it."""
    with db() as c:
        today = today_of(c)
        h = load_habits(c, uid)
        global_subs = c.all("SELECT name,subject_rank AS `rank` FROM subjects WHERE user_id=%s", (uid,))
        days = {}
        for d in b.days:
            if d.day < today or d.day >= today + timedelta(days=PLAN_DAYS):
                raise HTTPException(422, f"{d.day} is outside the next {PLAN_DAYS} days")
            days[d.day] = clean_day(d)
        if not any(p["subjects"] for p in days.values()):
            raise HTTPException(422, "Add at least one subject on at least one day")
        c.run("DELETE FROM weekly_plans WHERE user_id=%s AND day >= %s", (uid, today))
        report = []
        for day in sorted(days):
            c.run("INSERT INTO weekly_plans(user_id,day,data,created_on) VALUES(%s,%s,%s,%s)",
                  (uid, day, json.dumps(days[day]), today))
            if days[day]["subjects"] or global_subs:
                _, unplaced = build_day(c, uid, day, h, global_subs, days[day])
                report.append({"day": str(day), "unplaced": unplaced})
        return {**plan_status(c, uid), "report": report}


# ---------- schedule (UC-04, UC-05) ----------
def items(c, uid, day=None):
    day = day or today_of(c)
    return c.all("SELECT id,hour,label,kind,done FROM schedule_items WHERE user_id=%s AND day=%s ORDER BY hour", (uid, day))


def model_profile(h, n_blocks):
    """Per-student model inputs: the student's own self-ratings when given, plus the day's planned study load."""
    p = {k: float(h[k]) for k in ("adherence", "distraction", "stress_sleep") if h[k] is not None}
    p["day_load"] = n_blocks
    return p


@app.post("/api/schedule/generate")
def generate(uid: int = Depends(current_user)):
    """(Re)build TODAY's schedule. Uses today's weekly-plan entry when there is one."""
    with db() as c:
        today = today_of(c)
        subs = c.all("SELECT name,subject_rank AS `rank` FROM subjects WHERE user_id=%s", (uid,))
        plan = load_plan(c, uid, today)
        if not subs and not (plan and plan["subjects"]):
            raise HTTPException(422, "Add at least one subject before generating a schedule")
        h = load_habits(c, uid)
        build_day(c, uid, today, h, subs, plan)
        rows = items(c, uid, today)
        if not any(x["kind"] == "study" for x in rows):         # UC-04 exception: no feasible schedule
            raise HTTPException(422, "No feasible schedule: there are no free hours outside classes, meals and sleep. "
                                     "Adjust your habits or vacant time and try again.")
        return rows


@app.get("/api/schedule")
def get_schedule(uid: int = Depends(current_user)):
    with db() as c:
        return items(c, uid)


@app.get("/api/schedule/week")
def get_week_schedule(uid: int = Depends(current_user)):
    """Every scheduled day from today on, each row tagged with its date."""
    with db() as c:
        rows = c.all("SELECT id,day,hour,label,kind,done FROM schedule_items WHERE user_id=%s AND day >= CURDATE() "
                     "AND day < CURDATE() + INTERVAL 7 DAY ORDER BY day,hour", (uid,))
    return [{**r, "day": str(r["day"])} for r in rows]


@app.get("/api/schedule/summary")
def schedule_summary(uid: int = Depends(current_user)):
    """Numbers for the 'Review your schedule' step: study/hobby hours and the Random Forest's predicted completion."""
    with db() as c:
        today = today_of(c)
        h = load_habits(c, uid)
        rows = c.all("SELECT hour,kind,label,subject_rank FROM schedule_items WHERE user_id=%s AND day=%s", (uid, today))
        glob = c.all("SELECT name FROM subjects WHERE user_id=%s", (uid,))
        plan = load_plan(c, uid, today)
    wanted = [s["name"] for s in plan["subjects"]] if plan and plan["subjects"] else [s["name"] for s in glob]
    ps, pe = scheduler.PEAK[h["chrono"]]
    study = [r for r in rows if r["kind"] == "study"]
    score = ml.scorer_from(MODEL, model_profile(h, len(study)))
    pc = [score(r["hour"], int(ps <= r["hour"] < pe), r["subject_rank"] or 3) for r in study] if score else []
    got = {r["label"] for r in study}
    return {"study_hours": len(study), "hobby_hours": sum(1 for r in rows if r["kind"] == "hobby"),
            "study_blocks_in_focus_window": sum(1 for r in study if ps <= r["hour"] < pe),
            "predicted_completion": round(100 * sum(pc) / len(pc)) if pc else None,
            "focus_window": [ps, pe], "free_blocks": sum(1 for r in rows if r["kind"] in ("rest", "hobby", "study")),
            "ranked_subjects": len(wanted), "unscheduled": [n for n in wanted if n not in got]}


class Patch(BaseModel):
    done: Optional[bool] = None
    hour: Optional[int] = Field(default=None, ge=0, le=23)     # reschedule to another hour (UC-05)
    label: Optional[str] = Field(default=None, min_length=1, max_length=120)


@app.patch("/api/schedule/{iid}")
def patch_item(iid: int, b: Patch, uid: int = Depends(current_user)):
    with db() as c:
        cur = c.one("SELECT id,hour,day FROM schedule_items WHERE id=%s AND user_id=%s", (iid, uid))
        if not cur:
            raise HTTPException(404, "Not found")
        if b.hour is not None and b.hour != cur["hour"]:
            clash = c.one("SELECT label FROM schedule_items WHERE user_id=%s AND day=%s AND hour=%s AND id<>%s",
                          (uid, cur["day"], b.hour, iid))
            if clash:                                              # UC-05 exception: flag the conflict
                raise HTTPException(409, f"Conflict: {clash['label']} is already scheduled at that hour")
        sets, args = ["source='manual'"], []
        for col, val in (("done", None if b.done is None else int(b.done)), ("hour", b.hour), ("label", b.label)):
            if val is not None:
                sets.append(f"{col}=%s"); args.append(val)
        c.run(f"UPDATE schedule_items SET {','.join(sets)} WHERE id=%s", (*args, iid))
        return items(c, uid)


@app.delete("/api/schedule/{iid}")
def del_item(iid: int, uid: int = Depends(current_user)):
    with db() as c:
        c.run("DELETE FROM schedule_items WHERE id=%s AND user_id=%s", (iid, uid))
        return items(c, uid)


# ---------- progress, tips, goals (UC-06, UC-09, UC-10) ----------
@app.get("/api/progress")
def progress(uid: int = Depends(current_user)):
    """UC-06: weekly completion, streak and today's counts. A day counts toward the streak if >= 1 activity was ticked."""
    with db() as c:
        rows = c.all("SELECT day, COUNT(*) total, SUM(done) done FROM schedule_items WHERE user_id=%s AND kind<>'class' "
                     "AND day >= CURDATE() - INTERVAL 30 DAY AND day <= CURDATE() GROUP BY day ORDER BY day", (uid,))
        today = c.one("SELECT CURDATE() d")["d"]
    by_day = {r["day"]: r for r in rows}
    week = [{"day": str(r["day"]), "pct": round(100 * int(r["done"] or 0) / r["total"])}
            for r in rows if (today - r["day"]).days <= 6]
    streak, d = 0, today
    if d in by_day and not int(by_day[d]["done"] or 0):          # today not started yet: don't break the streak
        d = d - timedelta(days=1)
    while d in by_day and int(by_day[d]["done"] or 0) > 0:
        streak += 1
        d = d - timedelta(days=1)
    t = by_day.get(today)
    wd = sum(int(r["done"] or 0) for r in by_day.values() if (today - r["day"]).days <= 6)
    wt = sum(r["total"] for r in by_day.values() if (today - r["day"]).days <= 6)
    return {"week": week, "today_pct": round(100 * int(t["done"] or 0) / t["total"]) if t else 0, "streak": streak,
            "today_done": int(t["done"] or 0) if t else 0, "today_total": t["total"] if t else 0,
            "week_done": wd, "week_total": wt, "week_pct": round(100 * wd / wt) if wt else 0,
            "enough_data": wt >= 3}


@app.get("/api/tips")
def get_tips(uid: int = Depends(current_user)):
    """UC-09: personalized tips/reminders. Encouragement replaces corrective tips when the student is on track."""
    p = progress(uid)
    with db() as c:
        h = load_habits(c, uid)
        subs = c.all("SELECT name,subject_rank AS `rank` FROM subjects WHERE user_id=%s", (uid,))
    return scheduler.tips(h, subs, h["hobbies"], p)


class Ask(BaseModel):
    topic: Optional[str] = None
    question: Optional[str] = Field(default=None, max_length=300)


@app.post("/api/support")
def study_support(b: Ask, uid: int = Depends(current_user)):
    """UC-11 (optional): study tips, resources and time-management advice; general tips when there is little data."""
    p = progress(uid)
    with db() as c:
        h = load_habits(c, uid)
        subs = c.all("SELECT name,subject_rank AS `rank` FROM subjects WHERE user_id=%s", (uid,))
    return scheduler.support(h, subs, p, b.topic or b.question)


class Goal(BaseModel):
    title: str = Field(min_length=1)
    due_date: Optional[str] = None


def list_goals(uid):
    with db() as c:
        return c.all("SELECT id,title,progress,due_date FROM goals WHERE user_id=%s", (uid,))


@app.get("/api/goals")
def get_goals(uid: int = Depends(current_user)):
    return list_goals(uid)


@app.post("/api/goals")
def add_goal(b: Goal, uid: int = Depends(current_user)):
    due = None
    if b.due_date:
        try:
            due = date.fromisoformat(b.due_date)
        except ValueError:
            raise HTTPException(422, "Target date must look like YYYY-MM-DD")
        if due < date.today():                                  # UC-10 exception: target date in the past
            raise HTTPException(422, "Target date cannot be in the past")
    with db() as c:
        if c.one("SELECT COUNT(*) n FROM goals WHERE user_id=%s", (uid,))["n"] >= MAX_GOALS:
            raise HTTPException(422, f"You can have at most {MAX_GOALS} active goals")
        c.run("INSERT INTO goals(user_id,title,due_date) VALUES(%s,%s,%s)", (uid, b.title.strip(), due))
    return list_goals(uid)


@app.patch("/api/goals/{gid}")
def set_goal(gid: int, progress: int, uid: int = Depends(current_user)):
    with db() as c:
        c.run("UPDATE goals SET progress=%s WHERE id=%s AND user_id=%s", (min(100, max(0, progress)), gid, uid))
    return list_goals(uid)


@app.delete("/api/goals/{gid}")
def del_goal(gid: int, uid: int = Depends(current_user)):
    with db() as c:
        c.run("DELETE FROM goals WHERE id=%s AND user_id=%s", (gid, uid))
    return list_goals(uid)


# ---------- the learning loop ----------
def log_outcomes() -> int:
    """Copy the final outcome (done or skipped) of every study block from a PAST day into activity_log.
    Unticked blocks count as skipped, so the data isn't biased toward 'completed'."""
    with db() as c:
        rows = c.all(
            "SELECT s.id, s.user_id, s.day, s.hour, s.subject_rank, s.done, h.chronotype, h.adherence, h.distraction, "
            "h.stress_sleep, (SELECT COUNT(*) FROM schedule_items x WHERE x.user_id=s.user_id AND x.day=s.day "
            "AND x.kind='study') AS day_load "
            "FROM schedule_items s JOIN habit_profiles h ON h.user_id=s.user_id "
            "WHERE s.kind='study' AND s.logged=0 AND s.day < CURDATE()")
        for r in rows:
            ps, pe = scheduler.PEAK[r["chronotype"]]
            c.run("INSERT INTO activity_log(user_id,day,hour,priority,day_load,in_peak,adherence,distraction,stress_sleep,completed) "
                  "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                  (r["user_id"], r["day"], r["hour"], r["subject_rank"], r["day_load"], int(ps <= r["hour"] < pe),
                   r["adherence"], r["distraction"], r["stress_sleep"], int(r["done"])))
            c.run("UPDATE schedule_items SET logged=1 WHERE id=%s", (r["id"],))
        c.run("UPDATE schedule_items SET logged=1 WHERE kind<>'study' AND day < CURDATE() AND logged=0")
        return len(rows)


def retrain_if_due(force: bool = False):
    """Retrain on real, non-test rows once RETRAIN_MIN_ROWS exist and RETRAIN_EVERY new rows have arrived."""
    global MODEL
    with MODEL_LOCK:
        with db() as c:
            rows = c.all("SELECT a.hour,a.in_peak,a.priority,a.day_load,a.adherence,a.distraction,a.stress_sleep,a.completed "
                         "FROM activity_log a JOIN users u ON u.id=a.user_id WHERE u.is_test=0 AND a.kind='study'")
            last = c.one("SELECT COALESCE(MAX(real_rows),0) r FROM model_runs WHERE source='retrain'")["r"]
        if not force and len(rows) - last < RETRAIN_EVERY:
            return None
        bundle = ml.retrain_from_rows([tuple(r.values()) for r in rows], min_rows=RETRAIN_MIN_ROWS)
        if bundle is None:
            return None
        m = bundle["metrics"]
        with db() as c:
            c.run("INSERT INTO model_runs(source,real_rows,test_rows,test_accuracy,test_auc) VALUES('retrain',%s,%s,%s,%s)",
                  (m["real_rows"], m["real_test_rows"], m.get("real_test_accuracy"), m.get("real_test_auc")))
        MODEL = bundle
        log.info("Model retrained on %s real rows: %s", len(rows), m)
        return m


def maintenance_loop():
    while True:
        try:
            log_outcomes()
            retrain_if_due()
        except Exception:
            log.exception("maintenance failed")
        time.sleep(MAINTENANCE_SECONDS)


@app.post("/api/model/retrain")
def retrain_now(uid: int = Depends(current_user)):
    """Log finished days, then retrain immediately (needs RETRAIN_MIN_ROWS real rows)."""
    log_outcomes()
    m = retrain_if_due(force=True)
    if m is None:
        raise HTTPException(422, f"Not enough real completion data yet (need {RETRAIN_MIN_ROWS} study blocks from finished days); keeping the current model")
    return {"status": "retrained", **m}


@app.get("/api/model/status")
def model_status(uid: int = Depends(current_user)):
    with db() as c:
        real = c.one("SELECT COUNT(*) n FROM activity_log a JOIN users u ON u.id=a.user_id WHERE u.is_test=0")["n"]
        runs = c.all("SELECT trained_at,source,real_rows,test_accuracy,test_auc FROM model_runs ORDER BY id DESC LIMIT 5")
    return {"real_rows_logged": real, "retrain_min_rows": RETRAIN_MIN_ROWS, "retrain_every": RETRAIN_EVERY,
            "base_model_metrics": (MODEL or {}).get("metrics"), "recent_runs": runs}


# Serve the frontend at / (mounted last so /api/* wins)
app.mount("/", StaticFiles(directory=os.path.join(HERE, "..", "frontend"), html=True), name="frontend")
