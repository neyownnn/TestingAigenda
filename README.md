# AIgenda

AI-driven web app that builds a personalized student schedule (rule-based heuristic + Random Forest).

```
aigenda/
├── frontend/index.html      Single-page web app (HTML/CSS/JS, no build step)
└── backend/
    ├── schema.sql           MySQL/MariaDB schema (auto-created)
    ├── main.py              FastAPI REST API + learning loop
    ├── smoke_test.py        End-to-end test of the loop against MySQL
    ├── scheduler.py         Rule-based AI Scheduler + tips
    ├── prep_survey.py       Step 1: survey .xlsx -> data/survey_profiles.csv
    ├── synth_data.py        Step 2: survey-seeded synthetic data -> data/synthetic_activity.csv
    ├── ml.py                Step 3: train / evaluate / retrain the Random Forest
    ├── rf_model.joblib      Pre-trained model (ships ready to use)
    ├── data/                survey_responses.xlsx, survey_profiles.csv, synthetic_activity.csv, model_metrics.json
    └── requirements.txt
```

## Run with XAMPP (MySQL)
1. Open the XAMPP Control Panel and **Start MySQL** (Apache is not needed). Defaults work as-is: host 127.0.0.1, port 3306, user `root`, empty password.
2. Start the backend:
```bash
cd backend
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```
The `aigenda` database and all tables are created automatically on first start (browse them in phpMyAdmin at http://localhost/phpmyadmin).
Open http://localhost:8000. API docs: http://localhost:8000/docs
If your MySQL has a different user/password, set `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME` before starting.

The model ships pre-trained (`rf_model.joblib`). To rebuild it from the survey: `python prep_survey.py && python synth_data.py && python ml.py`.

## Alignment with the proposal deck
| Deck item | In the app |
|---|---|
| UC-01 Create account / Log in | Create account (confirm password) and Log in tabs; sessions via `/api/register`, `/api/login`, `/api/logout` |
| UC-02 Habits & hobbies | "Your usual day" page: sleep, class times, focus window, hobbies (add your own), optional 1-5 self-ratings; autosaves |
| UC-03 Subjects & ranking | Ranked list with up/down, max 15 subjects, blocks schedule generation when empty |
| UC-04 Generate schedule | 3-step screen (Analyze, Optimize with AI, Review) with study/hobby hours and Random Forest predicted completion (`/api/schedule/summary`); 422 when no feasible schedule |
| UC-05 View & manage | Tick done, move to another hour (409 on conflict), delete, reset to AI version |
| UC-06 / UC-09 Progress & tips | Weekly bars, streak, today/week %; encouragement instead of corrective tips when on track; "not enough data" message |
| UC-10 Goals (optional) | Target date (past dates rejected), max 10 goals, +10% progress |
| UC-11 AI Study Support (optional) | Tips, resource links (educational only), time-management advice, topic chips and a question box; general tips when data is thin |

## Weekly plan (new)

Every 7 days AIgenda asks for the coming week: for each day, the **subjects** the student has (optional class start/end) and their **vacant time**.

- `GET /api/weekly/status` returns `needs_prompt` (true when there is no plan, or 7 days have passed since the last one), `next_prompt_on`, and last week's plan so the form can prefill by weekday. The frontend opens **Weekly Plan** automatically when it is true and shows a dashboard banner.
- `PUT /api/weekly/plan` saves the 7 days (table `weekly_plans`) and generates a schedule for each day. The response lists, per day, any subject that did not fit (`report[].unplaced`).
- `GET /api/schedule/week` returns all scheduled days; **My Schedule** has a tab per day.
- Study and hobby blocks are placed only inside the vacant windows (no windows = any free hour). A day with no subjects entered falls back to the ranked subject list.

**Bug fix:** the scheduler gave the top subjects two blocks each and ran out of free hours after about 4 subjects, so the rest got nothing. Now every subject gets one block first, spare hours then go to second blocks, and anything that still cannot fit is reported to the student instead of silently dropped. Random Forest scoring is also batched (one model call per day), so generating a week takes about a second. Run `python test_scheduler.py` to check this without a database.

Not in this version: a "late night" focus option, meal-time inputs and subject due dates (all shown in the deck mockups).

## How the model learns from the database
1. Students fill in habits, hobbies and ranked subjects, then generate a schedule (`schedule_items`; study rows store the subject rank).
2. They tick blocks as done. Nothing is learned yet, because ticks can be undone.
3. Once a day is over, `log_outcomes()` copies **every** study block's final result into `activity_log`, with unticked blocks recorded as **skipped**. This runs hourly in the background (`MAINTENANCE_SECONDS`) and on `POST /api/model/retrain`.
4. When `RETRAIN_MIN_ROWS` (200) real rows exist, then each time `RETRAIN_EVERY` (50) new ones arrive, the Random Forest retrains on real rows (weight 5) blended with the synthetic base. The run is saved in `model_runs`, and accuracy/AUC are measured on a held-out 20% of the **real** rows.
5. The new model is used by the very next schedule generation. `GET /api/model/status` shows rows logged and recent runs.

Accounts with an email ending `@test.local` are flagged `is_test` and **never used for training**. Use them for your own testing.

Check everything end-to-end (uses a throwaway database `aigenda_test`): `python smoke_test.py`

## How the synthetic data was made
The survey (n = 9) has Likert answers but no hour-by-hour completion logs, so the data is simulated:
- Each of 600 synthetic students copies a real respondent's profile with +/-1 jitter on every item.
- 30 days x 2-6 study blocks each -> ~72k rows (hour, in focus window, subject priority, daily load, adherence, distraction, stress/sleep -> completed).
- Completion probability is a logistic model whose directions come from the survey items and literature:
  schedule adherence (+), focus-window slot (+, chronotype studies), lower subject priority (-), crowded day (-),
  social-media distraction in the evening (-), stress/lack of sleep in late hours (-), last-minute habit on heavy days (-, Steel 2007).
- Assumptions to state in your paper: coefficient sizes are chosen by the authors, and chronotype was not in the survey so it is sampled (35/30/35 morning/afternoon/evening).
- Sanity check: the generator's source data reproduces the deck's finding that 3 of 9 follow their schedule consistently.

## Model quality (held-out synthetic students)
Accuracy 0.64 vs 0.58 majority baseline, ROC AUC 0.68 (`data/model_metrics.json`). This only shows the model learned the simulated
behavior; it is not evidence about real students. Real validation needs real logs.

## Notes
- `frontend/index.html` now talks to the API for everything (accounts, habits, subjects, schedule, progress, goals, study support);
  only the login token is kept in the browser. Open it through the backend (http://localhost:8000), not by double-clicking the file.
- Per-student self-ratings (`adherence`, `distraction`, `stress_sleep`, 1-5) are optional on `PUT /api/habits`; when given, they are stored and used by the model, otherwise survey averages are used.
- `POST /api/model/retrain` is open to any logged-in user; restrict it to an admin before deploying.
- Passwords are hashed with PBKDF2-SHA256. For production, restrict CORS origins and use HTTPS.
