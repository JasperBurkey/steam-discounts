# Steam data collector

Records the parts of Steam that disappear if nobody saves them: prices and discounts for
every game and DLC, review summaries, tags, store pages, player counts, charts, front-page
placement, and Twitch viewership. It replaces `02_snapshot.py` from the MVP pipeline; the
MVP's review-history, event-study and clustering scripts still work alongside it.

See `DATA_DICTIONARY.md` for every table and column.

## Setup (about 15 minutes)

1. **Python 3.10+** and dependencies: `pip install -r requirements.txt`
2. **Steam Web API key** (free): log in at <https://steamcommunity.com/dev/apikey>, enter any
   domain name (e.g. `localhost`), and copy the key.
3. `cp .env.example .env` and paste the key after `STEAM_API_KEY=`.
4. Optional, but do it now if you can: **Twitch** keys (register an application at
   <https://dev.twitch.tv/console>, category "Analytics Tool", OAuth redirect
   `http://localhost`) and an **alert** URL (pick a long random topic on <https://ntfy.sh>
   and install its phone app). Twitch viewer counts can never be backfilled.
5. **Run the self-test:** `python collect.py selftest`. It makes one small request to each
   endpoint and prints what came back, e.g. current prices and review counts for CS2 and
   Stardew Valley. Every line should say PASS before you schedule anything.
6. **First run, by hand,** to seed the data:
   ```
   python collect.py run --job universe --force
   python collect.py run --job charts --force
   python collect.py run --job items --force       # about 1.5 h, plus up to an hour of capsule images the first time
   python collect.py status
   ```
7. **Schedule it** (below).

## Windows setup (recommended path for a home PC)

From the project folder, in PowerShell:

```
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env        # then fill in the keys with Notepad
.venv\Scripts\python collect.py selftest
.venv\Scripts\python collect.py run --job universe --force
.venv\Scripts\python collect.py run --job charts --force
.venv\Scripts\python collect.py run --job items --force
```

Then, in PowerShell **as Administrator**, from the same folder:

```
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1 -BackupDir E:\steam_backup      # use your backup drive; omit -BackupDir to skip
```

This registers three hourly tasks under Task Scheduler > SteamCollector that run whether
or not you are logged in, start late if a time was missed, and wake the PC if it is asleep.
It also disables sleep on AC power and, with `-BackupDir`, adds a nightly copy of `data/`
that never deletes anything at the destination. Logs go to `data\logs\task_*.log`.
`remove_windows_tasks.ps1` undoes the tasks. Keep the project and data out of OneDrive or
other synced folders.

## Scheduling (Mac and Linux)

Cron (or Task Scheduler) runs three groups every hour. Each run does only what is due and
exits, so a missed hour or a reboot is harmless: an overdue job runs at the next check, and
its log records how late.

Linux or Mac (`crontab -e`; adjust the paths):
```
5  * * * * cd /path/to/steam_collector && /usr/bin/python3 collect.py run --group fast  >> data/logs/cron_fast.log 2>&1
10 * * * * cd /path/to/steam_collector && /usr/bin/python3 collect.py run --group daily >> data/logs/cron_daily.log 2>&1
15 * * * * cd /path/to/steam_collector && /usr/bin/python3 collect.py run --group ccu   >> data/logs/cron_ccu.log 2>&1
```
Windows: three Task Scheduler tasks, each "Daily, repeat every 1 hour", running
`python collect.py run --group <name>` with "Start in" set to this folder.

| Group | Jobs (UTC) | Typical time per run |
| --- | --- | --- |
| fast | featured and Twitch every 6h; charts 04/16; top sellers 20:00 | a few minutes |
| daily | universe 19:00; store snapshot and regional prices 20:00; weekly on Wednesdays | about 2 h; weekly a few hours |
| ccu | tier A 04/16; tiers B and C from 06:00 | tier A 1-2 h; B and C several hours |

The daily price snapshot runs at 20:00 UTC because Steam sales and price changes start at
10:00 Pacific, so 20:00 UTC always sees that day's prices.

## Where to run it

This needs a machine that is on all day, every day, for years. A laptop that sleeps or
travels will leave gaps, and gaps cannot be filled later. Options, roughly in order:

- **A small cloud VM** (any provider's smallest Linux instance, or a free tier). Put
  `STEAM_DATA_DIR` on attached storage.
- **A Raspberry Pi or old desktop** at home with an external drive.
- **Your laptop,** only to start tonight while you set up one of the above. Then copy
  `data/` to the new machine and keep going.

Back up `data/` regularly (UW storage, cloud storage, or a second drive). It is the only copy
of something that cannot be recreated.

## Call budget

Steam's Web API terms allow 100,000 calls per day. The collector counts its own calls and
stops tier B and C player counts before exceeding `DAILY_STEAM_API_BUDGET` (90,000), trimming
the least-reviewed games first. Rough daily usage once tiers settle:

| Item | Calls/day (approx.) |
| --- | --- |
| Store snapshot, all games and DLC (100 per call) | 2,000-3,000 |
| Regional prices, tier A, 5 countries | ~600 |
| Player counts, tier A twice daily | ~2 x tier A size |
| Player counts, tier B daily, tier C rotation | the remainder, capped by the budget |
| Charts, top sellers, tag list | under 100 |

`python collect.py status` shows today's count. If tier B is regularly trimmed, raise
`TIER_B_MIN_REVIEWS` or reduce tier A in `settings.py`.

## Storage (rough estimates)

Parsed tables: tens of MB per day. Raw responses: the daily store snapshot dominates, on the
order of 50-150 MB per day compressed, so roughly 20-50 GB per year. Raw data is insurance
for fields not parsed today; after a year you can decide whether to keep it. Set
`SAVE_RAW_ITEMS = False` in `settings.py` to drop it if disk is tight.

## Monitoring

- `python collect.py status`: last success and last error per job, and today's calls.
- `python collect.py health`: flags jobs that are overdue, failing, or writing zero rows. It
  also runs after every scheduled run and sends an alert (at most one per issue per 12h) if
  `ALERT_WEBHOOK_URL` is set.
- `data/logs/runs.jsonl`: one line per run with calls, failures, rows, duration and lateness.

Steam's store endpoints are undocumented and occasionally change. If an alert fires, run
`python collect.py selftest` to see which endpoint broke.

## Reading the data

```python
import pandas as pd

items = pd.read_parquet("data/tables/store_items")  # adds a `date` column from folders
ccu = pd.read_parquet("data/tables/ccu", filters=[("date", ">=", "2026-12-01")])
```

## Tests

`python -m unittest discover -s tests -v` runs every job against a fake Steam server and
checks batching, tiers, budget trimming, schemas, text dedupe, capsule dedupe and scheduling.
It does not contact Steam; the self-test does that.
