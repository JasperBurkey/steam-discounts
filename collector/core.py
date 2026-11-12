"""HTTP, storage, state and locking shared by every job.

Storage design (for ML use later):
  raw/<job>/<date>/<run_id>.jsonl.gz   every response body exactly as received, plus
                                       request, status and capture time (key redacted)
  tables/<table>/date=<date>/<run_id>.parquet
                                       parsed rows with fixed schemas; every row has
                                       captured_at, so features can be built "as of" any time
  logs/runs.jsonl                      one line per job run: calls, failures, rows, timings
  state/                               schedule state, call budget, known apps, text hashes
"""

import gzip
import json
import os
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd
import requests
import settings as S

for d in (S.RAW, S.TABLES, S.STATE, S.LOGS, S.ASSETS):
    d.mkdir(parents=True, exist_ok=True)


def utcnow():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- state
def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


class Lock:
    """One process per group. A lock older than STALE_LOCK_HOURS is assumed dead."""

    def __init__(self, name):
        self.path = S.STATE / f"{name}.lock"

    def __enter__(self):
        if self.path.exists():
            age_h = (time.time() - self.path.stat().st_mtime) / 3600
            if age_h < S.STALE_LOCK_HOURS:
                raise RuntimeError(
                    f"{self.path.name} held (age {age_h:.1f}h); another run is active"
                )
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------- budget
def calls_today(host="api.steampowered.com"):
    day = utcnow().strftime("%Y-%m-%d")
    total = 0
    for f in S.STATE.glob(f"calls_{day}_*.json"):
        total += read_json(f, {}).get(host, 0)
    return total


# ---------------------------------------------------------------- HTTP
class Http:
    """Rate-limited JSON client. Never raises on HTTP errors: returns (body, status)."""

    def __init__(self, job, run_id):
        self.job, self.run_id = job, run_id
        self.s = requests.Session()
        self.s.headers["User-Agent"] = S.USER_AGENT
        self.last = {}
        self.calls, self.failures = {}, 0
        day = utcnow().strftime("%Y-%m-%d")
        self.budget_file = S.STATE / f"calls_{day}_{job}_{run_id}.json"
        self.raw_path = S.RAW / job / day / f"{run_id}.jsonl.gz"
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw = gzip.open(self.raw_path, "at", encoding="utf-8")

    def close(self):
        self.raw.close()
        write_json(self.budget_file, self.calls)

    def _wait(self, host):
        rate = S.REQS_PER_SEC.get(host, 1.0)
        gap = 1.0 / rate - (time.monotonic() - self.last.get(host, 0))
        if gap > 0:
            time.sleep(gap)
        self.last[host] = time.monotonic()

    def request(
        self,
        method,
        url,
        params=None,
        headers=None,
        data=None,
        save_raw=True,
        tries=5,
        binary=False,
    ):
        host = urlparse(url).netloc
        status, body = None, None
        for attempt in range(tries):
            self._wait(host)
            self.calls[host] = self.calls.get(host, 0) + 1
            try:
                r = self.s.request(
                    method, url, params=params, headers=headers, data=data, timeout=45
                )
                status = r.status_code
            except requests.RequestException as e:
                status = f"error:{type(e).__name__}"
                time.sleep(5 * (attempt + 1))
                continue
            if status == 200:
                body = r.content if binary else _safe_json(r)
                break
            if status in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After")
                time.sleep(
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 20 * (attempt + 1)
                )
                continue
            break  # 4xx other than 429: retrying will not help
        if status != 200:
            self.failures += 1
        if save_raw and not binary:
            self.raw.write(
                json.dumps(
                    {
                        "captured_at": iso(utcnow()),
                        "method": method,
                        "url": url,
                        "params": _redact(params),
                        "status": status,
                        "body": body,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
        return body, status

    def get(self, url, params=None, **kw):
        return self.request("GET", url, params=params, **kw)


def _safe_json(r):
    try:
        return r.json()
    except ValueError:
        return {"_non_json": r.text[:2000]}


def _redact(params):
    if not params:
        return params
    return {
        k: ("REDACTED" if k in ("key", "client_secret") else v)
        for k, v in dict(params).items()
    }


# ---------------------------------------------------------------- tables
def write_table(name, rows, schema, run_id, when=None):
    """Write rows with a fixed schema (column -> pandas dtype). Missing columns become null."""
    if not rows:
        return 0
    when = when or utcnow()
    df = pd.DataFrame(rows)
    for col, dtype in schema.items():
        if col not in df:
            df[col] = None
        if dtype == "datetime":
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        elif dtype in ("Int64", "Float64"):
            # numbers can arrive as strings; anything unparseable becomes null, not a crash
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
        else:
            df[col] = df[col].astype(dtype)
    df = df[list(schema)]
    out = S.TABLES / name / f"date={when:%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{run_id}.parquet", index=False)
    return len(df)


def read_table(name, days=None):
    """Read a table, optionally only the last `days` date partitions."""
    base = S.TABLES / name
    if not base.exists():
        return pd.DataFrame()
    parts = sorted(base.glob("date=*"))
    if days:
        parts = parts[-days:]
    files = [f for p in parts for f in sorted(p.glob("*.parquet"))]
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def latest_run(name):
    """Rows written by the most recent run of a table (run ids sort by time)."""
    base = S.TABLES / name
    files = (
        sorted(base.glob("date=*/*.parquet"), key=lambda f: f.stem)
        if base.exists()
        else []
    )
    return pd.read_parquet(files[-1]) if files else pd.DataFrame()


def log_run(record):
    with open(S.LOGS / "runs.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def new_run_id():
    return f"{utcnow():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"


def chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
