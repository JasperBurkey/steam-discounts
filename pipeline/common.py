"""HTTP with polite rate limiting and retries, plus ID hashing."""

import hashlib
import time

import config
import requests

SESSION = requests.Session()
SESSION.headers["User-Agent"] = config.USER_AGENT


def get_json(url, params=None, sleep=1.0, tries=5):
    """GET a JSON endpoint. Backs off on 429/5xx; returns None on failure."""
    for attempt in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  network error ({e}); retrying")
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 200:
            time.sleep(sleep)
            try:
                return r.json()
            except ValueError:
                print(f"  non-JSON response from {url}")
                return None
        if r.status_code in (429, 500, 502, 503, 504):
            wait = 15 * (attempt + 1)
            print(f"  HTTP {r.status_code}; waiting {wait}s")
            time.sleep(wait)
            continue
        print(f"  HTTP {r.status_code} from {url}; giving up on this request")
        return None
    return None


def hash_id(raw_id):
    """One-way hash so reviewers can be linked across games but not identified."""
    return hashlib.sha256((config.SALT + str(raw_id)).encode()).hexdigest()[:16]


def study_games(sample=None, n=None):
    """The study sample. Every script calls this, so reviews, prices and the analysis always
    use the same games.

    Full project: the stratified sample in study_sample.csv (built by 01b), if it exists.
    MVP: a reproducible random draw of mid-sized games from sample.csv.
    """
    import pandas as pd  # local import keeps common.py light for scripts that do not need it

    if config.STUDY_SAMPLE_CSV.exists():
        return pd.read_csv(config.STUDY_SAMPLE_CSV)
    n = n or config.MAX_REVIEW_APPS
    mid = sample[sample["mid_sized"]]
    return mid.sample(n=min(n, len(mid)), random_state=config.SAMPLE_SEED).reset_index(
        drop=True
    )
