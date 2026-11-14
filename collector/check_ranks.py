"""Live check of the top-seller ranking job, before relying on it. About 12 requests.

  .venv\\Scripts\\python check_ranks.py

Tests:
  1. Steam's response can be parsed (rows, app IDs, names, prices)
  2. How deep the global list goes (pages near 10,000)
  3. Stability: two fetches a minute apart give (nearly) the same order
  4. A tag-filtered ranking works
Send the full output to Claude.
"""

import time

import parse as P
from core import Http, new_run_id, utcnow
from jobs import _search_page

http = Http("check_ranks", new_run_id())
now = utcnow()
problems = []


def page(start, count=100, tagid=None):
    body, status = _search_page(http, start, count, tagid)
    rows = P.search_rank_rows(body, now, "check", tagid, start) if status == 200 else []
    return rows, status, (body or {})


# 1. parsing
rows, status, body = page(0)
keys = (
    sorted((body or {}).keys())[:8] if isinstance(body, dict) else type(body).__name__
)
html = (body or {}).get("results_html") or "" if isinstance(body, dict) else ""
print(f"1. First page: HTTP {status}; response keys {keys}")
print(
    f"   Steam says total_count {(body or {}).get('total_count')}; HTML length {len(html):,} characters"
)
print(f"   {len(rows)} rows parsed")
with open(
    "ranks_sample.html", "w", encoding="utf-8"
) as f:  # for diagnosis if parsing fails
    f.write(html)
if not rows:
    print("   Start of the HTML (send this to Claude):")
    print("   " + html[:1500].replace("\n", " ") if html else "   (empty)")
if rows:
    with_app = sum(r["appid"] is not None for r in rows)
    with_price = sum(r["price_final_cents"] is not None for r in rows)
    print(
        f"   app IDs found for {with_app}/{len(rows)}, prices for {with_price}/{len(rows)}"
    )
    for r in rows[:5]:
        print(
            f"   #{r['rank']}: {r['name']} (app {r['appid']}, {r['discount_pct']}% off)"
        )
if len(rows) < 50 or not rows or rows[0]["appid"] is None:
    problems.append("first page did not parse as expected")

# 2. depth
for start in (4900, 9900, 10000):
    r, st, _ = page(start)
    print(
        f"2. Page starting at {start + 1:,}: HTTP {st}, {len(r)} rows"
        + (f", first: {r[0]['name']}" if r else "")
    )
    if start == 9900 and len(r) < 50:
        problems.append("global list does not reach 10,000 places")

# 3. stability
print("3. Waiting 60 seconds, then fetching the first page again...", flush=True)
time.sleep(60)
again, _, _ = page(0)
a = [r["itemkey"] for r in rows]
b = [r["itemkey"] for r in again]
same_set = len(set(a) & set(b)) / max(len(a), 1)
same_pos = sum(x == y for x, y in zip(a, b)) / max(len(a), 1)
print(
    f"   Top 100: {same_set:.0%} the same games, {same_pos:.0%} in exactly the same position"
)
if same_set < 0.9:
    problems.append("ranking changed a lot within a minute (noisy)")

# 4. a tag ranking (492 = Indie, 1716 = Roguelike)
for tagid in (1716,):
    r, st, _ = page(0, tagid=tagid)  # noqa: F841
    print(
        f"4. Tag {tagid}: HTTP {st}, {len(r)} rows, total_count {_.get('total_count') if isinstance(_, dict) else None}"
        + (f"; #1 {r[0]['name']}" if r else "")
    )
    if len(r) < 50:
        problems.append(f"tag {tagid} ranking did not parse")

http.close()
print()
if problems:
    print("NEEDS ATTENTION: " + "; ".join(problems) + ". Send this output to Claude.")
else:
    print(
        "PASS: rankings parse, reach 10,000 places, are stable, and tag rankings work."
    )
    print("Next: .venv\\Scripts\\python collect.py run --job ranks --force")
