# Data dictionary

Every table lives in `data/tables/<table>/date=YYYY-MM-DD/<run_id>.parquet`. Every row has
`captured_at` (UTC, the moment of the request), so features can be built as of any point in
time without leaking later information. Nested values are JSON strings (`json.loads`).
Enum fields are stored as strings as Steam sends them.

Raw responses are kept in `data/raw/<job>/<date>/<run_id>.jsonl.gz` (API keys redacted), so
fields not parsed today can be extracted later.

"Recoverable later?" is the reason each item is collected: anything marked **No** is lost
for a given day if it is not captured that day.

## Daily store snapshot: `store_items` (every game and DLC, 20:00 UTC)

| Column | Meaning | Recoverable later? |
| --- | --- | --- |
| `appid`, `name`, `type`, `parent_appid` | Identity; `type` 0 = game, 4 = DLC; DLC's base game | Yes |
| `http_status`, `success`, `visible`, `unlisted`, `unavailable_for_country` | Whether the fetch worked and the store page was live. `failed` = no data, not "zero" | No |
| `is_free`, `is_early_access` | Business model and status on that day | No |
| `steam_release_date`, `original_*`, `is_coming_soon`, `is_preload`, `custom_release_date_message`, `coming_soon_display` | Release info as shown that day; captures release-date slips for upcoming games | No |
| `platform_*`, `vr_*`, `deck_compat` | Platforms; Steam Deck rating (changes over time) | No |
| `review_count`, `review_pct_positive`, `review_score` | Review summary with review bombs filtered out | Partly (rebuild from individual reviews, minus deleted ones) |
| `review_count_unfiltered`, `review_pct_positive_unfiltered` | Present only while Steam flags a review bomb | No |
| `price_final_cents`, `price_original_cents`, `discount_pct` | US price and discount | Partly (third-party histories) |
| `discount_end_date`, `discount_descriptions` | When the discount ends, and its event label (e.g. a seasonal sale) | No |
| `lowest_recent_price_cents`, `hide_discount_pct_for_compliance` | Steam's 30-day lowest price (EU-style rules) | No |
| `purchase_*`, `formatted_final_price` | Which package or bundle is the cheapest way to buy | No |
| `developers`, `publishers`, `franchises` | Credits as listed that day | Mostly |
| `capsule_headline` | Developer's capsule text ("Just updated!", "Free weekend") | No |
| `tags` | Top 20 user tags as `[tagid, weight]` pairs; weights drift over time | No |
| `tagids`, `categories`, `content_descriptorids` | Full tag list; feature categories; content warnings | No (as of date) |
| `asset_*` | Store art filenames; a change means new capsule art | No |
| `free_weekend_start`, `free_weekend_end` | Free-weekend promotions | No |

## Other daily and intraday tables

| Table | When | What | Recoverable later? |
| --- | --- | --- | --- |
| `universe` | Daily 19:00 UTC | Every game and DLC on the store, `last_modified`, `price_change_number`; delisted apps kept 30 days with `in_current_list = false` | No (delistings, new-app dates) |
| `tiers` | Daily, with `store_items` | Which player-count tier each game was in and why | Derived, saved for reproducibility |
| `regional_prices` | Daily 20:00 UTC | Tier A prices in DE, GB, BR, IN, JP, with formatted currency | Partly |
| `ccu` | Tier A at 04 and 16 UTC; B daily; C weekly in rotation; thesis study panel (`tier` = `study`, games in `study_apps.csv`) every 4 h at 00/04/08/12/16/20 UTC, plus 02/06/.../22 UTC during seasonal sales | Concurrent players; `http_status` and `result` separate "zero players" from "no data" | No |
| `price_changes` | Hourly | Games whose Steam price-change counter changed since the previous hour (`old_number`, `new_number`, `in_study`). Times every price change in the catalog to the hour | No |
| `study_prices` | Hourly, for study games whose counter changed; full snapshot of all study games daily at 09:00 UTC | US price, regular price and discount for the thesis study games; `reason` = `changed` or `snapshot` | No (IsThereAnyDeal has history, but not this check) |
| `charts_most_played` | 04 and 16 UTC | Steam's top 100 by daily peak players | No |
| `charts_concurrent` | 04 and 16 UTC | Steam's top games by current players | No |
| `topsellers` | Daily | Weekly top sellers (top 500), global plus 6 countries | Possibly (the API takes a past week) but not guaranteed |
| `featured` | Every 6 hours | Front-page and featured sections for an anonymous US visitor, with position and discount | No |
| `twitch_streams` | Every 6 hours | Top 10,000 live streams: streamer, game, viewers, language (no titles) | No |
| `twitch_games` | Every 6 hours | Viewers and streams per Twitch game, overall and by language | No |

## Weekly tables (Wednesday)

| Table | What | Notes |
| --- | --- | --- |
| `store_text` | Short and full store descriptions | Stored only when the text changes (`first_seen`, `sha1`) |
| `store_media` | Screenshot filenames, trailer names, supported languages, age rating | For image and multimodal models |
| `purchase_options` | Every package and bundle that includes each game, with prices | Bundle composition and pricing |
| `tag_names`, `store_categories` | ID-to-name lookups | Tags can be renamed |

Capsule (header) images for tier A games are saved to `data/assets/capsules/<appid>/` the
first time they are seen and whenever they change.

## Joining across sources

- Steam tables join on `appid`.
- Twitch uses its own `game_id`. Map to Steam later with IGDB (same Twitch credentials),
  whose external-games records list Steam app IDs. The mapping is recoverable later; the
  viewer counts are not, which is why only the counts are collected now.

## Known limitations

- **Outage, Nov 16-18, 2026:** no job ran from about 09:00 UTC on Nov 16 until the
  evening of Nov 18 (a syntax error in a local edit of `settings.py` stopped every run).
  All tables have a gap of about 2.7 days; the thesis study panel started after it.

- `featured` is what an anonymous US visitor sees; logged-in users get personalized pages.
- Player counts are single readings, not daily peaks (except `charts_most_played`).
- `review_pct_positive` is an integer, so for large games the positive count is approximate.
  Day-to-day changes in `review_count` are exact.
- Steam's undocumented endpoints can change. Check `python collect.py health` and the alerts.

## search_ranks (jobs `ranks` daily 17:00 UTC; `ranks_sale` 05/11/23 UTC during sales)

Steam's **Global Top Sellers** ranking from the store search (`filter=globaltopsellers`,
games only), the ordering SteamDB shows. Ranks are Steam's revenue ordering for recent
sales (game, DLC and in-game purchases); amounts are not published. One row per item per
list per snapshot.

| Column | Meaning |
| --- | --- |
| captured_at | When the ranking was recorded (UTC) |
| list | `global` (the whole ranking, about 7,300 games) or `tag` (top `RANKS_TAG_DEPTH` within one tag) |
| tagid | The tag for `tag` lists (names in `tag_names`); empty for `global` |
| rank | Position in that list, 1 = top seller |
| itemkey | Steam's item key: `App_<id>`, `Sub_<id>` (package) or `Bundle_<id>` |
| appid, packageid, bundleid | IDs parsed from the row (bundles: first app listed) |
| name | Title as shown |
| price_final_cents, discount_pct | Price and discount shown in the search row (US store), when present |
| page_start, total_count | Paging offset of the request, and how many items Steam said match |

Notes: games below Steam's sales threshold are not ranked at all (treat absence as "below
the cutoff"). Steam's search occasionally drops or repeats items between pages; repeats keep
their first rank. Ranks are relative: one item rising pushes others down.
