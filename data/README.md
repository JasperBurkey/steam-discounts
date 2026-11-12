# Data

The notebook reads two files here, both produced by `pipeline/04b_build_events.py`:

| File | Contents |
| --- | --- |
| `events.parquet` | One row per game x seasonal sale: discount status and depth, pre-event controls, outcomes, sample flags |
| `events_daily.parquet` | Daily review counts from 21 days before to 56 days after each sale (paged games) |

To update them, copy both files from the pipeline's `data` folder into this folder, then rerun the notebook. Raw reviews, prices and collector data are never committed (see the repository README).
