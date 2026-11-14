# Results produced outside the notebook

Outputs of pipeline scripts run on the full data, kept here so they can be read without
rerunning. Each folder names the script that produced it; rerun the script to reproduce.

| Folder | Script | What it holds |
| --- | --- | --- |
| `rank_validation/` | `pipeline/10_rank_validation.py` (full ensemble settings) | Discounting and Steam's weekly revenue charts, large and very large games; the revenue-proxy weighting test |
| `segments/` | `pipeline/11_segments.py` | Audience factors (K by held-out prediction), segments, tag mapping, robustness, rival lists, tag baselines. First run: big games mostly tag-placed until the 03f sample is added |

Game positions (`game_positions.parquet`, ~57 MB) are not committed; the script regenerates them.
