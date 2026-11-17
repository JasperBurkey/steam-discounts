"""Step 14: save the notebook's tables at full thesis settings to results/main/.

The notebook runs on GitHub with quick settings and keeps no outputs, so the thesis-setting
numbers are saved here for citing and for the write-up.

  python 14_notebook_tables.py      (about an hour at full settings)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auxiliary import data, estimates  # noqa: E402

OUT = ROOT / "results" / "main"


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "settings.txt").write_text(estimates.settings_note() + "\n")
    events = data.load_events()
    main_df = data.analysis_sample(events)
    long_df = data.long_sample(main_df)

    def save(name, table):
        table.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name} saved  [{(time.time() - t0) / 60:.0f} min]", flush=True)

    main_table, e_main = estimates.main_estimates(main_df, long_df)
    save("main_estimates", main_table)
    save("robustness", estimates.robustness(main_df, long_df, e_main))
    save("by_size", estimates.by_size(long_df))
    save("depth", estimates.depth(main_df))
    blp, gates, clan = estimates.heterogeneity(main_df)
    save("het_blp", blp)
    save("het_gates", gates)
    save("het_clan", clan)
    print(f"Done in {(time.time() - t0) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
