"""Independent entry point for the ten-seed temporal experiment."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys

from config import PROJECT_DIR
from experiments.temporal import refresh_report, refresh_statistics, run_temporal, verify_saved
from plotting.plot_temporal import plot_temporal


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        return len(text)
    def flush(self):
        self.console.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "time_slots_temporal_performance")
    parser.add_argument("--seeds", type=int, nargs=10, default=list(range(10)),
                        help="Ten distinct seeds, default 0 1 2 3 4 5 6 7 8 9")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plot-only", action="store_true", help="Recompute statistics and plot saved raw records")
    mode.add_argument("--verify-only", action="store_true", help="Audit saved records and pointwise intervals")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.verify_only:
        print(verify_saved(args.output_dir))
        return
    if args.plot_only:
        refresh_statistics(args.output_dir)
        verify_saved(args.output_dir)
        refresh_report(args.output_dir)
    else:
        (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "logs" / "run.log").open("a", encoding="utf-8", buffering=1) as handle:
            with redirect_stdout(_Tee(sys.stdout, handle)):
                run_temporal(args.output_dir, args.seeds)
    for path in plot_temporal(args.output_dir / "figure_data" / "fig8_performance_over_time.csv",
                              args.output_dir / "figures"):
        print(path)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
