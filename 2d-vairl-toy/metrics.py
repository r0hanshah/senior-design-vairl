from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os

import matplotlib

# Use GUI backend if a display is available, otherwise save-only
if os.environ.get("DISPLAY"):
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass
class MetricsLogger:
    metrics: dict[str, list[float]] = field(default_factory=dict)

    def log(self, name: str, value: Any) -> None:
        self.metrics.setdefault(name, []).append(float(value))

    def save_csv(self, out_path: Path | str) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        keys = list(self.metrics.keys())
        if not keys:
            return

        n = len(self.metrics[keys[0]])
        for k in keys:
            if len(self.metrics[k]) != n:
                raise ValueError("Metric lengths mismatch")

        with out_path.open("w", encoding="utf-8") as f:
            f.write(",".join(["episode", *keys]) + "\n")
            for i in range(n):
                row = [str(i)] + [str(self.metrics[k][i]) for k in keys]
                f.write(",".join(row) + "\n")
    def save_plots(self, out_dir: Path | str, show: bool = False) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.metrics:
            return

        episodes = range(len(next(iter(self.metrics.values()))))

        def safe_filename(name: str) -> str:
            # keep it simple: replace path separators/spaces with underscores
            return (
                name.replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
                .replace(":", "_")
            )

        figures = []
        for name, values in self.metrics.items():
            fig = plt.figure()
            plt.plot(episodes, values)
            plt.xlabel("episode")
            plt.ylabel(name)
            plt.title(name)
            plt.tight_layout()

            fname = safe_filename(name) + ".png"
            plt.savefig(out_dir / fname)

            if show:
                figures.append(fig)
            else:
                plt.close(fig)

        if show:
            plt.show()
            for fig in figures:
                plt.close(fig)


    def save_all(self, out_dir: Path | str, show: bool = False) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.save_csv(out_dir / "training_metrics.csv")
        self.save_plots(out_dir / "plots", show=show)

