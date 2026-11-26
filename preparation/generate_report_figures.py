"""Regenerate report figures from official CSVs and recorded experiment results."""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]


def label_frequency():
    with (ROOT / "data/raw/train_features.csv").open(newline="", encoding="utf-8-sig") as stream:
        treated = {
            row["sig_id"]
            for row in csv.DictReader(stream)
            if row["cp_type"] == "trt_cp"
        }
    with (ROOT / "data/raw/train_targets_scored.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        labels = reader.fieldnames[1:]
        counts = Counter({label: 0 for label in labels})
        for row in reader:
            if row["sig_id"] in treated:
                counts.update({label: int(row[label]) for label in labels})

    edges = [1, 2, 5, 10, 20, 50, 100, 250, float("inf")]
    frequencies = [
        sum(lower <= count < upper for count in counts.values())
        for lower, upper in zip(edges, edges[1:])
    ]
    intervals = ["1", "2-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250+"]
    figure, axis = plt.subplots(figsize=(3.5, 2.1), layout="constrained")
    bars = axis.bar(range(len(frequencies)), frequencies, color="#385c7e", width=0.72)
    axis.bar_label(bars, padding=2, fontsize=7)
    axis.set_xticks(range(len(intervals)), intervals, rotation=35, ha="right")
    axis.set_xlabel("Positive samples per label")
    axis.set_ylabel("Number of labels")
    axis.set_ylim(0, max(frequencies) * 1.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.18)
    output = ROOT / "figures/label_frequency.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    print(f"Saved {output}")
    print(f"Labels per interval: {dict(zip(intervals, frequencies))}")


def source_removal():
    scores = json.loads((ROOT / "evidence/report_combinations.json").read_text())["scores"]
    sources = ["response_cnn5", "response_aux5", "response_profile5", "library"]
    full = scores["response_cnn5+response_profile5+response_aux5+library"]
    ordered_tags = ["response_cnn5", "response_profile5", "response_aux5", "library"]
    increases = [scores["+".join(tag for tag in ordered_tags if tag != source)] - full
                 for source in sources]
    names = ["Direct CNN", "Auxiliary CNN", "ProfileMLP", "Drug library"]
    figure, axis = plt.subplots(figsize=(3.5, 2.25), layout="constrained")
    bars = axis.barh(names, [value * 1e5 for value in increases], color="#385c7e", height=0.6)
    axis.bar_label(bars, labels=[f"{value:.8f}" for value in increases], padding=3, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlim(0, max(increases) * 1e5 * 1.43)
    axis.set_xlabel(r"Increase in OOF log loss ($\times 10^{-5}$)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="x", alpha=0.18)
    output = ROOT / "figures/source_removal.pdf"
    figure.savefig(output)
    plt.close(figure)
    print(f"Saved {output}")


def fold_comparison():
    record = json.loads((ROOT / "runs/response_study/selected_cv.json").read_text())
    folds = record["folds"]
    figure, axis = plt.subplots(figsize=(3.5, 2.25), layout="constrained")
    axis.plot([row["fold"] for row in folds], [row["course5"] for row in folds],
              marker="o", markersize=4, linestyle="--", color="#7d8288", label="Earlier Course5 system")
    axis.plot([row["fold"] for row in folds], [row["outer"] for row in folds],
              marker="o", markersize=4, color="#385c7e", label="Final system")
    axis.set_xticks(range(5))
    axis.set_xlabel("Held-out fold")
    axis.set_ylabel("Treated-row log loss")
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.set_ylim(0.01635, 0.01765)
    axis.legend(frameon=False, fontsize=7, loc="upper center")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.18)
    output = ROOT / "figures/fold_comparison.pdf"
    figure.savefig(output)
    plt.close(figure)
    print(f"Saved {output}")


def model_overview():
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = {
        "input": (0.005, 0.38, 0.16, 0.23, "Cell-response input\n872 measurements\n+ time, dose, type"),
        "cnn_features": (0.205, 0.65, 0.185, 0.24, "CNN representation\nSample normalization\nQuantiles + PCA\n937 features"),
        "mlp_features": (0.205, 0.31, 0.185, 0.24, "MLP representation\nQuantiles, factors,\nstatistics, conditions\n1,032 features"),
        "library_features": (0.205, 0.015, 0.185, 0.20, "Library query\nNormalize + asinh\n128D projection"),
        "direct": (0.44, 0.77, 0.19, 0.18, "Direct CNN\n206-label training"),
        "aux": (0.44, 0.535, 0.19, 0.18, "Auxiliary CNN\nWarm-up + weighting"),
        "mlp": (0.44, 0.30, 0.19, 0.18, "ProfileMLP\nDirect + profile heads"),
        "library": (0.44, 0.015, 0.19, 0.20, "Drug-library vote\nSimilarity to training\ndrug prototypes"),
        "fusion": (0.69, 0.47, 0.125, 0.30, "Average\n8 seeds per\nconfiguration\n\nLogit fusion"),
        "output": (0.855, 0.35, 0.14, 0.29, "Similarity mix\nControl mask\n\n206 output\nprobabilities"),
    }
    for key, (x, y, width, height, label) in boxes.items():
        fill = "#eaf0f5" if key in ("direct", "aux", "mlp", "library") else "white"
        axis.add_patch(Rectangle((x, y), width, height, facecolor=fill,
                                 edgecolor="#385c7e", linewidth=0.8))
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=7.5)

    def arrow(start, end):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                       mutation_scale=8, linewidth=0.8, color="#385c7e"))

    arrow((0.165, 0.52), (0.205, 0.77))
    arrow((0.165, 0.48), (0.205, 0.43))
    arrow((0.165, 0.42), (0.205, 0.115))
    arrow((0.39, 0.81), (0.44, 0.86))
    arrow((0.39, 0.72), (0.44, 0.625))
    arrow((0.39, 0.43), (0.44, 0.39))
    arrow((0.39, 0.115), (0.44, 0.115))
    arrow((0.63, 0.86), (0.69, 0.71))
    arrow((0.63, 0.625), (0.69, 0.62))
    arrow((0.63, 0.39), (0.69, 0.53))
    arrow((0.815, 0.62), (0.855, 0.57))
    axis.plot([0.63, 0.925, 0.925], [0.115, 0.115, 0.27], color="#385c7e", linewidth=0.8)
    arrow((0.925, 0.27), (0.925, 0.35))
    axis.text(0.79, 0.155, "Votes + maximum similarity", ha="center", fontsize=7)
    output = ROOT / "figures/model_overview.pdf"
    figure.savefig(output)
    plt.close(figure)
    print(f"Saved {output}")


def main():
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    label_frequency()
    model_overview()
    source_removal()
    fold_comparison()


if __name__ == "__main__":
    main()
