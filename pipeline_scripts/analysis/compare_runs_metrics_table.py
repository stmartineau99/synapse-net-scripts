import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATASET = "deepict"
TAG = "easymode"
OUT_FILE = f"metrics_{TAG}.png"

RUNS = {
    "Easymode t=0.5": "easymode-actin.csv",
    "Easymode t=0.07": "easymode-actin_t0.07.csv",
    "SL Full Label": "actin-deepict-run28.csv",
}

METRICS = ["precision", "recall", "dice"]


def save_metrics_png(summary_df, out_path):
    table = summary_df.copy()
    table[METRICS] = table[METRICS].round(3)
    col_labels = ["Model"] + [m.capitalize() for m in METRICS]
    cell_text = table[["model"] + METRICS].values.tolist()
    n_rows, n_cols = len(cell_text), len(col_labels)

    fig, ax = plt.subplots(figsize=(n_cols * 1.6, (n_rows + 1) * 0.5))
    ax.axis("off")
    t = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 1.4)

    for col in range(n_cols):
        cell = t[0, col]
        cell.set_facecolor("white")
        cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("white")
        cell.visible_edges = "TB"

    for row in range(1, n_rows + 1):
        for col in range(n_cols):
            cell = t[row, col]
            cell.set_facecolor("white")
            cell.set_edgecolor("white")
            cell.visible_edges = "open"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parent_path = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/predictions")
    csv_path = parent_path / DATASET / "csv"
    png_path = parent_path / DATASET / "png" / OUT_FILE
    png_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, fname in RUNS.items():
        df = pd.read_csv(csv_path / fname, dtype={"tomogram": str})
        means = df[METRICS].mean().to_dict()
        rows.append({"model": label, **means})

    summary = pd.DataFrame(rows)
    save_metrics_png(summary, png_path)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
