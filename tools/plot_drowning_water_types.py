"""生成“青少年溺水事发水域类型占比统计”图表。

用法：
    python tools/plot_drowning_water_types.py

输出：
    科研/毕业设计/开题报告/assets/drowning_water_types.png
    科研/毕业设计/开题报告/assets/drowning_water_types.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch


DATA = {
    "河道": 61.16,
    "水库": 11.5,
    "水潭": 6.2,
    "海域": 4.3,
    "湖畔": 5.1,
    "泳池": 7.0,
    "其他": 4.7,
}

COLORS = {
    "河道": "#4FB6C2",
    "水库": "#8ED0D7",
    "水潭": "#BFE3E7",
    "海域": "#57AEB6",
    "湖畔": "#2E6F73",
    "泳池": "#6F8F99",
    "其他": "#D7E4E8",
}

TITLE = "青少年溺水事发水域类型占比统计"
SOURCE = "数据来源：人民网舆情数据中心根据网络公开资料整理（部分比例为视觉估计）"


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def darken_color(color: str, factor: float = 0.72) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def main() -> None:
    configure_matplotlib()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "科研" / "毕业设计" / "开题报告" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(DATA.keys())
    values = list(DATA.values())
    colors = [COLORS[label] for label in labels]
    depth_colors = [darken_color(color, 0.68) for color in colors]
    explode = [0.03 if label == "河道" else 0 for label in labels]

    fig = plt.figure(figsize=(12, 7), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.02)

    ax_pie = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])

    depth = 0.16
    layers = 14
    for i in range(layers):
        offset = -depth + (i / (layers - 1)) * depth
        ax_pie.pie(
            values,
            colors=depth_colors,
            startangle=120,
            counterclock=False,
            explode=explode,
            radius=1.0,
            center=(0, offset),
            wedgeprops={"edgecolor": "none", "linewidth": 0},
        )

    wedges, _ = ax_pie.pie(
        values,
        colors=colors,
        startangle=120,
        counterclock=False,
        explode=explode,
        radius=1.0,
        center=(0, 0),
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        shadow=True,
    )

    ax_pie.text(
        0,
        0.08,
        "61.16%",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="#1F4F56",
    )
    ax_pie.text(
        0,
        -0.12,
        "溺水发生在河道",
        ha="center",
        va="center",
        fontsize=14,
        color="#355F66",
    )
    ax_pie.set(aspect="equal")
    ax_pie.set_xlim(-1.35, 1.25)
    ax_pie.set_ylim(-1.28, 1.08)
    ax_pie.set_title(TITLE, fontsize=18, fontweight="bold", pad=18)

    legend_handles = [Patch(facecolor=COLORS[label], edgecolor="none", label=label) for label in labels]
    ax_legend.legend(
        handles=legend_handles,
        loc="center left",
        frameon=False,
        fontsize=13,
        labelspacing=1.2,
        handlelength=1.4,
        handletextpad=0.8,
    )
    ax_legend.axis("off")

    fig.text(0.5, 0.06, SOURCE, ha="center", fontsize=10, color="#5B6B73")
    fig.text(0.5, 0.025, "制图：GitHub Copilot", ha="center", fontsize=9, color="#8A989E")

    png_path = output_dir / "drowning_water_types.png"
    pdf_path = output_dir / "drowning_water_types.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"PNG 已生成：{png_path}")
    print(f"PDF 已生成：{pdf_path}")


if __name__ == "__main__":
    main()
