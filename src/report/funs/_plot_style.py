# plot_style.py
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass, field
from typing import Dict
import matplotlib.font_manager as fm
import os

# Try to load Roboto; fallback silently if missing
target_font_path = (
    r"C:\\Users\\carol\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Roboto-Regular.ttf"
)

if os.path.exists(target_font_path):
    fm.fontManager.addfont(target_font_path)
    primary_font = "Roboto"
else:
    primary_font = "Arial"


def latex_safe(text: str) -> str:
    return text.replace("%", r"\%")


@dataclass
class PlotStyle:
    base_font_size: int = 9

    colors: Dict[str, str] = field(
        default_factory=lambda: {
            "high": "#e48646",
            "mid": "#f7f7f7",
            "low": "#5e548e",
            "low-m": "#726b95",
            "high-m": "#d38a5d",
            "mid-m": "#a7a7a7",
            "grid": "#E6E6E6",
            "text": "#4D4D4D",
            "spines": "#CCCCCC",
            # "grey": "#909090",
            "teal": "#5a9e9e",
            "grey": "#6b7c8e",
        }
    )

    @staticmethod
    def figsize_from_pt(width_pt=426.79135, fraction=1.0, ratio=0.618):
        inches_per_pt = 1 / 72.27
        width_in = width_pt * inches_per_pt * fraction
        height_in = width_in * ratio
        return (width_in, height_in)

    def apply(self, figsize=None):
        plt.style.use("seaborn-v0_8-white")

        mpl.rcParams.update({
            # Rendering Fixes (Prevents "Two Layers" / Ghosting)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            # Fonts
            "font.family": "sans-serif",
            "font.sans-serif": [primary_font, "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": self.base_font_size,
            "font.weight": "light",
            # Colors & Sizes
            "axes.labelcolor": self.colors["text"],
            "xtick.color": self.colors["text"],
            "ytick.color": self.colors["text"],
            "axes.titlecolor": self.colors["text"],
            "axes.labelweight": "light",
            "axes.titleweight": "normal",
            "axes.labelsize": self.base_font_size - 1,
            "figure.titlesize": self.base_font_size + 1,
            "axes.titlesize": self.base_font_size,
            "legend.fontsize": self.base_font_size - 1,
            "xtick.labelsize": self.base_font_size - 2,
            "ytick.labelsize": self.base_font_size - 2,
        })

        return self

    def style_axes(self, ax):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(self.colors["spines"])
            spine.set_linewidth(0.5)

        ax.grid(True, color=self.colors["grid"], linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(
            colors=self.colors["spines"],
            labelcolor=self.colors["text"],
            length=2,
            width=0.5,
        )
        return ax
