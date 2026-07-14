# -*- coding: utf-8 -*-
"""ER diagram generator using matplotlib — visualizes SQLite table schemas."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

from config.settings import DB_PATH
from modules.sqlite_manager import TrainingDatabase
from utils.logger import get_logger

logger = get_logger(__name__)


def build_er_figure(db_path: str | None = None):
    """Generate an ER diagram as a matplotlib Figure.

    Draws two table boxes (persons, training_records) with columns
    and a relationship line between the foreign key.

    Returns:
        matplotlib.figure.Figure ready for st.pyplot().
    """
    db = TrainingDatabase(db_path if db_path else DB_PATH)
    conn = db.connect()
    persons_cols = conn.execute("PRAGMA table_info(persons)").fetchall()
    training_cols = conn.execute("PRAGMA table_info(training_records)").fetchall()

    # Get row counts
    p_rows = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    t_rows = conn.execute("SELECT COUNT(*) FROM training_records").fetchone()[0]
    db.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Database ER Diagram", fontsize=16, fontweight="bold", pad=20)

    def _draw_table(ax, x, y, width, name: str, cols, row_count: int, pk_col: str):
        """Draw a table box with columns."""
        col_h = 0.22
        header_h = 0.35
        pad = 0.08
        height = header_h + len(cols) * col_h + pad * 2

        # Table box
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.05",
            facecolor="#E8ECF3",
            edgecolor="#34495E",
            linewidth=1.5,
        )
        ax.add_patch(box)

        # Header
        ax.text(
            x + width / 2, y + height - header_h / 2,
            f"{name}", fontsize=12, fontweight="bold", color="#1A5276",
            ha="center", va="center",
        )
        ax.text(
            x + width / 2, y + height - header_h + 0.15,
            f"({row_count} rows)", fontsize=8, color="#7F8C8D",
            ha="center", va="center",
        )

        # Columns
        for i, c in enumerate(cols):
            cy = y + height - header_h - pad - (i + 0.5) * col_h
            is_pk = c["name"] == pk_col
            is_fk = c["name"] == "employee_code" and pk_col != "employee_code"
            prefix = "🔑" if is_pk else ("🔗" if is_fk else "  ")
            color = "#D35400" if is_pk else ("#E74C3C" if is_fk else "#2C3E50")
            fw = "bold" if (is_pk or is_fk) else "normal"
            ax.text(
                x + 0.1, cy,
                f"{prefix} {c['name']} ({c['type']})",
                fontsize=8.5, color=color, fontweight=fw,
                ha="left", va="center",
            )

        return height, None

    # Draw tables
    p_h, _ = _draw_table(ax, 0.5, 1.5, 4.5, "persons", persons_cols, p_rows, "employee_code")
    t_h, _ = _draw_table(ax, 6.5, 1.0, 5.0, "training_records", training_cols, t_rows, "id")

    # Draw FK relationship line
    # Find FK row index in training_records
    fk_idx = next((i for i, c in enumerate(training_cols) if c["name"] == "employee_code"), 0)
    col_h = 0.22
    header_h = 0.35
    pad = 0.08
    y0 = 1.0 + t_h - header_h - pad - (fk_idx + 0.5) * col_h
    y1 = y0  # Same row level

    # Arrow from persons (right edge = 5.0) to training (left edge = 6.5)
    ax.annotate(
        "", xy=(6.5, y0), xytext=(5.0, y0),
        arrowprops=dict(
            arrowstyle="->", color="#27AE60", lw=2.5,
            connectionstyle="arc3,rad=0.1",
        ),
    )
    ax.text(
        5.75, y0 + 0.18, "references", fontsize=8, color="#27AE60",
        ha="center", va="bottom",
    )

    # Legend
    ax.text(0.5, 0.3, "🔑 PK  🔗 FK  ⟶ reference", fontsize=9, color="#7F8C8D")

    fig.tight_layout()
    return fig
