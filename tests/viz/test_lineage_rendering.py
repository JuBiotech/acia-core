"""Module for testing visualization module for lineage rendering"""

import random
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from acia.viz import plot_cell_lineage, plotly_cell_lineage

# ----------------- TEST CASE ----------------


def build_lineage_large(seed=42, n_cells=32, max_branches=2, max_time=8):
    """
    Generate a random cell lineage graph for demo/testing.
    """
    random.seed(seed)
    G = nx.DiGraph()
    cell_count = 0
    alive = []
    for i in range(2):
        node = f"root{i}"
        G.add_node(
            node, t=0, label=f"Root{i}_L", color="red" if i == 0 else "blue", val=i
        )
        alive.append(node)
        cell_count += 1
    for t in range(1, max_time + 1):
        new_alive = []
        for cell in alive:
            n_daughters = random.choice([1, max_branches])
            for j in range(n_daughters):
                if cell_count >= n_cells:
                    break
                node = f"{cell}_{t}_{j}"
                node_label = node if random.random() > 0.2 else None
                extra = {"custom": f"extra_{node}", "val": random.randint(1, 10)}
                if node_label is not None:
                    G.add_node(node, t=t, label=node_label, **extra)
                else:
                    G.add_node(node, t=t, **extra)
                G.add_edge(cell, node)
                new_alive.append(node)
                cell_count += 1
        if cell_count < n_cells and random.random() < 0.4:
            node = f"new_{t}_{cell_count}"
            G.add_node(node, t=t, label=node, custom="from_new", val=-1)
            new_alive.append(node)
            cell_count += 1
        alive = new_alive
        if cell_count >= n_cells:
            break
    while G.number_of_nodes() < n_cells:
        t += 1
        node = f"late_{t}_{cell_count}"
        G.add_node(node, t=t, label=node, custom="late", val=-2)
        alive.append(node)
        cell_count += 1
    return G


class TestLineageRendering(unittest.TestCase):
    """Testing scalebar and time overlay"""

    def test_matplotlib(self):
        # Build test lineage graph
        G = build_lineage_large(n_cells=32, max_branches=2, max_time=10)

        # --- Matplotlib plot (static) ---
        plt.figure(figsize=(14, 7))
        plot_cell_lineage(
            G,
            orientation="horizontal",
            show_label=True,
            label_name="label",
            node_marker="o",
            node_ms=10,
            line_color="forestgreen",
            line_lw=3,
            mark_births=True,
            birth_color="lime",
            birth_marker=">",
            birth_ms=18,
            mark_ends=True,
            end_color="magenta",
            end_marker="s",
            end_ms=18,
            interactive_tooltip=True,  # Enable tooltips if mplcursors installed
            ax=plt.gca(),
        )
        plt.title("Matplotlib: node features on hover (try with mplcursors)")
        plt.savefig("lineage.png")

        self.assertTrue(Path("lineage.png").exists())

    def test_plotly(self):
        # Build test lineage graph
        G = build_lineage_large(n_cells=32, max_branches=2, max_time=10)

        # --- Plotly interactive plot ---
        fig = plotly_cell_lineage(
            G,
            orientation="horizontal",
            show_label=True,
            label_name="label",
            node_marker="circle",
            node_ms=14,
            line_color="forestgreen",
            line_width=3,
            mark_births=True,
            birth_color="lime",
            birth_marker=">",
            birth_ms=20,
            mark_ends=True,
            end_color="magenta",
            end_marker="s",
            end_ms=20,
            figure_title="Plotly: node features as pseudo-table in hover",
        )
        fig.write_html("lineage.html")

        self.assertTrue(Path("lineage.html").exists())


if __name__ == "__main__":
    unittest.main()
