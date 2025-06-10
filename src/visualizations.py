from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt

def visualize_start_end(population_dist, grid_cells, snapshots):
    cmap = ListedColormap(["lightgrey", "steelblue", "indianred", "yellowgreen"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for ax, snap, title in ((ax1, snapshots[0], "Initial State"),(ax2, snapshots[-1], "Final State"),):
        # Each cell in the Utrecht grid will get its agent value, which in turn gets its own color
        grid_cells["value"] = grid_cells.index.map(lambda cid: snap.get(cid, -1))
        grid_cells.plot(
            column="value",
            categorical=True,
            cmap=cmap,
            linewidth=0.2,
            edgecolor="white",
            ax=ax,
            legend=False
        )
        ax.set_title(title)
        ax.set_axis_off()

    # Custom legend
    categories = [
        ("Empty", "lightgrey"),
        (f"Low Income ({population_dist[0]*100:.1f}%)", "steelblue"),
        (f"Medium Income ({population_dist[1]*100:.1f}%)", "indianred"),
        (f"High Income ({population_dist[2]*100:.1f}%)", "yellowgreen"),
    ]
    handles = [mpatch.Patch(color=c, label=l) for l, c in categories]
    ax1.legend(
        handles=handles,
        title="Category",
        loc="lower right",
        frameon=True,
        fontsize="small"
    )

    plt.tight_layout()
    plt.savefig("plots/end_start.png", dpi=300)
    plt.show()