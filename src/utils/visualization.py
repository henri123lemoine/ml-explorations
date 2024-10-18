from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from tabulate import tabulate


def print_table(title, data, headers, sort_headers=None):
    assert len(headers) == len(data[0])

    # Sort the data based on the provided sort_headers list
    if sort_headers:
        for header in reversed(sort_headers):
            if header in headers:
                index = headers.index(header)
                data.sort(key=lambda x: x[index])

    # Using tabulate to print the table
    table = tabulate(data, headers=headers, tablefmt="grid")

    # Displaying the title and table
    print(title.center(len(title) + 6, "="))
    print(table)


def save_plot(x, ys, labels, title, x_label, y_label, filename):
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot(
    data: np.ndarray,
    title: str,
    main_labels: list[str],
    ax_titles: list[str],
    algs_info: list[tuple[str, str, str]],
    log_scale: tuple[bool, bool] = (False, False),
    range_x: tuple[float, float] | None = None,
    range_y: list[tuple[float, float] | None] | None = None,
    include_y0: bool = False,
    size: tuple[int, int] = (10, 10),
    fill_std: list[int | None] | None = None,
    custom_x_labels: list[str] | None = None,
    plot_box: Bbox | None = Bbox([[0, 0], [10, 10]]),
    filename: Path | None = None,
    show: bool = True,
):
    """
    A versatile plotting function that combines functionality from both original plot functions.

    Args:
        data (np.array): Data to be plotted. Shape: (n_figs, n_curves, n_steps, n_runs).
        title (str): The title of the plot.
        main_labels (list[str]): Labels for x-axis and y-axes.
        ax_titles (list[str]): Titles for each subplot.
        algs_info (list[tuple[str, str, str]]): Information about each curve (name, color, line style).
        log_scale (tuple[bool, bool]): Whether to use log scale for x and y axes.
        range_x (tuple[float, float]): Range for x-axis.
        range_y (list[tuple[float, float]]): Range for y-axes.
        include_y0 (bool): Whether to include y=0 in the plot.
        size (tuple[int, int]): Size of the plot.
        fill_std (list[int]): How to represent standard deviation for each subplot.
        custom_x_labels (list[str]): Custom labels for x-axis ticks.
        plot_box (Bbox): Bounding box for the plot.
        filename (str): Filename to save the plot.
        show (bool): Whether to display the plot.
    """
    nb_figs, nb_curves, nb_steps, nb_runs = data.shape

    if range_x is None:
        range_x = (0, nb_steps - 1)

    fig, axes = plt.subplots(nrows=nb_figs, ncols=1, figsize=size)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.94)

    if nb_figs == 1:
        axes = [axes]

    for i in range(nb_figs):
        ax = axes[i]
        ax.set_title(ax_titles[i], fontsize=12, fontweight="bold", loc="left")

        for j, (name, color, line_style) in enumerate(algs_info):
            mean = np.mean(data[i, j, :, :], axis=1)
            std = np.std(data[i, j, :, :], axis=1) / np.sqrt(nb_runs)

            if line_style == "line":
                ax.plot(mean, label=name, color=color)
            elif line_style == "dashed":
                ax.plot(mean, label=name, color=color, linestyle="--")
            elif line_style == "scatter":
                ax.scatter(range(len(mean)), mean, label=name, color=color, s=30)

            if fill_std and fill_std[i] is not None:
                if fill_std[i] == 0:
                    ax.fill_between(
                        range(len(mean)), mean - std, mean + std, alpha=0.2, color=color
                    )
                elif fill_std[i] == 1:
                    ax.plot(mean - std, color=color, linestyle="--")
                    ax.plot(mean + std, color=color, linestyle="--")
                elif fill_std[i] == 2:
                    ax.plot(std, color=color, linestyle="--")

        ax.legend(prop={"size": 10}, loc="upper left")

        if i == nb_figs - 1:
            ax.set_xlabel(main_labels[0], fontsize=14, labelpad=10)
        ax.set_ylabel(main_labels[i + 1], fontsize=14, labelpad=10)

        ax.set_xlim(range_x)
        if custom_x_labels:
            ax.set_xticks(range(len(custom_x_labels)))
            ax.set_xticklabels(custom_x_labels)

        if range_y and range_y[i] is not None:
            ax.set_ylim(range_y[i])
        elif include_y0:
            bottom, top = ax.get_ylim()
            ax.set_ylim(min(bottom, 0), top)

        if log_scale[0]:
            ax.set_xscale("log")
        if log_scale[1]:
            ax.set_yscale("log")

        ax.grid(True)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches=plot_box)
    if show:
        plt.show()
    else:
        plt.close()
