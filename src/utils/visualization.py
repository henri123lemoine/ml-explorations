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


def plot1(
    data: np.ndarray,
    title: str,
    main_labels: list[str],
    ax_titles: list[str],
    algs_info: list[tuple[str, str, int]],
    range_y: list[tuple[float, float]] | None = None,
    x_axis_relative_range: tuple[float, float] = (-0.5, 0.5),
    size: tuple[int, int] | None = (10, 10),
    fill_std: list[int] | None = None,
    custom_x_labels: list[str] | None = None,
    plot_box: Bbox | None = Bbox([[0, 0], [10, 10]]),
    filename: str | None = None,
    show: bool = False,
):
    """
    Args:
        data (np.array): All that needs to be plotted. Shape: (n_figs, n_curves, n_steps, n_runs).
        title (str): The title of the plot.
        main_labels (list[str]): A list of strings, the first element being the label of the x axis, and the rest being the labels of the y axes.
        ax_titles (list[str]): A list of titles for each subplot.
        algs_info (list[tuple[str, str, int]]): A list of tuples containing the information of each curve. Each tuple contains the following: (name, color, marker type). The marker type is an integer denoting the type of marker to be used. 0: normal full line, 1: dashed line, 2: scatter plot.
        range_y (list[tuple[float, float]], optional): A list of tuples of length n_figs, each denoting the range of the y axis of the corresponding subplot. Defaults to None.
        x_axis_relative_range (tuple[float, float], optional): A tuple of length 2, denoting the relative range of the x axis. Defaults to [-0.5, 0.5] from first and last x.
        size (list[int], optional): The size of the plot. Defaults to (10, 10).
        fill_std (list[int], optional): A list of integers of length n_figs, each denoting whether and how to fill the standard deviation of the corresponding subplot. If None, no standard deviation will be filled. If 0, the standard deviation will be filled with a transparent color. If 1, the standard deviation will be a dashed line above and bellow the mean. If 2, just plot the standard deviation with a dashed line. Defaults to None.
        custom_x_labels (list[str], optional): A list of strings of length n_steps, each denoting the label of the corresponding x axis tick. If None, the x axis ticks will be the integers from 0 to n_steps. Defaults to None.
        plot_box (list[list[float]], optional): A list of lists of length 2, each denoting the coordinates of the bottom left and top right corners of the bounding box of the plot. Defaults to Bbox([[0, 0], [13, 20]]).
        filename (str, optional): The name of the file to save the plot to. If none, the plot will not be saved. Defaults to None.
        show (bool, optional): Whether to show the plot or not. Defaults to False.

    Returns:
        None
    """
    nb_figs, nb_curves, nb_steps, nb_runs = data.shape

    range_x = (0, nb_steps)

    fig, axes = plt.subplots(nrows=nb_figs, ncols=1, figsize=size)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.94)

    for i in range(nb_figs):
        if nb_figs == 1:
            ax = axes
        else:
            ax = axes[i]

        ax.set_title(ax_titles[i], fontsize=12, fontweight="bold", loc="left")

        for j in range(nb_curves):
            if algs_info[j][2] == 0:
                mean = np.mean(data[i, j, :, :], axis=1)
                std = np.std(data[i, j, :, :], axis=1) / np.sqrt(nb_runs)

                ax.plot(mean, label=algs_info[j][0], color=algs_info[j][1])

                if fill_std is not None:
                    if fill_std[i] is None:
                        pass
                    elif fill_std[i] == 0:
                        ax.fill_between(
                            np.arange(nb_steps),
                            mean - std,
                            mean + std,
                            alpha=0.2,
                            color=algs_info[j][1],
                        )
                    elif fill_std[i] == 1:
                        ax.plot(mean - std, color=algs_info[j][1], linestyle="--")
                        ax.plot(mean + std, color=algs_info[j][1], linestyle="--")
                    elif fill_std[i] == 2:
                        ax.plot(std, color=algs_info[j][1], linestyle="--")
                    else:
                        raise ValueError("Invalid fill_std value.")

            elif algs_info[j][2] == 1:
                ax.axhline(
                    y=np.mean(data[i, j, -1, :]),
                    label=algs_info[j][0],
                    color=algs_info[j][1],
                    linestyle="--",
                )

            elif algs_info[j][2] == 2:
                ax.scatter(
                    np.arange(nb_steps),
                    np.mean(data[i, j, :, :], axis=1),
                    s=30,
                    label=algs_info[j][0],
                    color=algs_info[j][1],
                )

            else:
                raise ValueError("Invalid marker type.")

        ax.legend(prop={"size": 10}, loc="right", bbox_to_anchor=(1, 0.5))  # (1.5, 0.5)

        if i == nb_figs - 1:
            ax.set_xlabel(main_labels[0], fontsize=14, labelpad=10)
        ax.set_ylabel(main_labels[i + 1], fontsize=14, labelpad=10)

        ax.set_xlim(range_x)
        if custom_x_labels is not None:
            ax.set_xticks(np.arange(len(custom_x_labels)))
            ax.set_xticklabels(custom_x_labels)
            ax.set_xlim(
                x_axis_relative_range[0],
                len(custom_x_labels) + x_axis_relative_range[1],
            )

        if range_y is not None and range_y[i] is not None:
            ax.set_ylim(range_y[i])

        ax.locator_params(nbins=10, axis="x")
        ax.locator_params(nbins=5, axis="y")
        ax.grid()

    if filename is not None:
        # plt.savefig(filename, bbox_inches=Bbox([[0, 0], [13, 20]]))
        plt.savefig(filename, bbox_inches=plot_box)
    if show:
        plt.show()  # bbox_inches=plot_box)


def plot(
    data: np.ndarray,
    title: str,
    main_labels: list[str],
    ax_titles: list[str],
    algs_info: list[tuple[str, str, str]],  # Updated to use string for marker type
    log_scale: tuple[bool, bool] = (False, False),
    range_x: tuple[float, float] | None = None,
    range_y: list[tuple[float, float] | None] | None = None,
    include_y0: bool = False,
    size: tuple[int, int] = (10, 10),  # Updated to use tuple
    fill_std: list[int | None] | None = None,  # Updated type hint
    plot_box: Bbox | None = Bbox([[0, 0], [10, 10]]),  # Updated type hint
    filename: str | None = None,
    show: bool = True,  # Default to True if filename is None
):
    """
    Args:
        data (np.array): All that needs to be plotted. Shape: (n_figs, n_curves, n_steps, n_runs).
        title (str): The title of the plot.
        main_labels (list[str]): A list of strings, the first element being the label of the x axis, and the rest being the labels of the y axes.
        ax_titles (list[str]): A list of titles for each subplot.
        algs_info (list[tuple[str, str, str]]): A list of tuples containing the information of each curve. Each tuple contains the following: (name, color, marker type). The marker type can be one of the following: 'line', 'dashed', 'scatter'.
        log_scale (tuple[bool, bool], optional): A tuple of length 2, denoting whether to use a log scale for the x and y axes respectively. Defaults to (False, False).
        range_x (tuple[float, float], optional): A list of tuples of length n_figs, each denoting the range of the x axis of the corresponding subplot. Defaults to None.
        range_y (list[tuple[float, float]], optional): A list of tuples of length n_figs, each denoting the range of the y axis of the corresponding subplot. Defaults to None.
        x_axis_relative_range (tuple[float, float], optional): A tuple of length 2, denoting the relative range of the x axis. Defaults to [-0.5, 0.5] from first and last x.
        size (list[int], optional): The size of the plot. Defaults to (10, 10).
        fill_std (list[int], optional): A list of integers of length n_figs, each denoting whether and how to fill the standard deviation of the corresponding subplot. If None, no standard deviation will be filled. If 0, the standard deviation will be filled with a transparent color. If 1, the standard deviation will be a dashed line above and bellow the mean. If 2, just plot the standard deviation with a dashed line. Defaults to None.
        plot_box (list[list[float]], optional): A list of lists of length 2, each denoting the coordinates of the bottom left and top right corners of the bounding box of the plot. Defaults to Bbox([[0, 0], [13, 20]]).
        filename (str, optional): The name of the file to save the plot to. If none, the plot will not be saved. Defaults to None.
        show (bool, optional): Whether to show the plot or not. Defaults to False.

    Returns:
        None
    """
    nb_figs, nb_curves, nb_steps, nb_runs = data.shape

    if range_x is None:
        range_x = (1, nb_steps)

    fig, axes = plt.subplots(nrows=nb_figs, ncols=1, figsize=size)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.94)

    if nb_figs == 1:
        axes = [axes]

    for i in range(nb_figs):
        ax = axes[i]
        ax.set_title(ax_titles[i], fontsize=12, fontweight="bold", loc="left")

        for j in range(nb_curves):
            xs = np.arange(*range_x)

            # Find the index of the first NaN or Inf value
            invalid_index = np.where(~np.isfinite(data[i, j, :, :]))[0]
            if len(invalid_index) > 0:
                first_invalid_index = invalid_index[0]
            else:
                first_invalid_index = nb_steps

            # Only use the data up to the first NaN or Inf value
            valid_data = data[i, j, :first_invalid_index, :]

            if algs_info[j][2] == "line":
                mean = np.mean(valid_data, axis=1)
                std = np.std(valid_data, axis=1) / np.sqrt(nb_runs)
                ax.plot(mean, label=algs_info[j][0], color=algs_info[j][1])

                if fill_std and fill_std[i] is not None:
                    if fill_std[i] == 0:
                        ax.fill_between(
                            xs, mean - std, mean + std, alpha=0.2, color=algs_info[j][1]
                        )
                    elif fill_std[i] == 1:
                        ax.plot(mean - std, color=algs_info[j][1], linestyle="--")
                        ax.plot(mean + std, color=algs_info[j][1], linestyle="--")
                    elif fill_std[i] == 2:
                        ax.plot(std, color=algs_info[j][1], linestyle="--")
                    else:
                        raise ValueError("Invalid fill_std value.")

            elif algs_info[j][2] == "dashed":
                ax.axhline(
                    y=np.mean(valid_data[-1, :]),
                    label=algs_info[j][0],
                    color=algs_info[j][1],
                    linestyle="--",
                )

            elif algs_info[j][2] == "scatter":
                ax.scatter(
                    xs,
                    np.mean(valid_data, axis=1),
                    s=30,
                    label=algs_info[j][0],
                    color=algs_info[j][1],
                )

            else:
                raise ValueError("Invalid marker type.")

        ax.legend(prop={"size": 10}, loc="upper left")

        ax.set_xlabel(main_labels[0], fontsize=14, labelpad=10)
        ax.set_ylabel(main_labels[i + 1], fontsize=14, labelpad=10)

        ax.set_xlim(range_x)
        if log_scale[0]:
            ax.set_xscale("log")
        if log_scale[1]:
            ax.set_yscale("log")

        if range_y and range_y[i] is not None:
            ax.set_ylim(range_y[i])
        else:
            top = np.nanmax(data[i, :, :, :])
            bottom = np.nanmin(data[i, :, :, :])
            if include_y0:
                bottom = min(bottom, 0)
                top *= 1.1
            else:
                mean = (top + bottom) / 2
                top = mean + (top - mean) * 1.05
                bottom = mean - (mean - bottom) * 1.05
            ax.set_ylim(bottom, top)

        ax.grid()

    if filename:
        plt.savefig(filename, bbox_inches=plot_box)
    if show:
        plt.show()


def plot2(
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
