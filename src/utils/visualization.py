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
    data: np.array,
    title: str,
    main_labels: list[str],
    ax_titles: list[str],
    algs_info: list[tuple[str, str, int]],
    range_y: list[tuple[float, float]] = None,
    x_axis_relative_range: tuple[float, float] = [-0.5, 0.5],
    size: list[int] = (10, 10),
    fill_std: list[int] = None,
    custom_x_labels: list[str] = None,
    plot_box: list[list[float]] = Bbox([[0, 0], [10, 10]]),
    filename: str = None,
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
