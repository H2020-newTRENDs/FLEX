import os
from typing import Sequence, Dict
import matplotlib.pyplot as plt
import numpy as np

from basics.config import Config
from basics.enums import Color


class Plotter:
    def __init__(self, config: "Config"):
        self.fig_folder = config.fig
        self.setup_fig_params()

    def setup_fig_params(self):
        self.fig_size = (12, 7.417)
        self.fig_axe_area = (0.1, 0.1, 0.8, 0.8)
        self.fig_axe_area_with_legend = (0.1, 0.1, 0.8, 0.7)
        self.fig_format = "PNG"
        self.fig_dpi = 200
        self.line_width = 1.5
        self.line_style = "-"
        self.alpha = 0.2
        self.fontsize = 15
        self.legend_fontsize = 12
        self.labelpad = 6
        self.top_bbox_to_anchor = (0, 1.02, 1, 0.1)
        self.grid = True

    @staticmethod
    def legend_ncol(legend_num: int) -> int:
        ncol = 2
        if legend_num in [2, 4]:
            pass
        elif legend_num in [3, 5, 6, 9]:
            ncol = 3
        elif legend_num in [7, 8] or legend_num >= 10:
            ncol = 4
        return ncol

    def save_fig(self, fig, fig_name):
        fig.savefig(
            os.path.join(self.fig_folder, fig_name + ".png"),
            dpi=self.fig_dpi,
            format=self.fig_format,
        )
        plt.close(fig)

    def get_figure_template(
        self, x_label: str = "X-axis", y_label: str = "Y-axis", x_lim=None, y_lim=None
    ):

        figure = plt.figure(figsize=self.fig_size, dpi=self.fig_dpi, frameon=False)
        ax = figure.add_axes(self.fig_axe_area)

        ax.grid(self.grid)
        ax.set_xlabel(x_label, fontsize=self.fontsize, labelpad=self.labelpad)
        ax.set_ylabel(y_label, fontsize=self.fontsize, labelpad=self.labelpad)

        if x_lim is not None:
            ax.set_ylim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(self.fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(self.fontsize)

        return figure, ax

    def add_legend(self, figure, ax, num_items: int):
        ax.set_position(self.fig_axe_area_with_legend)
        figure.legend(
            fontsize=self.legend_fontsize,
            bbox_to_anchor=self.top_bbox_to_anchor,
            bbox_transform=ax.transAxes,
            loc="lower left",
            ncol=self.legend_ncol(num_items),
            borderaxespad=0,
            mode="expand",
            frameon=True,
        )

    def step_figure(
        self,
        values_dict: "Dict[str, Sequence]",
        fig_name: str,
        x_label: str = "X-axis",
        y_label: str = "Y-axis",
        x_lim=None,
        y_lim=None,
    ):
        figure, ax = self.get_figure_template(x_label, y_label, x_lim, y_lim)
        for key, values in values_dict.items():
            x = [i + 1 for i in range(0, len(values))]
            ax.step(x, values, where="mid", label=key)
        self.add_legend(figure, ax, len(values_dict))
        self.save_fig(figure, fig_name)

    def bar_plot(
        self,
        values_dict: "Dict[str, np.array]",
        fig_name: str,
        x_label: str = "X-axis",
        y_label: str = "Y-axis",
        x_lim=None,
        y_lim=None,
    ):
        figure, ax = self.get_figure_template(x_label, y_label, x_lim, y_lim)
        x = [i + 1 for i in range(0, len(list(values_dict.values())[0]))]
        bottom_positive = np.zeros(
            len(x),
        )
        bottom_negative = np.zeros(
            len(x),
        )
        for key, values in values_dict.items():
            if values.mean() > 0:
                ax.bar(
                    x,
                    values,
                    bottom=bottom_positive,
                    label=key,
                    color=[Color.__dict__[key].value for i in range(0, len(x))],
                )
                bottom_positive += values
            else:
                ax.bar(
                    x,
                    values,
                    bottom=bottom_negative,
                    label=key,
                    color=[Color.__dict__[key].value for i in range(0, len(x))],
                )
                bottom_negative += values
        self.add_legend(figure, ax, len(values_dict))
        self.save_fig(figure, fig_name)