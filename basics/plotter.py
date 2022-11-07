import os
from typing import Sequence, Dict

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from basics.config import Config
from basics.enums import Color


class Plotter:
    def __init__(self, config: "Config"):
        self.fig_folder = config.fig
        self.setup_fig_params()

    def setup_fig_params(self):
        self.fig_size = (12, 7.417)
        self.fig_axe_area = (0.1, 0.1, 0.8, 0.8)
        self.fig_axe_area_with_legend = (0.1, 0.15, 0.8, 0.7)
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

    def line_figure(
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
            ax.plot(x, values, label=key)
        self.add_legend(figure, ax, len(values_dict))
        self.save_fig(figure, fig_name)

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

    def bar_figure(
            self,
            values_dict: "Dict[str, np.array]",
            fig_name: str,
            x_label: str = "X-axis",
            y_label: str = "Y-axis",
            x_lim=None,
            y_lim=None,
            x_tick_labels: list = None,
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

        if x_tick_labels is not None:
            ax.set_xticks(ticks=x, labels=x_tick_labels, rotation=90)
        self.add_legend(figure, ax, len(values_dict))
        self.save_fig(figure, fig_name)

    def directed_graph(
            self,
            graph: "Dict[str, str]",
            nodeweight,
            nodesize,
            figname: str,
            draw_pathway=[],
            draw_pathway_benefits=[]
    ):
        G = nx.DiGraph()

        node_labels = {}
        node_size = []
        node_color = []
        for node in graph:
            label = ''
            if node in draw_pathway:
                label = node
            G.add_node(node, layer=len(node) - sum(c1 != c2 for c1, c2 in zip('1' * len(node), node)))
            node_color.append(nodeweight[node])
            node_size.append(nodesize[node])
            node_labels[node] = label

        edge_labels = {}
        for node, edges in graph.items():
            for edge in edges:
                color = 'lightgrey'
                width = 0.4
                label = ''
                if node in draw_pathway:
                    idx = draw_pathway.index(node)
                    if draw_pathway[idx+1] == edge:
                        label = f'{draw_pathway_benefits[idx]} (€/a)'
                        color = 'red'
                        width = 1
                G.add_edge(node, edge, color=color, width=width)
                edge_labels[(node, edge)] = label
        edge_color = nx.get_edge_attributes(G, 'color').values()
        widths = list(nx.get_edge_attributes(G, 'width').values())

        pos = nx.multipartite_layout(G, subset_key="layer", )

        nx.draw(G, pos=pos, arrows=True, connectionstyle='arc3, rad = -0.1',
                node_color=node_color, node_size=[x * 2.5 + 7 for x in node_size], cmap=plt.cm.get_cmap('PiYG').reversed(), linewidths=0.5,
                edgecolors='grey',
                edge_color=edge_color, width=widths)
        nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=6, verticalalignment='top')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=4, verticalalignment='bottom', rotate=False)

        plt.savefig(os.path.join(self.fig_folder, figname + ".png"), dpi=300)
