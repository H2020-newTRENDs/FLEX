import os
from typing import Sequence, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from flex.plotter import Plotter


class OperationPlotter(Plotter):

    def violin_figure(
            self,
            data: pd.DataFrame,
            fig_name: str,
            component: str,
            impacted_var: str,
            x_label: str,
            y_label: str,
    ):
        figure, ax = self.get_figure_template()
        sns.violinplot(
            x=component,
            y=impacted_var,
            hue="",
            data=data.rename(columns={"Option": ""}),
            ax=ax,
            split=True,
            inner="stick",
            palette="muted",
            cut=0
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.legend([], [], frameon=False)
        figure.legend(
            fontsize=self.legend_fontsize,
            bbox_to_anchor=self.top_bbox_to_anchor,
            bbox_transform=ax.transAxes,
            loc="lower left",
            ncol=2,
            borderaxespad=0,
            mode="expand",
            frameon=True,
        )
        self.save_fig(figure, fig_name)

    def dot_figure(
            self,
            values_dict: "Dict[str, Sequence]",  # dict with keys: Component State, values: Filtered Df
            fig_name: str,
            x_label: str = "X-axis",
            y_label: str = "Y-axis",
            x_lim=None,
            y_lim=None,
    ):
        figure, ax = self.get_figure_template(x_label, y_label, x_lim, y_lim)
        for key, values in values_dict.items():
            ax.scatter(list(values['Scenario']), list(values["CostChange"] / 100), label=key)
        self.add_legend(figure, ax, len(values_dict))
        self.save_fig(figure, fig_name)

    def dot_subplots_figure(
            self,
            values_dict: "Dict[str, Sequence]",  # dict with keys: component, values: dict with values per component
            components: "List[Tuple[str, int]]",
            fig_name: str,
            x_label: str = "X-axis",
            y_label: str = "Y-axis",
            x_lim=None,
            y_lim=None,
    ):
        rows, cols = int(np.ceil(len(values_dict) / 3)), 3
        figure, ax = plt.subplots(nrows=rows, ncols=cols, sharey=True, sharex=True)

        for row in range(rows):
            for col in range(cols):
                idx = 3 * row + col
                if idx >= len(components):  # then all components already have been plotted
                    break
                for key, values in values_dict[components[idx][0]].items():
                    ax[row, col].scatter(list(values['Scenario']), list(values["CostChange"] / 100), label=key, s=10)
                ax[row, col].legend()
                plt.xlim(x_lim)
                plt.ylim(y_lim)

        plt.setp(ax[-1, :], xlabel=x_label)  # figure.supxlabel(x_label) # for super labels
        plt.setp(ax[:, 0], ylabel=y_label)  # figure.supylabel(y_label) # for super labels
        figure.set_size_inches(18.5, 8)
        figure.suptitle(fig_name, fontsize=20)

        plt.savefig(os.path.join(self.fig_folder, fig_name + ".png"), dpi=300)

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
                    if draw_pathway[idx + 1] == edge:
                        label = f'{draw_pathway_benefits[idx]} (€/a)'
                        color = 'red'
                        width = 1
                G.add_edge(node, edge, color=color, width=width)
                edge_labels[(node, edge)] = label
        edge_color = nx.get_edge_attributes(G, 'color').values()
        widths = list(nx.get_edge_attributes(G, 'width').values())

        pos = nx.multipartite_layout(G, subset_key="layer", )

        nx.draw(G, pos=pos, arrows=True, connectionstyle='arc3, rad = -0.1',
                node_color=node_color, node_size=[x * 2.5 + 7 for x in node_size],
                cmap=plt.cm.get_cmap('PiYG').reversed(), linewidths=0.5,
                edgecolors='grey',
                edge_color=edge_color, width=widths)
        nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=6, verticalalignment='top')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=4, verticalalignment='bottom',
                                     rotate=False)

        plt.savefig(os.path.join(self.fig_folder, figname + ".png"), dpi=300)
