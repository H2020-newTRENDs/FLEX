from basics.plotter import Plotter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
