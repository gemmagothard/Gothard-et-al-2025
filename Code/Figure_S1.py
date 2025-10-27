import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats as st
from plotting_functions import add_figure_letter
from matplotlib_venn import venn2
from matplotlib.patches import Patch
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
sns.set_context('paper')



def FigureS1(datapath,
             savepath):
    

    data = pd.read_csv(datapath)

    fig,  [[ax_venn, ax_legend],
            [ax_bar, ax_violin]] = plt.subplots(2, 2, gridspec_kw={'height_ratios': [0.6, 1], 'width_ratios': [1, 0.6]}, 
                                                figsize=(4, 6),
                                                constrained_layout=True)
    
    palette = { 'GFP+'      : '#8DC170',
                'GFP+TdT+'  : '#EF8706',
                'red'       :'#C70909'} 

    label_map = {'GFP+TdT+' :'basal IP-derived',
                'GFP+'      :'apical IP-derived'}

    order = ['GFP+','GFP+TdT+']

    ax_legend.axis('off')


    # Create custom legend handles in desired order
    legend_elements = [
        Patch(facecolor=palette[cell], edgecolor='none', label=label_map[cell])
        for cell in ['GFP+', 'GFP+TdT+']
    ]
    # Draw the legend in the custom subplot
    ax_legend.legend(
        handles=legend_elements,
        loc='center',
        frameon=False,
        fontsize='medium'
    )

    ## new bar plot with only green / yellow cells
    sns.barplot(data=data,
                y='counts_normalized',
                x='cell_id',
                hue='cell_id',
                estimator='mean',
                errorbar='se',
                palette=palette,
                alpha=0.5,
                ax=ax_bar,
                order=order,
                )
    sns.stripplot(data=data,
                y='counts_normalized',
                x='cell_id',
                hue='cell_id',
                order=order,
                ax=ax_bar,
                palette=palette,
                jitter=0,
                size=10,
                alpha=1,
                )
    ax_bar.set_ylabel('L2/3 cell type (%)')
    ax_bar.set_xlabel('')
    ax_bar.xaxis.set_visible(False)
    # Optional: connect paired points with lines
    # Assumes a 'pair_id' column exists indicating matched samples
    pair_col = 'experiment_ID'  # <-- change this to your actual column name
    # Map cell_ids to x-axis positions
    x_pos = {label: i for i, label in enumerate(order)}
    for pid, group in data.groupby('experiment_ID'):
        available = [label for label in order if label in group['cell_id'].values]
        if len(available) >= 2:
            group_sorted = group.set_index('cell_id').loc[available]
            x_vals = [x_pos[label] for label in available]
            y_vals = group_sorted['counts_normalized'].values
            ax_bar.plot(x_vals, y_vals, color='gray', alpha=0.3, linewidth=2)


    sns.violinplot(data=data,
                y='pial_dist',
                x='cell_id',
                hue='cell_id',
                ax=ax_violin,
                palette=palette,
                inner=None,  # 'box', 'quartile', 'point', 'stick', or None
                split=True,
                alpha=1
                )
    sns.pointplot(data=data,
                y='pial_dist',
                x='cell_id',
                ax=ax_violin,
                alpha=1,
                linestyle="none",
                palette=['black','black'],
                )
    ax_violin.invert_yaxis()
    ax_violin.set_ylabel('Pial distance (\u03bcm)')
    ax_violin.xaxis.set_visible(False)

    # Plot Venn diagram with green and yellow only
    size_green = np.round(np.nanmean(data[data.cell_id=='GFP+'].counts_normalized),0)
    size_yellow = np.round(np.nanmean(data[data.cell_id=='GFP+TdT+'].counts_normalized),0)
    size_red = 0

    venn2(
        subsets=(size_green, size_red, size_yellow),
        set_colors=(palette['GFP+'], palette['red']),
        alpha=0.8,
        ax=ax_venn
        )


    sns.despine(fig, ['top', 'right'])
    fig.tight_layout()
    fig.savefig(os.path.join(savepath, 'Fig_S1.svg'), bbox_inches='tight')

    plt.show()



FigureS1(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data/Fig_S1_data.csv',
         savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Figures'
         )