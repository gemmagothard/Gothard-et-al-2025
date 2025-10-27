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
sns.set_context('talk')


def FigureS2(datapath,
             savepath):

    data = pd.read_csv(datapath)

    palette_bar = { 'Ta1+ Glast-'   : '#EAF3E4',
                    'Ta1+ Glast+'   : '#FDF1DB',
                    'Ta1- Glast+'   : '#F4D1D1'}

    palette_points = { 'Ta1+ Glast-'   : '#8DC170',
                        'Ta1+ Glast+'   : '#EC8F0E',
                        'Ta1- Glast+'   : '#C40D0D'}  
    
    order = ['Ta1+ Glast-', 'Ta1+ Glast+', 'Ta1- Glast+']
    order_pial_dist = ['Ta1+ Glast-','Ta1- Glast+']

    n_plots = 3
    fig, (cell_type_ax, venn_ax, pial_dist_ax) = plt.subplots(1,n_plots,figsize=(4*n_plots,4),width_ratios=[1,1.3,1])


    sns.barplot(data=data,
                x='cell_id',
                y='cell_counts',
                ax=cell_type_ax,
                order=order,
                hue='cell_id',
                palette=palette_bar)
    sns.stripplot(data=data,
            x='cell_id',
            y='cell_counts',
            ax=cell_type_ax,
            order=order,
            hue='cell_id',
            palette=palette_points)
    cell_type_ax.set_ylabel('L2/3 cell type (%)')
    cell_type_ax.set_xlabel('')
    # Map cell_ids to x-axis positions
    x_pos = {label: i for i, label in enumerate(order)}
    for pid, group in data.groupby('Brain'):
        available = [label for label in order if label in group['cell_id'].values]
        if len(available) >= 2:
            group_sorted = group.set_index('cell_id').loc[available]
            x_vals = [x_pos[label] for label in available]
            y_vals = group_sorted['cell_counts'].values
            cell_type_ax.plot(x_vals, y_vals, color='gray', alpha=0.3, linewidth=2)



    data_pial_dist = data[data.cell_id != 'Ta1+ Glast+']
    sns.barplot(data=data_pial_dist,
                x='cell_id',
                y='pial_dist',
                ax=pial_dist_ax,
                order=order_pial_dist,
                hue='cell_id',
                palette=palette_bar)
    sns.stripplot(data=data_pial_dist,
            x='cell_id',
            y='pial_dist',
            ax=pial_dist_ax,
            order=order_pial_dist,
            hue='cell_id',
            palette=palette_points)
    pial_dist_ax.set_ylabel('Pial distance (\u03BCm)')
    
    # Map cell_ids to x-axis positions
    x_pos = {label: i for i, label in enumerate(order_pial_dist)}
    for pid, group in data_pial_dist.groupby('Brain'):
        available = [label for label in order_pial_dist if label in group['cell_id'].values]
        if len(available) >= 2:
            group_sorted = group.set_index('cell_id').loc[available]
            x_vals = [x_pos[label] for label in available]
            y_vals = group_sorted['pial_dist'].values
            pial_dist_ax.plot(x_vals, y_vals, color='gray', alpha=0.3, linewidth=2)



    # Plot Venn diagram with green and yellow only
    size_green = np.round(np.average(data[data.cell_id=='Ta1+ Glast-'].cell_counts),0)
    size_yellow = np.round(np.average(data[data.cell_id=='Ta1+ Glast+'].cell_counts),0)
    size_red = np.round(np.average(data[data.cell_id=='Ta1- Glast+'].cell_counts),0)

    venn = venn2(
        subsets=(size_green, size_red, size_yellow),
        set_colors=(palette_points['Ta1+ Glast-'], palette_points['Ta1- Glast+']),
        alpha=0.8,
        ax=venn_ax
    )

    sns.despine(fig,right=True,top=True)

    plt.show()
    



FigureS2(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data/Fig_S2_data.csv',
         savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Figures'
         )