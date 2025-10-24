

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotting_functions
import os
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'

def Figure2(
        datapath,
        savepath,
        ):
    
    palette_bars = {'unlabeled': '#E7E7E7',
                    'IP-OP': '#EAF3E4'}
        
    palette_points = {'unlabeled': '#808080',
                    'IP-OP': '#8DC170'}
    
    red_green_palette = ['#C70909','#8DC170']
    
    
    data_to_plot = pd.read_csv(datapath)

    n_plots = 4
    fig,ax = plt.subplots(1,n_plots,figsize=(4*n_plots,4))
    sns.despine(fig,right=True,top=True)
    order = ['unlabeled', 'IP-OP']

    y = 'percent_shared'
    this_ax = ax[0]
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('Shared inhibitory input (%) ')
    sns.barplot(data=data_to_plot,
                x='Pair_type',
                y=y,
                hue='Pair_type',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='Pair_type',
                hue='Pair_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)

    y = 'difference_ratio'
    this_ax = ax[1]
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('IPSC amplitude difference (%) ')
    sns.barplot(data=data_to_plot,
                x='Pair_type',
                y=y,
                hue='Pair_type',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='Pair_type',
                hue='Pair_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)
    


    y = 'Intersomatic_distance'
    this_ax = ax[2]
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('Inter-somatic distance (\u03BCm) ')
    sns.barplot(data=data_to_plot,
                x='Pair_type',
                y=y,
                hue='Pair_type',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='Pair_type',
                hue='Pair_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)
    

    OP_derived = np.concat([data_to_plot[data_to_plot.Neuron1_ID=='OP-derived'].Neuron1_pial_dist,
                    data_to_plot[data_to_plot.Neuron2_ID=='OP-derived'].Neuron2_pial_dist])
    IP_derived = np.concat([data_to_plot[data_to_plot.Neuron2_ID=='IP-derived'].Neuron2_pial_dist,
                data_to_plot[data_to_plot.Neuron1_ID=='IP-derived'].Neuron1_pial_dist])
    data = [OP_derived, IP_derived]
    plotting_functions.scatter_bar_plot(ax[3], data, red_green_palette, ['OP','IP'], True ,'Pial distance (\u03BCm)', False, savepath,False)

    

    
    fig.savefig(os.path.join(savepath,'Fig_2.svg'))



    plt.tight_layout()
    plt.show()

    





Figure2(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data/Fig_2_data.csv',
        savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Figures')