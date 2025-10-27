import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotting_functions
import os
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'

def FigureS6(
        datapath,
        savepath,
        ):
    

    palette_bars = {'unlabeled': '#E7E7E7',
                    'IP-OP': '#EAF3E4'}
        
    palette_points = {'unlabeled': '#808080',
                    'IP-OP': '#8DC170'}
    
    data_to_plot = pd.read_csv(datapath)

    n_plots = 2
    fig,ax = plt.subplots(1,n_plots,figsize=(4*n_plots,4))
    sns.despine(fig,right=True,top=True)
    order = ['unlabeled', 'IP-OP']

    data_to_plot['mean_amplitude'] = data_to_plot[['Neuron1_amplitude','Neuron2_amplitude']].mean(axis=1)
    data_to_plot['mean_pial_dist'] = data_to_plot[['Neuron1_pial_dist','Neuron2_pial_dist']].mean(axis=1)

    group_by_pairs = data_to_plot.groupby(['Pair_ID']).agg({'mean_amplitude':'mean',
                                                            'mean_pial_dist':'mean',
                                                            'Pair_type':'first'})


    print(group_by_pairs)

    y = 'mean_amplitude'
    this_ax = ax[0]
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('IPSC amplitude (pA)')
    sns.barplot(data=group_by_pairs,
                x='Pair_type',
                y=y,
                hue='Pair_type',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=group_by_pairs,
                x='Pair_type',
                hue='Pair_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)
    

    y = 'mean_pial_dist'
    this_ax = ax[1]
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('Pial distance (\u03BCm)')
    sns.barplot(data=group_by_pairs,
                x='Pair_type',
                y=y,
                hue='Pair_type',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=group_by_pairs,
                x='Pair_type',
                hue='Pair_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)
    
    plt.tight_layout()
    fig.savefig(os.path.join(savepath,'Fig_S6.svg'))

    plt.show()



FigureS6(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data/Fig_4_data.csv',
        savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Figures')
