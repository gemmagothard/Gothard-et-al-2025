import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'



def Figure_S5(datapath,
              savepath
              ):


    palette_bars = {'unlabeled': '#E7E7E7',
                    'IP-OP': '#EAF3E4',
                    'OP-derived': '#F4D1D1'}
        
    palette_points = {'unlabeled': '#808080',
                    'IP-OP': '#8DC170',
                    'OP-derived': '#C70909'}
    
    order = ['unlabeled','IP-OP']


    data_to_plot = pd.read_csv(datapath)

    n_plots = 2
    fig,ax = plt.subplots(1,n_plots,figsize=(4*n_plots,4))
    sns.despine(fig,right=True,top=True)

    y = 'shared_input_amp'
    this_ax = ax[0]
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('Shared input amplitude (pA)')
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
    
    y = 'Pair_dist'
    this_ax = ax[1]
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('Pial distance (\u03BCm) ')
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
    

    plt.show()

    plt.tight_layout()
    fig.savefig(os.path.join(savepath,'Figure_S5.svg'))
    



Figure_S5(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data/Fig_S5_data.csv',
        savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Figures')
