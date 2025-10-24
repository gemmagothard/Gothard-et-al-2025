import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'


def FigureS4(
        datapath,
        savepath,
        ):
    

    palette_bars = {'unlabeled': '#E7E7E7',
                    'IP-derived': '#EAF3E4',
                    'OP-derived': '#F4D1D1'}
        
    palette_points = {'unlabeled': '#808080',
                    'IP-derived': '#8DC170',
                    'OP-derived': '#C70909'}
    
    order = ['unlabeled','OP-derived','IP-derived']
    

    
    data_to_plot = pd.read_csv(datapath)
    print(data_to_plot)


    n_plots = 2
    fig,ax = plt.subplots(1,n_plots,figsize=(4*n_plots,4))
    sns.despine(fig,right=True,top=True)

    y = 'IPSC_freq'
    this_ax = ax[0]
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('IPSC Frequency (Hz)')
    sns.barplot(data=data_to_plot,
                x='Neuron_ID',
                y=y,
                hue='Neuron_ID',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='Neuron_ID',
                hue='Neuron_ID',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)
    
    y = 'IPSC_amp'
    this_ax = ax[1]
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('IPSC amplitude (pA)')
    sns.barplot(data=data_to_plot,
                x='Neuron_ID',
                y=y,
                hue='Neuron_ID',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='Neuron_ID',
                hue='Neuron_ID',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)
    


    plt.show()
    plt.tight_layout()
    fig.savefig(os.path.join(savepath,'Fig_S4.svg'))










FigureS4(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data/Fig_S4_data.csv',
        savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Figures')