import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats as st
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
sns.set_context('talk')



def Figure7(
        cell_datapath,
        savepath,
        ):
    

    palette_bars = {'Unlabeled': '#E7E7E7',
                    'IP-derived': '#EAF3E4'}
        
    palette_points = {'Unlabeled': '#808080',
                    'IP-derived': '#8DC170'}
    order = ['Unlabeled','IP-derived']
    
    data_to_plot = pd.read_csv(cell_datapath)
    data_to_plot = data_to_plot[data_to_plot.cell_type !='distal']


    fig = plt.figure(constrained_layout=True, figsize=(7, 9))
    gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1, 1.5],
                          wspace=0.1, 
                          hspace=0.1)
    



    ginh_freq = fig.add_subplot(gs[0,0])
    y = 'Input_Freq_nS'
    this_ax = ginh_freq
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('ginh event freq (Hz)')
    sns.barplot(data=data_to_plot,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)
    

    lfp_freq = fig.add_subplot(gs[0,1])
    y = 'LFP_Freq'
    this_ax = lfp_freq
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('LFP event freq (Hz)')
    sns.barplot(data=data_to_plot,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)
    
    
    
    ginh_event_amp = fig.add_subplot(gs[1,0])
    y = 'Input_Amp_nS'
    this_ax = ginh_event_amp
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('ginh event amp (nS)')
    sns.barplot(data=data_to_plot,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)
    

    ginh_event_amp = fig.add_subplot(gs[1,1])
    y = 'LFP_Amp'
    this_ax = ginh_event_amp
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('LFP event amp (\u03BCV)')
    sns.barplot(data=data_to_plot,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)
    
    ettc = fig.add_subplot(gs[2,1])
    y = 'STTC'
    this_ax = ettc
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel('ETTC')
    sns.barplot(data=data_to_plot,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=data_to_plot,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)
    
    sns.despine(fig,top=True,right=True)

    fig.savefig(os.path.join(savepath,'Fig_7.svg'))
    plt.show()
    





datapath = '/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data'
Figure7(
        cell_datapath=os.path.join(datapath,'Fig_6_and_7_data.csv'),
        savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Figures'
)