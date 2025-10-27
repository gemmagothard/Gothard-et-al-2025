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

def Figure6(
        NBQX_datapath,
        traces_datapath,
        cell_datapath,
        savepath,
        ):

    palette_bars = {'Unlabeled': '#E7E7E7',
                    'IP-derived': '#EAF3E4'}
        
    palette_points = {'Unlabeled': '#808080',
                    'IP-derived': '#8DC170'}
    
    NBQX_palette_bars = {'NBQX_proximal_pre':'#C3C3C3',
                         'NBQX_proximal_post':'#808080',
                         'NBQX_distal_pre':'#FAD088',
                         'NBQX_distal_post':'#F49A00'
                         }
    
    order = ['Unlabeled','IP-derived']
    
    NBQX_data = pd.read_csv(NBQX_datapath)
    traces = np.load(traces_datapath)
    cells = pd.read_csv(cell_datapath)
    cells = cells[np.logical_or(cells.cell_type=='IP-derived',cells.cell_type=='Unlabeled')]


    fig = plt.figure(constrained_layout=True, figsize=(7, 9))
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[3, 2], height_ratios=[1, 1],
                          wspace=0, 
                          hspace=0.2)

    IP_mean = np.mean(traces['IP-der'],axis=0)
    IP_sem = st.sem(traces['IP-der'],axis=0)
    unlabeled_mean = np.mean(traces['unlabeled'],axis=0)
    unlabeled_sem = st.sem(traces['unlabeled'],axis=0)

    it_LFP_examples = fig.add_subplot(gs[1,0])
    it_LFP_examples.plot(traces['timevec_trace'],IP_mean,color=palette_points['IP-derived'])
    it_LFP_examples.fill_between(traces['timevec_trace'],IP_mean+IP_sem,IP_mean-IP_sem,color=palette_bars['IP-derived'])
    it_LFP_examples.plot(traces['timevec_trace'],unlabeled_mean,color=palette_points['Unlabeled'])
    it_LFP_examples.fill_between(traces['timevec_trace'],unlabeled_mean+unlabeled_sem,unlabeled_mean-unlabeled_sem,color=palette_bars['Unlabeled'])
    it_LFP_examples.set_ylabel('IT-LFP (\u03BCV)')
    it_LFP_examples.axvline(0,linestyle='dotted',color='black')
    it_LFP_examples.set_xlim(-50,50)


    it_LFP_averages = fig.add_subplot(gs[1,1],sharey=it_LFP_examples)
    y = 'IT_lfp'
    this_ax = it_LFP_averages
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel(' ')
    sns.barplot(data=cells,
                x='cell_type',
                y=y,
                hue='cell_type',
                palette=palette_bars,
                ax=this_ax,
                order = order,
                edgecolor='black',)
    sns.stripplot(data=cells,
                x='cell_type',
                hue='cell_type',
                y=y,
                palette=palette_points,
                order = order,
                ax=this_ax)
    
    
    NBQX_plot = fig.add_subplot(gs[0,:])
    sns.set_context('talk')
    order = ['NBQX_proximal_pre','NBQX_proximal_post','NBQX_distal_pre','NBQX_distal_post']
    sns.barplot(data=NBQX_data, 
                x='condition', 
                y='frequency', 
                errorbar='se', 
                ax=NBQX_plot,
                hue='condition',
                palette=NBQX_palette_bars)
    sns.stripplot(data=NBQX_data, 
                x='condition', 
                y='frequency', 
                hue='condition',
                ax=NBQX_plot,
                jitter=.01,
                palette=['black']*4)
    NBQX_plot.set_ylabel('LFP event frequency (Hz)')
    NBQX_plot.set_xticks([0,1,2,3],['Pre','Post','Pre','Post'])
    NBQX_plot.set_xlabel('Local                            Distal')

    # Connect pre ↔ post for proximal and distal separately
    pairs = [
        ('NBQX_proximal_pre', 'NBQX_proximal_post'),
        ('NBQX_distal_pre',   'NBQX_distal_post')]
    for folder, sub in NBQX_data.groupby('folder'):
        sub = sub.set_index('condition')
        for a, b in pairs:
            if a in sub.index and b in sub.index:          # robust to missing conditions
                x = [order.index(a), order.index(b)]
                y = [sub.at[a, 'frequency'], sub.at[b, 'frequency']]
                NBQX_plot.plot(x, y, color='black', alpha=0.6, linewidth=1, zorder=1)


    sns.despine(fig,right=True,top=True)
    plt.show()
    fig.savefig(os.path.join(savepath,'Fig_6.svg'))



datapath = '/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data'
Figure6(NBQX_datapath=os.path.join(datapath,'Fig_6_NBQXdata.csv'),
        traces_datapath=os.path.join(datapath,'Fig_6_IT_LFP_traces.npz'),
        cell_datapath=os.path.join(datapath,'Fig_6_and_7_data.csv'),
        savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Figures'
)