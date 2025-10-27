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



def FigureS9(
        cell_datapath,
        traces_datapath,
        savepath,
        ):
    

    palette_bars = {'Unlabeled': '#E7E7E7',
                    'IP-derived': '#EAF3E4',
                    'distal': '#FDECCF'}
        
    palette_points = {'Unlabeled': '#808080',
                    'IP-derived': '#8DC170',
                    'distal':'#FACD80'}
    
    order = ['Unlabeled','distal']
    
    data_to_plot = pd.read_csv(cell_datapath)
    data_to_plot = data_to_plot[data_to_plot.cell_type !='IP-derived']

    traces = np.load(traces_datapath)

    fig, (it_LFP_examples, it_LFP_averages) = plt.subplots(1,2,figsize=(8,6),width_ratios=[2,1.3],sharey=True)


    distal_mean = np.mean(traces['distal'],axis=0)
    distal_sem = st.sem(traces['distal'],axis=0)
    unlabeled_mean = np.mean(traces['unlabeled'],axis=0)
    unlabeled_sem = st.sem(traces['unlabeled'],axis=0)

    it_LFP_examples.plot(traces['timevec_trace'],distal_mean,color=palette_points['distal'])
    it_LFP_examples.fill_between(traces['timevec_trace'],distal_mean+distal_sem,distal_mean-distal_sem,color=palette_bars['distal'])
    it_LFP_examples.plot(traces['timevec_trace'],unlabeled_mean,color=palette_points['Unlabeled'])
    it_LFP_examples.fill_between(traces['timevec_trace'],unlabeled_mean+unlabeled_sem,unlabeled_mean-unlabeled_sem,color=palette_bars['Unlabeled'])
    it_LFP_examples.set_ylabel('IT-LFP (\u03BCV)')
    it_LFP_examples.axvline(0,linestyle='dotted',color='black')
    it_LFP_examples.set_xlim(-50,50)


    y = 'IT_lfp'
    this_ax = it_LFP_averages
    this_ax.set_xlabel(' ')
    this_ax.set_ylabel(' ')
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
                size=10,
                ax=this_ax,
                alpha=.7)
    
    this_ax.set_xticks([0,1],['Proximal','Distal'])
    sns.despine(fig,right=True,top=True)
    plt.show()
    fig.savefig(os.path.join(savepath,'Fig_S9.svg'))
    








datapath = '/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data'
FigureS9(cell_datapath=os.path.join(datapath,'Fig_6_and_7_data.csv'),
        traces_datapath=os.path.join(datapath,'Fig_6_IT_LFP_traces.npz'),
        savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Figures'
)