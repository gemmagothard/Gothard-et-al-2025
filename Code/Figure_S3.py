
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotting_functions
import os

def FigureS3(
        datapath,
        savepath,
        ):

    data_to_plot = pd.read_csv(datapath)

    interneuron_subclasses = ['PV','SST','VIP']
    pyramidal_type = data_to_plot['pyramidal_type'].iloc[0] 
    savefig = False  # make true to save individual plots

    fig,ax = plt.subplots(3,2,gridspec_kw={'width_ratios':[1,.5],'height_ratios':[1,1,1]},figsize=(10,12))
    labels = ['Cell 1','Cell 2']

    for idx,interneuron_subclass in enumerate(interneuron_subclasses):

        data_subset = data_to_plot[data_to_plot.interneuron_type == interneuron_subclass]
        interneuron_type = data_subset['interneuron_type'].iloc[0]
        
        plotting_functions.equality_plot(ax[idx,0],data_subset.N1_IPSC_peak.values,data_subset.N2_IPSC_peak.values,pyramidal_type,f'{interneuron_subclass} IPSC peak',savefig,savepath,interneuron_type)
        plotting_functions.biasplot(ax[idx,1],data_subset.N1_IPSC_peak.values,data_subset.N2_IPSC_peak.values,' ',pyramidal_type,savefig,savepath,interneuron_type)


    plt.tight_layout()
    fig.savefig(os.path.join(savepath, 'Fig_S3_interneuron_panels.svg'), format='svg', bbox_inches='tight')

    plt.show()


FigureS3(datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Fig_S3_data.csv',
        savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Fig_S3',
)