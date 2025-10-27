    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import re
import seaborn as sns
import os
from plotting_functions import add_figure_letter
import scipy.stats as st
from scipy.stats import ttest_ind

from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import *
import matplotlib as mpl
sns.set_context('paper')
mpl.rcParams['font.family'] = 'Arial'
"""to write mu: \u03BC """
mpl.rcParams['svg.fonttype'] = 'none'
cm = 1/2.54  # centimeters in inches


def Figure5_and_S7(PV_datapath,
            input_datapath,
            all_starters_datapath,
            starters_by_brain_datapath,
            savepath):
    

    PV_df = pd.read_csv(PV_datapath)
    all_starters_df = pd.read_csv(all_starters_datapath)
    starter_grouped_by_brain = pd.read_csv(starters_by_brain_datapath)
    inputs_by_layers = pd.read_csv(input_datapath)

    # set the colors for Cre-ON and Cre-OFF 
    palette = {'Ta1 Cre-ON':'#8DC170',
                'Ta1 Cre-OFF':'#C70909'}  
    hue_order = ['Ta1 Cre-ON', 'Ta1 Cre-OFF']

    main_fig,  [[example_im1, example_im2],
                [PV_cloud_plot_OFF, PV_cloud_plot_ON],
                [cumulative_plot, distribution_plot]]  = plt.subplots(3,2,figsize=(7,10),gridspec_kw={'width_ratios':[1,1],
                                                                                    'height_ratios':[1.3,1,1]})
    add_figure_letter(example_im1,'a')
    add_figure_letter(example_im2,'c')
    add_figure_letter(PV_cloud_plot_OFF,'d')
    add_figure_letter(cumulative_plot,'e')
    add_figure_letter(distribution_plot,'f')

    example_im1.axis('off')
    example_im2.axis('off')


    # add in PV inputs by layer plot
    sub_norm_PV = inset_axes(
            parent_axes=example_im2,
            width="30%",
            height="100%",
            loc='upper right',
            borderpad=1 # padding between parent and inset axes
        )
    # order of layers for plotting
    layer_map = {'Layer 1': 'L1',
                 'Layer 2/3': 'L2/3',
                 'Layer 4': 'L4',
                 'Layer 5a': 'L5a',
                 'Layer 5b': 'L5b',
                 'Layer 6': 'L6'}
    inputs_by_layers['layer'] = inputs_by_layers['layer'].map(layer_map)
    layers = ['L1',
            'L2/3',
            'L4',
            'L5a',
            'L5b',
            'L6']
    sns.barplot(inputs_by_layers[inputs_by_layers.cell_id=='pv_inputs'],
                    y='layer',
                    x='normalized_counts',
                    hue='condition',
                    hue_order=hue_order,
                    orient='h',
                    order=layers,
                    palette=palette,
                    edgecolor='black',
                    ax = sub_norm_PV,
                    estimator='mean',
                    errorbar='se')
    sns.stripplot(inputs_by_layers[inputs_by_layers.cell_id=='pv_inputs'],
                    y='layer',
                    x='normalized_counts',
                    hue='condition',
                    hue_order=hue_order,
                    orient='h',
                    order=layers,
                    palette=palette,
                    dodge=True,
                    size=3,
                    ax = sub_norm_PV)
    sub_norm_PV.set_xlabel('PV input cells (%)')
    sub_norm_PV.get_legend().remove()
    sub_norm_PV.set_ylabel(' ')

    sub_barplot = inset_axes(
                parent_axes=cumulative_plot,
                width="25%",
                height="50%",
                loc='lower right',
                borderpad=1.5  # padding between parent and inset axes
            )

    aspect_ratio = 14.51/ 11.62

    supp_fig = plt.figure(figsize=(8,8/aspect_ratio))
    supp_gs = gridspec.GridSpec(2, 4, height_ratios=[1.3, 1])  # 2 rows, 4 columns
    
    starter_pial_dist = supp_fig.add_subplot(supp_gs[1, 0])
    y_spread_plot = supp_fig.add_subplot(supp_gs[1, 1])
    z_spread_plot = supp_fig.add_subplot(supp_gs[1, 2])
    pv_starter_ratio = supp_fig.add_subplot(supp_gs[1, 3])
    cre_off_fits = supp_fig.add_subplot(supp_gs[0, 0:2])
    cre_on_fits = supp_fig.add_subplot(supp_gs[0, 2:4])


    add_figure_letter(cre_off_fits,'a')
    add_figure_letter(cre_on_fits,'b')
    add_figure_letter(starter_pial_dist,'c')
    add_figure_letter(y_spread_plot,'d')
    add_figure_letter(z_spread_plot,'e')
    add_figure_letter(pv_starter_ratio,'f')


    sub_residual_ON = inset_axes(
            parent_axes=cre_on_fits,
            width="15%",
            height="20%",
            loc='upper right',
            borderpad=1 # padding between parent and inset axes
        )
    sub_residual_OFF = inset_axes(
        parent_axes=cre_off_fits,
        width="15%",
        height="20%",
        loc='upper right',
        borderpad=1 # padding between parent and inset axes
    )

    sns.ecdfplot(data=PV_df,
                 x="XYZ_distance",
                 hue='condition',
                 palette=palette,
                 hue_order=hue_order,
                 ax=cumulative_plot,
                 legend=False,
                 stat='proportion'
                 )
    cumulative_plot.set_ylabel('Proportion of PV inputs')
    cumulative_plot.set_xlabel('Euclidean distance to mean starter position (\u03BCm)')

    # Now make the 2D KDE plot
    sns.kdeplot(data=PV_df[PV_df.condition=='Ta1 Cre-ON'],
        x="X_distance",
        y="PV_relative_to_starter_Y",
        hue='condition',
        palette=palette,
        ax= PV_cloud_plot_ON,
        fill=True,
        legend=False,
        rasterized=True,
        zorder=-10,
        levels=100)
    all_starters_df['starter_X'] = all_starters_df.starter_X - np.mean(all_starters_df.starter_X)
    
    PV_cloud_plot_ON.set_rasterization_zorder(0)
    sns.kdeplot(data=PV_df[PV_df.condition=='Ta1 Cre-OFF'],
        x="X_distance",
        y="PV_relative_to_starter_Y",
        hue='condition',
        palette=palette,
        ax= PV_cloud_plot_OFF,
        fill=True,
        zorder=-10,
        legend=False,
        levels=100)
    PV_cloud_plot_OFF.set_rasterization_zorder(0)
    ON_starter = all_starters_df[all_starters_df.condition=='Ta1 Cre-ON'].starter_Y
    OFF_starter = all_starters_df[all_starters_df.condition=='Ta1 Cre-OFF'].starter_Y
    PV_cloud_plot_ON.plot(0,np.average(ON_starter), 'o', color='black', markersize=7, label='Mean \nstarter \nposition')
    PV_cloud_plot_OFF.plot(0,np.average(OFF_starter), 'o', color='black', markersize=7, label='Mean \nstarter \nposition')
    ON_starter_x = all_starters_df[all_starters_df.condition=='Ta1 Cre-ON'].starter_X
    OFF_starter_x = all_starters_df[all_starters_df.condition=='Ta1 Cre-OFF'].starter_X
    PV_cloud_plot_ON.errorbar(0, np.average(ON_starter), yerr=np.std(ON_starter), xerr=np.std(ON_starter_x), fmt='none', color='black', capsize=5)
    PV_cloud_plot_OFF.errorbar(0, np.average(OFF_starter), yerr=np.std(OFF_starter), xerr=np.std(OFF_starter_x), fmt='none', color='black', capsize=5)
    PV_cloud_plot_OFF.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=8, frameon=False)
    PV_cloud_plot_ON.set_xlabel('Position in horizontal space (\u03BCm)')
    PV_cloud_plot_ON.invert_yaxis()
    PV_cloud_plot_OFF.invert_yaxis()
    PV_cloud_plot_OFF.sharex(PV_cloud_plot_ON)
    PV_cloud_plot_OFF.sharey(PV_cloud_plot_ON)
    PV_cloud_plot_ON.yaxis.set_visible(False)
    PV_cloud_plot_ON.set_title('PV inputs to IP-derived starter neurons')
    PV_cloud_plot_OFF.set_title('PV inputs to OP-derived starter neurons')
    PV_cloud_plot_OFF.set_ylabel('Pial distance (\u03BCm)')
    PV_cloud_plot_ON.set_ylabel(' ')
    PV_cloud_plot_ON.set_xlabel('Lateral distance (\u03BCm)')
    PV_cloud_plot_OFF.set_xlabel('Lateral distance (\u03BCm)')


    ON_std_lateral = np.std(PV_df[PV_df.condition=='Ta1 Cre-ON'].X_distance)
    ON_std_vertical = np.std(PV_df[PV_df.condition=='Ta1 Cre-ON'].PV_relative_to_starter_Y)

    OFF_std_lateral = np.std(PV_df[PV_df.condition=='Ta1 Cre-OFF'].X_distance)
    OFF_std_vertical = np.std(PV_df[PV_df.condition=='Ta1 Cre-OFF'].PV_relative_to_starter_Y)

    print(f"Standard deviations along x and y axis, ON data: {ON_std_lateral}, {ON_std_vertical}")
    print(f"Standard deviations along x and y axis, OFF data: {OFF_std_lateral}, {OFF_std_vertical}")



    # test the modality of the distributions
    # take the X_distances where Y is between 400 and 600um
    ON_data = PV_df.loc[(PV_df['condition']=='Ta1 Cre-ON') &
                            (PV_df['PV_relative_to_starter_Y'].between(400, 600)),
                            ['X_distance']].sort_values('X_distance').values.reshape(1,-1)
    
    OFF_data = PV_df.loc[(PV_df['condition']=='Ta1 Cre-OFF') &
                            (PV_df['PV_relative_to_starter_Y'].between(400, 600)),
                            ['X_distance']].sort_values('X_distance').values.reshape(1,-1)
    

    points = np.linspace(-700,700,250)
    bins = np.arange(-750,750,90)

    distribution_ON = st.gaussian_kde(np.array(ON_data)).evaluate(points)
    distribution_OFF = st.gaussian_kde(np.array(OFF_data)).evaluate(points)

    distribution_plot.plot(points,distribution_ON, label='Data',color=palette['Ta1 Cre-ON'])
    distribution_plot.plot(points,distribution_OFF, label='Data',color=palette['Ta1 Cre-OFF'])
    distribution_plot.set_xlabel('Lateral distance to mean starter position (\u03BCm)')
    distribution_plot.set_ylabel('Probability density of PV inputs')
    
    ON_histogram = np.histogram(ON_data, bins=bins, density=True)[0]
    OFF_histogram = np.histogram(OFF_data, bins=bins, density=True)[0]


    # fit a bimodal and unimodal model to the data
    bimodal_model_ON = GeneralMixtureModel([Normal(), Normal()],verbose=True).fit(ON_data.reshape(-1,1))
    unimodal_model_ON = Normal().fit(ON_data.reshape(-1,1))

    # evaluate the models on the points and plot alongside the KDE (fit based on real data)
    cre_on_fits.plot(points, bimodal_model_ON.probability(points.reshape(-1,1)), label='Bimodal fit', color='black',linestyle='--')
    cre_on_fits.plot(points, unimodal_model_ON.probability(points.reshape(-1,1)), label='Unimodal fit',color='grey', linestyle='--')
    cre_on_fits.plot(points,distribution_ON, label='IP-derived data',color=palette['Ta1 Cre-ON'],linewidth=2)
    cre_on_fits.set_xlabel('Lateral distance to average starter neuron position (\u03BCm)')
    cre_on_fits.set_ylabel('Probability density of PV inputs')
    cre_on_fits.legend(loc='upper left')

    # get residuals 
    ON_residuals_bimodal = np.abs(distribution_ON - np.array(bimodal_model_ON.probability(points.reshape(-1,1))))
    ON_residuals_unimodal = np.abs(distribution_ON - np.array(unimodal_model_ON.probability(points.reshape(-1,1))))

    ON_summed_residuals_bimodal = np.sum(ON_residuals_bimodal)
    ON_summed_residuals_unimodal = np.sum(ON_residuals_unimodal)
    print(f'Summed residuals, ON data, bimodal: {ON_summed_residuals_bimodal}, unimodal: {ON_summed_residuals_unimodal}')

    bimodal_model_OFF = GeneralMixtureModel([Normal(), Normal()],verbose=True).fit(OFF_data.reshape(-1,1))
    unimodal_model_OFF = Normal().fit(OFF_data.reshape(-1,1))

    cre_off_fits.plot(points, bimodal_model_OFF.probability(points.reshape(-1,1)), label='Bimodal fit', color='black',  linestyle='--')
    cre_off_fits.plot(points, unimodal_model_OFF.probability(points.reshape(-1,1)), label='Unimodal fit', color='grey', linestyle='--')
    cre_off_fits.plot(points,distribution_OFF, label='OP-derived data',color=palette['Ta1 Cre-OFF'],linewidth=2)
    cre_off_fits.legend(loc='upper left')
    cre_off_fits.set_xlabel('Lateral distance to average starter neuron position (\u03BCm)')
    cre_off_fits.set_ylabel('Probability density of PV inputs')


    # get residuals 
    OFF_residuals_bimodal = np.abs(distribution_OFF - np.array(bimodal_model_OFF.probability(points.reshape(-1,1))))
    OFF_residuals_unimodal = np.abs(distribution_OFF - np.array(unimodal_model_OFF.probability(points.reshape(-1,1))))

    OFF_summed_residuals_bimodal = np.sum(OFF_residuals_bimodal)
    OFF_summed_residuals_unimodal = np.sum(OFF_residuals_unimodal)

    print(f'Summed residuals, OFF data, bimodal: {OFF_summed_residuals_bimodal}, unimodal: {OFF_summed_residuals_unimodal}')


    residual_df = pd.DataFrame({'condition':['Ta1 Cre-ON', 'Ta1 Cre-ON', 'Ta1 Cre-OFF', 'Ta1 Cre-OFF'],
                    'Fit':  ['bimodal', 'unimodal', 'bimodal', 'unimodal'],
        'Summed residuals': [ON_summed_residuals_bimodal, ON_summed_residuals_unimodal, OFF_summed_residuals_bimodal, OFF_summed_residuals_unimodal]})

    palette_residuals = {'bimodal':'black',
                         'unimodal':'grey'}
    
    sns.barplot(residual_df[residual_df.condition=='Ta1 Cre-ON'],
                x='Fit',
                y='Summed residuals',
                hue='Fit',
                palette=palette_residuals,
                ax= sub_residual_ON,
                legend=False,
                )
    sub_residual_ON.set_title('Residuals')
    sub_residual_ON.set_ylabel(' ')
    sub_residual_ON.set_xlabel(' ')
    sub_residual_ON.xaxis.set_visible(False)

    sns.barplot(residual_df[residual_df.condition=='Ta1 Cre-OFF'],
            x='Fit',
            y='Summed residuals',
            hue='Fit',
            palette=palette_residuals,
            ax= sub_residual_OFF,
            legend=False
            )
    sub_residual_OFF.set_title('Residuals')
    sub_residual_OFF.set_ylabel(' ')
    sub_residual_OFF.set_xlabel(' ')
    sub_residual_OFF.xaxis.set_visible(False)

    # KS test
    res = st.kstest(np.array(ON_data.flatten()),np.array(OFF_data.flatten()),alternative='two-sided')
    print(f'ks test two sided: {res.pvalue}')

    res = st.kstest(bimodal_model_ON.probability(points.reshape(-1,1)),unimodal_model_OFF.probability(points.reshape(-1,1)), alternative='two-sided')
    print(f'KS test, statistic: {res.statistic}, p-value: {res.pvalue}')

    sns.barplot(PV_df,
            x='condition',
            y='XYZ_distance',
            hue='condition',
            hue_order=hue_order,
            ax = sub_barplot,
            palette=palette,
            estimator='mean',
            errorbar='se')
    sub_barplot.set_xlabel(' ')
    sub_barplot.set_ylabel('Distance (\u03BCm)')
    sub_barplot.xaxis.set_visible(False)

    pv_on = PV_df[PV_df.condition=='Ta1 Cre-ON'].XYZ_distance
    pv_off = PV_df[PV_df.condition=='Ta1 Cre-OFF'].XYZ_distance
    stat, p1 = st.shapiro(pv_on)
    stat, p2 = st.shapiro(pv_off)
    print(str(p1)+ " " + str(p2))
    alpha = 0.05
    if np.any([p1 > alpha, p2 > alpha]):
        print("Distribution is normal")    
    else:
        print("Sample is not normal")

    t_stat, p_value = st.mannwhitneyu(pv_on, pv_off)
    print(f'Starter-PV dist, number of cre-ON brains: {len(pv_on)}, mean: {np.mean(pv_on)}, sem: {st.sem(pv_on)} number of cre-OFF brains: {len(pv_off)}, mean: {np.mean(pv_off)}, sem: {st.sem(pv_off)}, p_val: {p_value}, t_stat: {t_stat}')

    
    starter_on = starter_grouped_by_brain[starter_grouped_by_brain.condition=='Ta1 Cre-ON'].mean_starter_pial_dist
    starter_off = starter_grouped_by_brain[starter_grouped_by_brain.condition=='Ta1 Cre-OFF'].mean_starter_pial_dist
    t_stat, p_value = ttest_ind(starter_on, starter_off, equal_var=False)
    print(f'Starter cell pial dist number of cre-ON brains: {len(starter_on)}, number of cre-OFF brains: {len(starter_off)}, p_val: {p_value}, t_stat: {t_stat}')
    
    sns.barplot(starter_grouped_by_brain,
                x='condition',
                y='mean_starter_pial_dist',
                hue='condition',
                hue_order=hue_order,
                ax=starter_pial_dist,
                palette=palette,
                legend=True,
                )
    starter_pial_dist.set_ylabel('Starter neuron pial distance (\u03BCm)')
    starter_pial_dist.xaxis.set_visible(False)
    starter_pial_dist.legend(loc='upper right', fontsize=8, frameon=False)
    
    sns.barplot(starter_grouped_by_brain,
        x='condition',
        y='starter_X_spread',
        hue='condition',
        hue_order=hue_order,
        ax= y_spread_plot,
        palette=palette,)
    y_spread_plot.set_ylabel('Starter neuron lateral position SD')
    y_spread_plot.xaxis.set_visible(False)

    sns.barplot(starter_grouped_by_brain,
        x='condition',
        y='starter_Z_spread',
        hue='condition',
        hue_order=hue_order,
        ax= z_spread_plot,
        palette=palette,)
    z_spread_plot.set_ylabel('Starter neuron Z position SD')
    z_spread_plot.xaxis.set_visible(False)

    starter_grouped_by_brain['input_starter_ratio'] = starter_grouped_by_brain.number_of_PV_inputs / starter_grouped_by_brain.number_of_starters 
    sns.barplot(starter_grouped_by_brain,
        x='condition',
        y='input_starter_ratio',
        hue='condition',
        hue_order=hue_order,
        ax= pv_starter_ratio,
        palette=palette)
    pv_starter_ratio.set_ylabel('PV input to starter neuron ratio')
    pv_starter_ratio.xaxis.set_visible(False)

    sns.despine(fig=main_fig,top=True,right=True)
    sns.despine(fig=supp_fig,top=True,right=True)

    # make supplementary figure
    plt.rcParams.update({'font.size': 11})
    supp_fig.tight_layout()
    main_fig.tight_layout()

    plt.show()

    main_fig.savefig(os.path.join(savepath,'Fig_5.svg'),bbox_inches='tight',dpi=300)
    supp_fig.savefig(os.path.join(savepath,'Fig_S7.svg'),bbox_inches='tight',dpi=300)




Figure5_and_S7(PV_datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data/Fig_5_PVdata.csv',
        input_datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data/Fig_5_PV_inputs.csv',
        all_starters_datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data/Fig_5_starterdata.csv',
        starters_by_brain_datapath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Data/Fig_5_starterbybraindata.csv',
        
        savepath='/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Figures'
        )