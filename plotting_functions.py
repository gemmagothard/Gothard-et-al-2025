#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:14:14 2020

@author: gemmagothard
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import scipy.signal as spsig
import seaborn as sns
import math
import matplotlib.ticker as plticker
import matplotlib as mpl
from numpy import median
mpl.rcParams['svg.fonttype'] = 'none'



sns.set_context("talk")

def find_nearest(array, value):
    
    array = np.asarray(array)
    
    idx = (np.abs(array - value)).argmin()
    
    if array[idx] < value:
        idx = idx+1
    
    return array[idx],idx

#%%
###### ========================= EQUALITY PLOT
def equality_plot(ax,data1,data2,neuron_type,title,save_figs,savefolder,animal_type):

    #fig, ax = plt.subplots(1,1,sharey=True,sharex=True)

    min_val = np.nanmin([data1, data2]) - np.nanstd([data1,data2])
    max_val = np.nanmax([data1, data2]) + np.nanstd([data1,data2])

    abs_max = np.nanmax(np.abs([data1, data2])) + np.nanstd([data1,data2])

    equality_line = np.arange(min_val,max_val,0.01)
    ax.plot(equality_line,equality_line,'k',alpha=0.3,linewidth = '3',linestyle=':')
    
    ax.scatter(data1,data2,marker="D",s=50,color='grey')
    ax.scatter(np.nanmean(data1), np.nanmean(data2),s=50,color='black',marker="D")
    ax.errorbar(np.nanmean(data1), np.nanmean(data2),xerr=st.sem(data1,nan_policy='omit'), yerr =st.sem(data2,nan_policy='omit'), 
             fmt=' ', color = 'black', elinewidth = 2, capthick = 2)
    
    max_vals   = [-600, -70, -60,  -50, -45, -40, 0.2,  0.3, 0.5, 1.5,  2,  5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]  # maximum tick label 
    step_sizes = [-100, -10, -10,  -50,   5,   5, 0.05, 0.1, 0.1, 0.2, 0.5, 1,  2, 5, 10,  20,  50,  100,  200,  500, 1000,  2000, 5000]   # tick step size correpsonding to max tick label. e.g. if you max tick is 200 you tick step size will be 50

    rounded_maxval,idx = find_nearest(max_vals, abs_max)
    this_stepsize = step_sizes[idx]
    
    loc = plticker.MultipleLocator(abs(this_stepsize)) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    ax.tick_params(axis='both', which='major')  
    #ax.rcParams['svg.fonttype'] = 'none'

    
    ratio = 1
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_title(title)
    
    if neuron_type == 'IP_OP':
        ax.set_xlabel("OP-derived IPSC (pA)")
        ax.set_ylabel("IP-derived IPSC (pA)")
        ax.xaxis.label.set_color('red')
        ax.yaxis.label.set_color('green')
        
    if neuron_type == 'unlabeled':
        ax.set_xlabel("Unlabelled cell 1")
        ax.set_ylabel("Unlabelled cell 2")
        
    if save_figs == True:
        figname = f'{neuron_type}_{title}_equality_plot.svg'
        fig.savefig(savefolder+figname,format='svg',bbox_inches='tight')

    
        
        
#%%
###### ========================= PAIRED LINE PLOT 
    
def pairedlineplot(ax,data1,data2,xlabel,ylabel,neuron_type,save_figs,savefolder,distance_plot,animal_type):
    
    counter = np.arange(0,len(data1))

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    
    one_vec = np.ones(len(data1))
    two_vec = np.ones(len(data2))*2
    
    #fig, ax = plt.subplots(1,1)

    for j in counter:
        ax.plot([one_vec[j], two_vec[j]], [data1[j], data2[j]],'.-',color='grey',markersize=7,alpha=0.7)



    ax.plot([1, 2], [np.nanmean(data1),np.nanmean(data2)],'.-',color='black',markersize=9)                
    ax.errorbar([1,2], [np.nanmean(data1),np.nanmean(data2)], yerr=[st.sem(data1,nan_policy='omit'),st.sem(data2,nan_policy='omit')],color='black')
  
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(0,3)
    
    if distance_plot == True:
        ax.set_ylim(np.nanmax([data1,data2])+20,0)
    
    ratio = 2
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_xlim(0,3)
    ax.tick_params(axis='x',bottom=False,labelbottom=False)
    ax.tick_params(axis='y')
    #ax.rcParams['svg.fonttype'] = 'none'
    
    if save_figs == True:
        figname = animal_type + ylabel + '.svg'
        fig.savefig(savefolder+figname,format='svg')
        
        
        
#%% 
##### ============== BIAS PLOT

def biasplot(ax,data1,data2,title,neuron_type,save_figs,savefolder,animal_type):

    bias_index = data2 / np.sum([data1,data2],axis=0)

    
    #fig, ax1 = plt.subplots(1,1,sharey=True,figsize=(2,6))
    sns.stripplot(ax=ax,data=bias_index,marker="D",s=5,color='grey',zorder=1,jitter=0.1)
    
    ax.errorbar(0, np.nanmean(bias_index),yerr =st.sem(bias_index,nan_policy='omit'), 
             fmt=' ', color = 'black', elinewidth = 2, capthick = 2,marker='D',markersize=6,zorder=2)
 
    ax.set_ylim(0,1)
    ax.hlines(0.5,-0.1,0.1,color='black')
    ax.set_xlim(-0.5,0.5)
    ax.xaxis.set_visible(False)
    ax.set_title(title)
    
    if neuron_type == "IP_OP":
        ax.set_ylabel("Bias index (IP / IP + OP)")
        
    if neuron_type == "unlabeled":
        ax.set_ylabel("Bias index (cell1 / cell1 + cell2)")
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    #ax1.rcParams['svg.fonttype'] = 'none'
    
    ratio = 2
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    
    #ax2 = sns.distplot(bias_index,vertical=True)
    #ax2.rcParams['svg.fonttype'] = 'none'
    
    if save_figs == True:
        figname = f'{neuron_type}_{title}_biasplot.svg'
        fig.savefig(savefolder+figname,format='svg',bbox_inches='tight')
        

    return 
    #%%
    
    #####========== SCATTER BAR PLOT
    
def scatter_bar_plot(ax,data, palette, labels, paired ,ylabel, save_figs, savefolder,flip_y):

    mpl.rcParams['svg.fonttype'] = 'none'
    #fig = plt.figure(figsize = (4,4))

    sns.stripplot(data = data, jitter = 0.1, zorder = 1, palette = palette, ax=ax,clip_on = False,alpha=1)
    sns.barplot(data=data, edgecolor = 'black', palette=palette,errorbar = 'se', ax=ax,zorder = 0,alpha=.3) #estimate=median to plot median
 
    ax.set_xlim(-.75, len(data[:-1])+0.75)
 
    if paired == True:
       
    #paired plot only works if there are two conditions   
 
        for points in range(len(data[0])):
 
            ax.plot((0, 1), (data[0][points], data[1][points]), color = 'grey', alpha = .2, zorder = 2)
 
    
    if flip_y == True:
        axes = plt.gca()
        y_min, y_max = axes.get_ylim()
        ax.set_ylim(y_max,y_min)
    
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
 
    if save_figs == True:
        
        figname = f'{ylabel} .svg'
        fig.savefig(savefolder + figname,format='svg',bbox_inches='tight')
        
        
    return 


def plot_each_pair(df,timevec2):
    
    # plot each pair
    for x in df.iterrows():
        
        fig, (ax,ax1) = plt.subplots(2,1)
        ax.set_title(x[1].Pair)
        ax.set_ylabel("Current (pA)")
        ax.plot(timevec2,x[1].trace1,c=x[1].Pipette1)
        ax.plot(timevec2,x[1].trace2,c=x[1].Pipette2)
    
        ax1.plot(timevec2,x[1].normalised1,c=x[1].Pipette1)
        ax1.plot(timevec2,x[1].normalised2,c=x[1].Pipette2)
        
    
def plot_averages(data1,data2,timevec2,c1,c2,ylabel,stim_windows,save_figs,title,animal_type,savefolder):

    data1_sem = np.std(np.array(data1)) / np.sqrt(len(data1))
    data2_sem = np.std(np.array(data2)) / np.sqrt(len(data2))
    
    # plot averages
    fig, (ax,ax1) = plt.subplots(2,1,sharex=True,sharey=True,figsize=(6,4))
    ax1.set_ylabel(ylabel)
    ax1.plot(timevec2,np.average(data1),c=c1,linewidth=1)
    ax1.plot(timevec2,np.average(data2),c=c2,linewidth=1)
    
    ax1.fill_between(timevec2,np.average(data1)-data1_sem,np.average(data1)+data1_sem,color=c1,alpha=.2,lw=0)
    ax1.fill_between(timevec2,np.average(data2)-data2_sem,np.average(data2)+data2_sem,color=c2,alpha=.2,lw=0)
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    peak1 = []
    peak2 = []

    for c in stim_windows:
        area1 = c
        area2 = area1 + 100
        peak1.append([np.max(x[area1:area2]) for x in data1])
        peak2.append([np.max(x[area1:area2]) for x in data2])
        
        
    sem_data1 = np.std(np.array(peak1),axis=1) / np.sqrt(len(peak1))
    sem_data2 = np.std(np.array(peak2),axis=1) / np.sqrt(len(peak2))
    

    ax.plot(timevec2[stim_windows],np.average(peak1,axis=1),'.-',c=c1)
    ax.fill_between(timevec2[stim_windows],np.average(peak1,axis=1)-sem_data1,np.average(peak1,axis=1)+sem_data1,color=c1,alpha=.2,lw=0)
    
    ax.plot(timevec2[stim_windows],np.average(peak2,axis=1),'.-',c=c2)
    ax.fill_between(timevec2[stim_windows],np.average(peak2,axis=1)-sem_data2,np.average(peak2,axis=1)+sem_data2,color=c2,alpha=.2,lw=0)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    
    if save_figs == True:
        figname = animal_type + title + 'average.svg'
        fig.savefig(savefolder + figname,format='svg')
        
    

def add_figure_letter(ax, letter, fontsize=12):
    """
    Add a letter to the top left corner of an axis for figure labeling.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axis to add the letter to.
        letter (str): The letter to add.
        fontsize (int): Font size for the letter.
    """
    ax.text(-0.1, 1.1, letter, fontsize=fontsize, fontweight='bold', va='top', ha='left', transform=ax.transAxes)




data_to_plot = pd.read_csv('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Fig_1_data.csv')
interneuron_subclasses = ['PV','SST','VIP']
savefig = False  # make true to save individual plots
savefolder =  '/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_paper/PAPER TEXT/files for submission/2025 nature neuro/Gothard-et-al-github/Fig_1_figures'
palette = ['#C70909', '#8DC170']  # red and green 

fig,ax = plt.subplots(3,3,gridspec_kw={'width_ratios':[1,.5,1],'height_ratios':[1,1,1]},figsize=(10,12))
for idx,interneuron_subclass in enumerate(interneuron_subclasses):

    data_subset = data_to_plot[data_to_plot.interneuron_type == interneuron_subclass]
    pyramidal_type = data_subset['pyramidal_type'].iloc[0] 
    interneuron_type = data_subset['interneuron_type'].iloc[0]
    
    equality_plot(ax[idx,0],data_subset.OP_IPSC_peak.values,data_subset.IP_IPSC_peak.values,pyramidal_type,f'{interneuron_subclass} IPSC peak',savefig,savefolder,interneuron_type)
    biasplot(ax[idx,1],data_subset.OP_IPSC_peak.values,data_subset.IP_IPSC_peak.values,' ',pyramidal_type,savefig,savefolder,interneuron_type)
    scatter_bar_plot(ax[idx,2],[data_subset.OP_pial_dist.values,data_subset.IP_pial_dist.values], palette, ['OP','IP'], True ,"Pial distance (\u03BCm)", savefig, savefolder,False)

plt.tight_layout()
fig.savefig(os.path.join(savefolder, 'Fig_1_interneuron_panels.svg'), format='svg', bbox_inches='tight')

plt.show()
