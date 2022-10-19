from calendar import c
import os
import sys
from turtle import color
# import torch
import numpy as np
# import scipy.spatial
# import matplotlib as mplmpl
# mplmpl.use('Agg')
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import math
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams['axes.facecolor'] = '#eaeaf2'

fig = plt.figure(figsize=(10, 3.5))
gs = gridspec.GridSpec(1, 2)
gs.update(wspace=0.3, hspace=0.5)
gs.update(left=0.09, right=0.97, bottom=0.2, top=0.78)

c = 4
LABEL_FONT_SIZE = 21 +c
TICK_LABEL_FONT_SIZE = 16 +c
LEGEND_FONT_SIZE = 21
# LEGEND_LINE_SIZE = 4
TITLE_PAD = 6
LINEWIDTH = 2.5

TITLE_FONT_SIZE = 21

x_LABEL_FONT_SIZE = 21 
y_LABEL_FONT_SIZE =  21 


MARKER_SIZE = 13
# MARK_EVERY =  [1,3,5,7,9,11,13,15,17,19]
MARK_EVERY =  np.arange(5,100,10,dtype=int).tolist()
MARK_EVERY2 =  np.arange(5,290,30,dtype=int).tolist()
# MARK_EVERY =  []
color1 = '#ad190a'
color2 = '#024272'

# import seaborn as sns
# sns.set()
# from matplotlib import rc
# rc('text', usetex=True)
# ------------------------------------------
#  different dataset

# save_name = 'acc_epochs_cifar80n.pdf'
# data_root_dir = r'./cifar80no'

# cifar100_noise0.5
save_name = 'precision_kkvsk.pdf'


# save_name = 'acc_epochs_cifar100n.pdf'
# data_root_dir = r'./cifar100nc'

# ------------------------------------------


title_list = [
    'Symmetric-20', 
    'Symmetric-50', 
    'Symmetric-80', 
    'Asymmetric-20',
    'Asymmetric-40'
]

# color = {
#     'Standard': 'orange',#'purple',
#     'Decoupling': 'lightskyblue',#'maroon',#'dimgray',
#     'Co-teaching': 'green',
#     'Co-teaching+': 'maroon',
#     'JoCoR': 'grey',
#     'Peer-learning':'blue',
#     'Ours': 'red'
# }

# color = {
#     'Standard': '#922793',#'purple',
#     'Decoupling': 'grey',#'maroon',#'dimgray',
#     'Co-teaching': 'green',
#     'Co-teaching+': 'maroon',
#     'JoCoR': 'orange',
#     'Peer-learning':'#1f77b4',
#     'Ours': '#ff0505'
# }
color1 = '#4c72b0'

method = ['Standard',
          'Decoupling',
          'Co-teaching', 
          'Co-teaching+',
          'JoCoR',
          'Peer-learning',
          'Ours']



# color1 = '#ad190a'

# color2 = '#030303'
# color3 = '#024272'
# color4 = '#878787'

# color1 = '#ff2727'
# color2 = '#1f77b4'
# color3 = '#ff7f0e'
# color4 = '#75c175'

color1 = '#1f77b4'
color2 = '#ff7f0e'
color3 = '#ff4e4e'
# color4 = '#9abde4'
color4 = '#9abde4'
color5 = '#929497'

color3 = '#b52330'
color4 =  '#0f5ba6'
color3 = '#2f9796'
color4 =  '#9666bc'
color5 = '#8a554d'
# color5 =  '#0a346e'

color3 = '#ff4e4e'
color4 = '#92bee1'

color3 = '#de3933'
color4 = '#35899f'
color5 = '#7d7a8f'

color1 = 'green'
color3 = 'red'
color4 = '#1f77b4'
color5 = '#636363'
color5 = '#966a20'
color5 = 'purple'
color5 = '#6e526a'
color5 = '#8f4cbf'
color5 = '#6D758D'
# color5 = '#795548'
# color5 = '#607d8b'

# color6 = color5
# color5 = color4
# color4 = color6


class MaxMeter(object):
    def __init__(self):
        self.val = 0
    def reset(self):
        self.val = 0
    def update(self, val):
        self.val = max(self.val, val)
        
class MinMeter(object):
    def __init__(self):
        self.val = float("inf")
    def reset(self):
        self.val = float("inf")
    def update(self, val):
        self.val = min(self.val, val)
        
def max_min_acc(m):
    total_max = MaxMeter()
    total_min = MinMeter()
    for i in m:
        total_max.update(i)
        total_min.update(i)
    return total_max.val, total_min.val
        

def get_acckkvsk(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    # epoch_np = np.zeros(1)
    kkacc = []
    kacc = []
    # epoch_np = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    for idx in range(10,len(lines)): # len(lines)
        line = lines[idx].strip()
        acc1, acc2, = line.split(' | ')[10], line.split(' | ')[11]
        kkacc.append( float(acc1.split(': ')[1]))
        kacc.append( float(acc2.split(': ')[1]))
    return  kkacc, kacc


def get_accsplit(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    # epoch_np = np.zeros(1)
    cleanacc = []
    idacc = []
    oodacc = []
    # epoch_np = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    for idx in range(10,len(lines)): # len(lines)
        line = lines[idx].strip()
        acc1, acc2, acc3 = line.split(' | ')[7], line.split(' | ')[9], line.split(' | ')[12] 
        cleanacc.append( float(acc1.split(': ')[1]))
        idacc.append( float(acc2.split(': ')[1]))
        oodacc.append( float(acc3.split(': ')[1]))

    return  cleanacc, idacc, oodacc

# w_pc_err = np.array([0.09, 0.14, 0.14,0.09,0.1,0.1,0.23,0.17])

def getpathkkvsk():
    current_root = os.getcwd()
    for leaf_dir in os.listdir(current_root):
        if leaf_dir.startswith('k1_250'):
            return leaf_dir+'/log.txt'
        
def getpathsplit():
    current_root = os.getcwd()
    for leaf_dir in os.listdir(current_root):
        if leaf_dir.startswith('split'):
            return leaf_dir+'/log.txt'

if __name__ == '__main__':            
    
    kkacc, kacc,  = get_acckkvsk(getpathkkvsk())
    # ------------------------------------------------------------------------------------
    # fig 1 
    ax1 = plt.subplot(gs[0, 0]) 
    x = np.arange(10,200)    
   
    ax1.plot( x, gaussian_filter1d(kacc, sigma=2), color=color1, linestyle='-', linewidth = LINEWIDTH, label="Ranking", ) # marker='s', markersize=MARKER_SIZE, clip_on=False ,markevery=np.arange(20,200,30,dtype=int).tolist()) #,markevery=MARK_EVERY)
    # ax1.plot( x[1:51], cpacc[:50], color=color2, linestyle='-', linewidth = LINEWIDTH, label="clean",marker='o', markersize=MARKER_SIZE, clip_on=False ,markevery=np.arange(5,50,10,dtype=int).tolist())
    ax1.plot( x, gaussian_filter1d(kkacc, sigma=2), color=color2, linestyle='-', linewidth = LINEWIDTH, label="Re-ranking",) # marker='D', markersize=MARKER_SIZE, clip_on=False ,markevery=np.arange(20,200,30,dtype=int).tolist())
    # ax1.plot(x, y1,color=color2, linestyle='--', linewidth = LINEWIDTH, label="FGCrossNet",marker='o',  markersize=MARKER_SIZE, clip_on=False,markevery=MARK_EVERY)
    ax1.set_xlim(10-0.05, 200+0.05)
    ax1.set_ylim(20, 50)
    ax1.set_xticks( [0, 50,  100,  150,  200 ]) 
    # ax1.set_xticks( [0, 40,  80, 120,  160,  200 ]) 
    # ax1.set_yticks( [ 20,30, 40, 50] ) 
    ax1.set_xlabel('Epochs' , fontsize=x_LABEL_FONT_SIZE)
    ax1.set_ylabel('Relabeling ACC', fontsize=y_LABEL_FONT_SIZE)
    # ax1.tick_params(direction='in', pad=4, labelsize=TICK_LABEL_FONT_SIZE)
    ax1.tick_params( labelsize=TICK_LABEL_FONT_SIZE, direction='in', pad=4,) # which='major',length=0,
    # ax1.legend(loc=4,edgecolor='black', fontsize=LEGEND_FONT_SIZE)
    # ax1.set_title('Image to all other modalities',fontdict={'fontsize': LABEL_FONT_SIZE}, pad=TITLE_PAD)
    
    # ax.plot(w_pc, w_pc_val, '-', linewidth=LINE_WIDTH, color=color1, marker='o', markeredgecolor = '', markersize=)
    # # ax.errorbar(w_pc, w_pc_val, yerr= w_pc_err,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)

    # ax1.grid(True)
    # ax1.grid(color='w', linestyle='-', linewidth=2 ,alpha=1, )
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)
    # ax1.spines['left'].set_visible(False)
    
    # ax.set_title(title_list[0], pad=TITLE_PAD, fontsize=TITLE_FONT_SIZE, y= 1.02)

    # ax1.legend(fontsize=LEGEND_FONT_SIZE, loc=4,  handlelength=1.6, columnspacing=1.2)
    ax1.legend(fontsize=LEGEND_FONT_SIZE, loc='upper center', ncol=2, bbox_to_anchor=(0.47, 1.43), handlelength=1.2, columnspacing=1.2)
    # ax.legend([ 'ResNet-50', 'Decoupling', 'Co-teaching', 'JoCoR', 'Ours'], loc='lower right', fontsize=LEGEND_FONT_SIZE)
    # ------------------------------------------------------------------------------------

    # # ------------------------------------------------------------------------------------
    # # fig 2
    cleanacc, idacc, oodacc = get_accsplit(getpathsplit())
    x2 = np.arange(100+100+1)    
    ax2 = plt.subplot(gs[0, 1])

    model=make_interp_spline(np.arange(11,100+100+1) , idacc)
    xs=np.linspace(11,200,40)
    ys=model(xs)
    ax2.plot( x2[11:], gaussian_filter1d(cleanacc, sigma=1), color=color3, linestyle='-', linewidth = LINEWIDTH, label="Clean",)# marker='s', markersize=MARKER_SIZE, clip_on=False ,markevery=np.arange(10,100,15,dtype=int).tolist()) #,markevery=MARK_EVERY)
    ax2.plot( xs, ys, color=color4, linestyle='-', linewidth = LINEWIDTH, label="ID",)
    # ax2.plot( x2[11:], gaussian_filter1d(idacc, sigma=3), color=color2, linestyle='-', linewidth = LINEWIDTH, label="id",)# marker='o', markersize=MARKER_SIZE, clip_on=False ,markevery=np.arange(10,100,15,dtype=int).tolist())
    ax2.plot( x2[11:], gaussian_filter1d(oodacc, sigma=1),  color=color5, linestyle='-', linewidth = LINEWIDTH, label="OOD",)#  marker='D', markersize=MARKER_SIZE, clip_on=False ,markevery=np.arange(10,100,15,dtype=int).tolist())
    # ax2.plot(x, y1,color=color2, linestyle='--', linewidth = LINEWIDTH, label="FGCrossNet",marker='o',  markersize=MARKER_SIZE, clip_on=False,markevery=MARK_EVERY)
    ax2.set_xlim(0-0.05, 200+0.05)
    ax2.set_ylim(30, 100)
    ax2.set_xticks( [0, 50, 100, 150, 200 ]) 
    # ax2.set_yticks( [  0, 25, 50, 75, 100] ) 
    # plt.xticks([0,100,200,296] ,[0,100,4000,4096],) 
    
    ax2.set_xlabel('Epochs' , fontsize=x_LABEL_FONT_SIZE)
    ax2.set_ylabel('Division ACC', fontsize=y_LABEL_FONT_SIZE)
    # ax2.tick_params(direction='in', pad=4, labelsize=TICK_LABEL_FONT_SIZE)
    ax2.tick_params( labelsize=TICK_LABEL_FONT_SIZE, direction='in', pad=4,)  #  which='major',length=0,
    # ax2.legend(loc=4,edgecolor='black', fontsize=LEGEND_FONT_SIZE)
    # ax2.set_title('Image to all other modalities',fontdict={'fontsize': LABEL_FONT_SIZE}, pad=TITLE_PAD)
    
    # ax.plot(w_pc, w_pc_val, '-', linewidth=LINE_WIDTH, color=color1, marker='o', markeredgecolor = '', markersize=)
    # # ax.errorbar(w_pc, w_pc_val, yerr= w_pc_err,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)

    # ax2.grid(True)
    # ax2.grid(color='w', linestyle='-', linewidth=2 ,alpha=1, )
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # # ax.set_title(title_list[0], pad=TITLE_PAD, fontsize=TITLE_FONT_SIZE, y= 1.02)
    # ax2.legend(fontsize=LEGEND_FONT_SIZE, loc=4 , handlelength=1.6, columnspacing=1.2)
    ax2.legend(fontsize=LEGEND_FONT_SIZE, loc='upper center', ncol=3, bbox_to_anchor=(0.47, 1.43), handlelength=1.2, columnspacing=1.2)
    
    # # ax.legend([ 'ResNet-50', 'Decoupling', 'Co-teaching', 'JoCoR', 'Ours'], loc='lower right', fontsize=LEGEND_FONT_SIZE)
    # # ------------------------------------------------------------------------------------
    # # # ------------------------------------------------------------------------------------
    # # # fig 3
    # ax3 = plt.subplot(gs[0, 2])
    # w_re = [0, 0.01, 0.05, 0.1, 0.5, 1]
    # w_re_val = [57.39, 57.87, 57.75, 57.81, 54.51, 51.98]

   
    # ax3.plot( [0, 1, 2, 3, 4, 5], w_re_val,color=color1, linestyle='-', linewidth = LINEWIDTH, label="Ours",marker='v', markersize=MARKER_SIZE, clip_on=False ) #,markevery=MARK_EVERY)
    # # ax3.plot(x, y1,color=color2, linestyle='--', linewidth = LINEWIDTH, label="FGCrossNet",marker='o',  markersize=MARKER_SIZE, clip_on=False,markevery=MARK_EVERY)
    # ax3.set_xlim(0-0.25, 5+0.25)
    # # ax3.set_ylim(minre, maxre)
    # plt.xticks( [0, 1, 2, 3, 4, 5], [0, 0.01, 0.05, 0.1, 0.5, 1]) 
    # ax3.set_yticks( [ 51,  53,  55,  57, 59,  ] ) 
    # ax3.set_xlabel(r'$w_{re}$' , fontsize=x_LABEL_FONT_SIZE)
    # ax3.set_ylabel('Test Accuracy', fontsize=y_LABEL_FONT_SIZE)
    # # ax3.tick_params(direction='in', pad=4, labelsize=TICK_LABEL_FONT_SIZE)
    # ax3.tick_params(which='major',length=0, labelsize=TICK_LABEL_FONT_SIZE)
    # # ax3.legend(loc=4,edgecolor='black', fontsize=LEGEND_FONT_SIZE)
    # # ax3.set_title('Image to all other modalities',fontdict={'fontsize': LABEL_FONT_SIZE}, pad=TITLE_PAD)
    
    # # ax.plot(w_re, w_re_val, '-', linewidth=LINE_WIDTH, color=color1, marker='o', markeredgecolor = '', markersize=)
    # # # ax.errorbar(w_re, w_re_val, yerr= w_pc_err,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)

    # ax3.grid(True)
    # ax3.grid(color='w', linestyle='-', linewidth=2 ,alpha=1, )
    # ax3.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['bottom'].set_visible(False)
    # ax3.spines['left'].set_visible(False)
    
    # # # # ------------------------------------------------------------------------------------
    # # # fig 4
    # ax4 = plt.subplot(gs[1, 0])
    # k1 = [ 5, 10, 15, 20, 50, 100]
    # k1_val = [57.39, 57.73, 57.87, 57.46, 57.79, 57.46]

   
    # ax4.plot( [0, 1, 2, 3, 4, 5], k1_val,color=color1, linestyle='-', linewidth = LINEWIDTH, label="Ours",marker='v', markersize=MARKER_SIZE, clip_on=False ) #,markevery=MARK_EVERY)
    # # ax4.plot(x, y1,color=color2, linestyle='--', linewidth = LINEWIDTH, label="FGCrossNet",marker='o',  markersize=MARKER_SIZE, clip_on=False,markevery=MARK_EVERY)
    # ax4.set_xlim(0-0.25, 5+0.25)
    # # ax4.set_ylim(minre, maxre)
    # plt.xticks( [0, 1, 2, 3, 4, 5], [5, 10, 15, 20, 50, 100]) 
    # # ax4.set_yticks( [ 51,  53,  55,  57, 59,  ] ) 
    # ax4.set_yticks( [ 55,  56,  57,  58, 59,  ] ) 
    # ax4.set_xlabel(r'$k_{1}$' , fontsize=x_LABEL_FONT_SIZE - 2 )
    # ax4.set_ylabel('Test Accuracy', fontsize=y_LABEL_FONT_SIZE)
    # # ax4.tick_params(direction='in', pad=4, labelsize=TICK_LABEL_FONT_SIZE)
    # ax4.tick_params(which='major',length=0, labelsize=TICK_LABEL_FONT_SIZE)
    # # ax4.legend(loc=4,edgecolor='black', fontsize=LEGEND_FONT_SIZE)
    # # ax4.set_title('Image to all other modalities',fontdict={'fontsize': LABEL_FONT_SIZE}, pad=TITLE_PAD)
    
    # # ax.plot(k1, k1_val, '-', linewidth=LINE_WIDTH, color=color1, marker='o', markeredgecolor = '', markersize=)
    # # # ax.errorbar(k1, k1_val, yerr= w_pc_err,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)

    # ax4.grid(True)
    # ax4.grid(color='w', linestyle='-', linewidth=2 ,alpha=1, )
    # ax4.spines['top'].set_visible(False)
    # ax4.spines['right'].set_visible(False)
    # ax4.spines['bottom'].set_visible(False)
    # ax4.spines['left'].set_visible(False)
    # # # # ------------------------------------------------------------------------------------
    # # # fig 5
    # ax5 = plt.subplot(gs[1, 1])
    # sigma = [ 0.05, 0.10, 0.15, 0.20, 0.25, 0.3]
    # sigma_val = [57.75, 57.92, 57.87, 58.07, 57.83, 57.64]

   
    # ax5.plot( [0, 1, 2, 3, 4, 5], sigma_val,color=color1, linestyle='-', linewidth = LINEWIDTH, label="Ours",marker='v', markersize=MARKER_SIZE, clip_on=False ) #,markevery=MARK_EVERY)
    # # ax5.plot(x, y1,color=color2, linestyle='--', linewidth = LINEWIDTH, label="FGCrossNet",marker='o',  markersize=MARKER_SIZE, clip_on=False,markevery=MARK_EVERY)
    # ax5.set_xlim(0-0.25, 5+0.25)
    # # ax5.set_ylim(minre, maxre)
    # plt.xticks( [0, 1, 2, 3, 4, 5], sigma) 
    # # ax5.set_yticks( [ 51,  53,  55,  57, 59,  ] ) 
    # ax5.set_yticks( [ 55,  56,  57,  58, 59,  ] ) 
    # ax5.set_xlabel(r'$\sigma$' , fontsize=x_LABEL_FONT_SIZE)
    # ax5.set_ylabel('Test Accuracy', fontsize=y_LABEL_FONT_SIZE)
    # # ax5.tick_params(direction='in', pad=4, labelsize=TICK_LABEL_FONT_SIZE)
    # ax5.tick_params(which='major',length=0, labelsize=TICK_LABEL_FONT_SIZE)
    # # ax5.legend(loc=4,edgecolor='black', fontsize=LEGEND_FONT_SIZE)
    # # ax5.set_title('Image to all other modalities',fontdict={'fontsize': LABEL_FONT_SIZE}, pad=TITLE_PAD)
    
    # # ax.plot(sigma, sigma_val, '-', linewidth=LINE_WIDTH, color=color1, marker='o', markeredgecolor = '', markersize=)
    # # # ax.errorbar(sigma, sigma_val, yerr= w_pc_err,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)

    # ax5.grid(True)
    # ax5.grid(color='w', linestyle='-', linewidth=2 ,alpha=1, )
    # ax5.spines['top'].set_visible(False)
    # ax5.spines['right'].set_visible(False)
    # ax5.spines['bottom'].set_visible(False)
    # ax5.spines['left'].set_visible(False)
    # # ------------------------------------------------------------------------------------

    # # # fig 6
    # ax6 = plt.subplot(gs[1, 2])
    # epsilon = [ 0, 0.1, 0.3, 0.5, 0.7, 0.9]
    # epsilon_val = [58.73, 59.49, 59.45, 58.47, 58.91, 57.87, 56.8, 54.45, 52.82, 45.77, ]

   
    # ax6.plot( [0, 1, 2, 3, 4, 5, 6 ,7 , 8, 9], epsilon_val,color=color1, linestyle='-', linewidth = LINEWIDTH, label="Ours",marker='v', markersize=MARKER_SIZE, clip_on=False ) #,markevery=MARK_EVERY)
    # # ax6.plot(x, y1,color=color2, linestyle='--', linewidth = LINEWIDTH, label="FGCrossNet",marker='o',  markersize=MARKER_SIZE, clip_on=False,markevery=MARK_EVERY)
    # ax6.set_xlim(0-0.25, 9+0.25)
    # # ax6.set_ylim(minre, maxre)
    # plt.xticks( [0, 1, 3,  5, 7 , 9 ], epsilon) 
    # ax6.set_yticks( [ 40,  45,  50,  55, 60 ,  ] ) 
    # ax6.set_xlabel(r'$\epsilon$' , fontsize=x_LABEL_FONT_SIZE)
    # ax6.set_ylabel('Test Accuracy', fontsize=y_LABEL_FONT_SIZE)
    # # ax6.tick_params(direction='in', pad=4, labelsize=TICK_LABEL_FONT_SIZE)
    # ax6.tick_params(which='major',length=0, labelsize=TICK_LABEL_FONT_SIZE)
    # # ax6.legend(loc=4,edgecolor='black', fontsize=LEGEND_FONT_SIZE)
    # # ax6.set_title('Image to all other modalities',fontdict={'fontsize': LABEL_FONT_SIZE}, pad=TITLE_PAD)
    
    # # ax.plot(epsilon, epsilon_val, '-', linewidth=LINE_WIDTH, color=color1, marker='o', markeredgecolor = '', markersize=)
    # # # ax.errorbar(epsilon, epsilon_val, yerr= w_pc_err,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)

    # ax6.grid(True)
    # ax6.grid(color='w', linestyle='-', linewidth=2 ,alpha=1, )
    # ax6.spines['top'].set_visible(False)
    # ax6.spines['right'].set_visible(False)
    # ax6.spines['bottom'].set_visible(False)
    # ax6.spines['left'].set_visible(False)
    # # ------------------------------------------------------------------------------------
    # # plt.legend(bbox_to_anchor=(0., 0.8, 5, .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    # # handles, labels = ax1.get_legend_handles_labels()
    # # fig.legend(handles, labels,  loc='upper center', ncol=7, mode="expand", borderaxespad=0., fontsize= LEGEND_FONT_SIZE, handlelength=LEGEND_LINE_SIZE)
    
    plt.savefig(save_name, format='pdf', dpi=1000)
    plt.show()


# -*- coding: utf-8 -*-






