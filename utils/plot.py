import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator
from utils import mkdir_if_missing
import os
from matplotlib import  rcParams
import matplotlib.gridspec as gridspec
config = {
"font.family":'serif',
"font.size": 20,
"mathtext.fontset":'stix',
"font.serif": ['SimSun'],#SimSun
}
rcParams.update(config)
plt.rc('font',family='Times New Roman') #全局变为Times

plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


def mined_num(epoch,log_dir,dataset,ap_mined_it_epoch,an_mined_it_epoch,ap_mined_epoch,an_mined_epoch, ptl, pog):

    fgs =(12,8.5)
    fts = 50
    ftslgd = 40
    plt.rc('font', family='Times New Roman')  # 全局变为Times

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # fig, axes = plt.subplots()
    # ax = axes
    # len(ap_mined_epoch)/2
    # print(epoch)
    # plt.rc('font', family='Times New Roman')
    print('Type:',type(ap_mined_it_epoch),type(an_mined_it_epoch),type(ap_mined_epoch),type(ap_mined_epoch))
    epochs = np.arange(1, len(ap_mined_it_epoch) + 1)/6
    plt.figure(figsize=fgs, dpi=300)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)
    plt.plot(epochs, ap_mined_it_epoch, color='r', label='pos', linewidth=1)
    plt.plot(epochs, an_mined_it_epoch, color='b', label='neg', linewidth=1)
    plt.xticks(fontsize=fts)
    plt.yticks(fontsize=fts)
    plt.ylim(0, 3000)
    # plt.title('Mined Pairs Number')
    plt.xlabel("Epoch", fontsize=fts)
    plt.ylabel('mined pairs number', fontsize=fts)
    # x_major_locator = plt.MultipleLocator(100)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    x_major_locator = plt.MultipleLocator(200)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(fontsize=ftslgd)
    im_path = '%s/%s_pnum_apan_it.png' % (log_dir, dataset)
    txt_path = '%s/%s_pair.txt' % (log_dir, dataset)
    f = open(txt_path,'w')
    list = [ap_mined_it_epoch,an_mined_it_epoch,ap_mined_epoch,an_mined_epoch]
    for line in list:
        f.write(str(line)+'\n')
    f.close()
    mkdir_if_missing(os.path.dirname(im_path))
    plt.savefig(im_path, dpi=1000)
    # print("save done " + im_path)
    plt.close()

    epochs = np.arange(1, epoch + 1)
    plt.figure(figsize=fgs, dpi=300)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)
    # print('eeeeeeeeeeeeee',epochs/6,(len(ap_mined_epoch))/6+1,type((len(ap_mined_epoch))/6+1))
    plt.plot(epochs,ap_mined_epoch,color ='r',label='pos',linewidth=1)
    plt.plot(epochs, an_mined_epoch,color ='b',label='neg',linewidth=1)
    plt.xticks(fontsize=fts)
    plt.yticks(fontsize=fts)
    plt.ylim(0,3000)
    # plt.title('Mined Pairs Number')
    plt.xlabel("Epoch", fontsize=fts)
    plt.ylabel('mined pairs number', fontsize=fts)
    # x_major_locator = plt.MultipleLocator(100)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    x_major_locator = plt.MultipleLocator(200)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(fontsize=ftslgd)
    im_path = '%s/%s_pnum_apan.png' % (log_dir, dataset)
    mkdir_if_missing(os.path.dirname(im_path))
    plt.savefig(im_path, dpi=600)
    # print("save done " + im_path)
    plt.close()

    plt.figure(figsize=fgs, dpi=300)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)
    plt.plot(epochs, ap_mined_epoch, color='r', label='pos', linewidth=1)
    # plt.plot(epochs, an_mined_epoch, color='b', label='neg', linewidth=1)
    plt.ylim(0, 200)
    plt.xticks(fontsize=fts)
    plt.yticks(fontsize=fts)
    # plt.title('Mined Pairs Number')
    plt.xlabel("Epoch",fontsize=fts)
    plt.ylabel('mined pairs number',fontsize=fts)
    # x_major_locator = plt.MultipleLocator(100)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    x_major_locator = plt.MultipleLocator(200)
    y_major_locator = plt.MultipleLocator(50)
    ax = plt.gca()
    ay = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ay.yaxis.set_major_locator(y_major_locator)
    plt.legend(fontsize=ftslgd)
    im_path = '%s/%s_pnum_ap.png' % (log_dir, dataset)
    mkdir_if_missing(os.path.dirname(im_path))
    plt.savefig(im_path, dpi=600)
    # print("save done " + im_path)
    plt.close()

    epochs = np.arange(1, epoch + 1)
    plt.figure(figsize=fgs, dpi=300)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)
    # print((np.array(ptl).T).tolist())
    # print(np.array(ptl).T.tolist()[0])
    ptl = (np.array(ptl).T).tolist()
    pog = (np.array(pog).T).tolist()
    # print(ptl[])
    # plt.plot(epochs, pog[0], color = 'b',label='Asym:n_n/n_p',linewidth=1,linestyle = '--')
    plt.plot(epochs, ptl[0], color = 'b',label='AsymAT:n_n/n_p',linewidth=1)
    # plt.plot(epochs, pog[2], color = 'r',label='Asym:n_n/N_p',linewidth=1,linestyle = '--')
    plt.plot(epochs, ptl[2], color = 'r',label='AsymAT:n_n/N_p',linewidth=1)
    # plt.plot(epochs, pog[0], color='r', label='n_pos/N_pos')
    # plt.plot(epochs, ptl[0], label='T_n_neg/T_n_pos')

    # plt.xlim(-0.5, 11)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0, 20)
    plt.xticks(fontsize=fts)
    plt.yticks(fontsize=fts)
    # plt.title('Mined Pairs Number')
    # plt.title('Mined Pairs Number')
    plt.xlabel("Epoch",fontsize=fts)
    plt.ylabel('Ratio',fontsize=fts)
    x_major_locator = plt.MultipleLocator(200)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(fontsize=ftslgd)
    im_path = '%s/%s_ratio.png' % (log_dir, dataset)
    mkdir_if_missing(os.path.dirname(im_path))
    plt.savefig(im_path, dpi=600)
    plt.close()


    # plt.plot(epochs, pog[3], label='Asym:Sig(n_n/N_p)')
    # plt.plot(epochs, ptl[3], label='AsymAdaT:Sig(n_n/N_p)')
    # plt.figure(figsize=(13,5), dpi=80)
    # plt.ylim(0, 1.1)
    # plt.xticks(fontsize=40)
    # plt.yticks(fontsize=40)
    # # plt.title('Mined Pairs Number')
    # plt.xlabel("Epoch",fontsize=30)
    # plt.ylabel('Ratio',fontsize=30)
    # x_major_locator = plt.MultipleLocator(200)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.legend(fontsize=30)
    # im_path = '%s/%s_sigratio.png' % (log_dir, dataset)
    # mkdir_if_missing(os.path.dirname(im_path))
    # plt.savefig(im_path, dpi=600)
    # plt.close()

def plot_loss(epoch,loss, log_dir, dataset, step,is_log=False):
    fig, axes = plt.subplots()
    ax = axes
    # print(np.arange(1,epoch+1,1))
    loss = np.log10(loss) if is_log else np.array(loss)
    # print(loss)
    epoch_list = np.arange(1,epoch+1,1)
    ax.plot(epoch_list, loss, label='losses')
    plt.ylim(0, 1)
    ylable = "Losses(log10)" if is_log else "Losses"
    x_major_locator = plt.MultipleLocator(100)
    # x_major_locator = MultipleLocator(step)
    # ax.xaxis.set_major_locator(x_major_locator)
    ax.set_ylabel(ylable)
    ax.set_xlabel("Epoch")
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    # ax.legend()
    im_path = '%s/%s_loss.png' % (log_dir, dataset)
    mkdir_if_missing(os.path.dirname(im_path))
    plt.savefig(im_path)
    # print("save done " + im_path)
    plt.close()


def plot_rank(rank, log_dir, dataset, step, epoch):
    fig, axes = plt.subplots()
    if epoch == 0:
        epoch = np.array(1)
    else:
        epoch = np.arange(1,epoch+1,step)
    ax = axes
    # plt.ylim(40, 80)
    # print('e',epoch)
    # print('R@1',rank)
    ax.plot(epoch, rank, label='R@1(%)')
    # ax.xaxis.set_major_locator(xmajorLocator)
    ax.set_ylabel('R@1(%)')
    ax.set_xlabel("Epoch")
    x_major_locator = plt.MultipleLocator(100)
    ax.xaxis.set_major_locator(x_major_locator)
    ax = plt.gca()
    ax.legend()

    im_path = '%s/%s_R@.png' % (log_dir, dataset)
    mkdir_if_missing(os.path.dirname(im_path))
    plt.savefig(im_path, dpi=600)
    # print("save done " + im_path)
    plt.close()

def heat(x, log_dir, dataset, epoch):
    xx = np.array(x)
    for i in range(xx.shape[0]):
        t = xx[i]
        min = t.min()
        max = t.max()
        xx[i] = (t - min)/(max-min)
        # print('0',xx[i])
        xxx = np.sort(xx[i])
        # print('1',xxx)
        # print('2',xx[i])
        xx[i][xx[i] < xxx[-1000]] = 0
        xx[i] = (xx[i] - xxx[-1000]) / (xxx[-1] - xxx[-1000])
    # xx = xx[0:100 ,0:100]
    # print('xx',xx)
    plt.matshow(xx, cmap='Greens', vmin=0, vmax=1)
    # plt.colorbar()
    im_path = '%s/%s/%s/%s_%s_Heat.png' % (log_dir,'Heat','O', dataset, epoch)
    mkdir_if_missing(os.path.dirname(im_path))
    plt.savefig(im_path,dpi =200)
    plt.close()

    plt.matshow(xx[0:250 ,0:250], cmap='Greens' , vmin=0, vmax=1)
    # plt.colorbar()
    im_path = '%s/%s/%s/%s_%s_HeatL.png' % (log_dir,'Heat','L', dataset, epoch)
    mkdir_if_missing(os.path.dirname(im_path))
    plt.savefig(im_path, dpi=600)

    plt.close()

def hist(simmat,log_dir, dataset, epoch, query_lbs, gallery_lbs):
    fgs = (12, 8.5)
    fts = 50
    ftslgd = 40
    smt = simmat.cuda()

    gl = torch.tensor(np.asarray(gallery_lbs)).cuda()
    ql = torch.tensor(np.asarray(query_lbs)).cuda()
    gl_O = gl.expand(len(gl), len(gl))
    ql_O = ql.expand(len(ql), len(ql)).t()
    aplb_smt = (gl_O == ql_O).cuda().float()
    anlb_smt = (gl_O != ql_O).cuda().float()
    ap_sim = (smt * aplb_smt).view(-1)
    an_sim = (smt * anlb_smt).view(-1)
    plt.figure(figsize=fgs, dpi=300)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)
    plt.xticks(fontsize=fts)
    plt.yticks(fontsize=fts)
    plt.ylim(0, 15)
    plt.xlabel("similarity", fontsize=fts)
    # x_major_locator = plt.MultipleLocator(200)
    y_major_locator = plt.MultipleLocator(5)
    # ax = plt.gca()
    ay = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    ay.yaxis.set_major_locator(y_major_locator)
    # ap_sim_weights = np.ones_like(np.array(ap_sim.cpu())) / float(len(np.array(ap_sim.cpu())))
    # an_sim_weights = np.ones_like(np.array(an_sim.cpu())) / float(len(np.array(an_sim.cpu())))
    # plt.hist(np.array(ap_sim.cpu()),range=(0.000001, 1), alpha=0.7, density=True, bins=100, rwidth=0.9,color='green',stacked =True)
    # plt.hist(np.array(an_sim.cpu()),range=(0, 1), alpha=0.5, density=True, bins=100, rwidth=0.9,stacked =True)
    plt.hist(np.array(ap_sim.cpu()), range=(0.000001, 1), alpha=1, density=True, bins=100, rwidth=0.9,stacked=False,label='pos' , color='sandybrown')
    plt.hist(np.array(an_sim.cpu()), range=(0.000001, 1), alpha=0.6, density=True, bins=100, rwidth=0.9,stacked=False,label='neg', color='steelblue')
    plt.legend(fontsize=ftslgd)
    im_path = '%s/%s/%s_%s_Hist.png' % (log_dir, 'Hist',dataset, epoch)
    mkdir_if_missing(os.path.dirname(im_path))
    plt.savefig(im_path, dpi=600)
    print('histout')
    plt.close()