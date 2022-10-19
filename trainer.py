# coding=utf-8
from __future__ import print_function, absolute_import
import time
from torch import nn
from torch.nn import init
import numpy as np
from utils import AverageMeter, orth_reg
import  torch
from torch.autograd import Variable
from torch.backends import cudnn
# from Attention.model.attention.ExternalAttention import ExternalAttention
cudnn.benchmark = True


def train(epoch, model, criterion, optimizer, train_loader, args, it_100, it_100_g):

    # if args.warm > 0:
    #
    #     unfreeze_model_param = list(model.module.classifier.parameters()) + list(criterion.parameters())
    #
    #     if epoch == 0:
    #         for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
    #             param.requires_grad = False
    #     if epoch == args.warm:
    #         for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
    #             param.requires_grad = True

    losses = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()
    pos_sims = AverageMeter()
    neg_sims = AverageMeter()
    losses_ap = AverageMeter()
    losses_an = AverageMeter()
    ap_mined_batch = []
    an_mined_batch = []
    pt_b = []
    po_b = []
    end = time.time()
    ap_dim_list_e = list()

    freq = min(args.print_freq, len(train_loader))

    for i, data_ in enumerate(train_loader, 0):
        # print('len',data_)
        s_it = time.time()

        inputs, labels = data_
        # inputat = inputs.unsqueeze(1)
        # print('iptout', inputat.shape)
        # inputat = torch.randn(50, 512, 7, 7)


        # wrap them in Variable
        inputs = Variable(inputs).cuda() #torch.Size([batchsize, 3, 227, 227])
        labels = Variable(labels).cuda()

        optimizer.zero_grad()

        # embed_feat = model(inputs) #torch.Size([batchsize, 512])
        # embed_feat_x, embed_feat = model(inputs)  # torch.Size([batchsize, 512])

        embed_feat = model(inputs)
        # embed_feat1 = torch.randn(180, 1, 512)
        # print('iptout', embed_feat.shape,embed_feat1.shape)
        # print(ea(embed_feat1).shape)
        #
        # print('Attout', attoutput.shape)

        # print('len', np.shape(embed_feat))
        loss, inter_, dist_ap, dist_an,nap,nan,anapr,anaprog = criterion(embed_feat, labels, args.margin)
        # loss, inter_, dist_ap, dist_an, ap_dim_list = criterion(embed_feat, labels, args.margin)
        # print('lap', nap, '||', 'lan', nan)
        # print(anapr)
        ap_mined_batch.append(nap)
        an_mined_batch.append(nan)
        pt_b.append(anapr)
        po_b.append(anaprog)
        # print(pt_b)
        # print(len(anapr),len(anAPr))
        # loss, inter_, dist_ap, dist_an, ap_dim_list = criterion(embed_feat, labels, args.margin)
        # ap_dim_list_e.append(ap_dim_list)
        # loss = criterion(embed_feat, labels, num_classes=100)

        if args.orth_reg != 0:
            loss = orth_reg(net=model, loss=loss, cof=args.orth_reg)

        # loss.requires_grad_(True)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        # if args.loss == 'PAnchor':
            # print('PA')
            # torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
        #autograd操作输出矩阵，两个列表一个是loss集合，一个标记loss属于哪个anker的集合，输出后对第一个list做autograd，得到grade提出值平方和
        #然后归一化，然后控制方差。
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())
        # losses_ap.update(dist_ap.item())
        losses_ap.update(dist_ap)
        # print(losses_an)

        # losses_an.update(dist_an.item())
        losses_an.update(dist_an)
        accuracy.update(inter_)
        # pos_sims.update(dist_ap)
        # neg_sims.update(dist_an)
        e_it = time.time()
        itt = e_it-s_it
        it_100.append(itt)
        # print('itteration_time',itt)
        # if len(it_100) ==3:
        # print('smtm',it_100)
        # print('lenit', len(it_100))
        if len(it_100) == 100:
            print('100it_time',sum(it_100))
            it_100_g.append(sum(it_100))
            print('m100it', len(it_100_g),sum(it_100_g)/len(it_100_g))
            it_100 = []

        if (i + 1) % freq == 0 or (i+1) == len(train_loader):
            print('Epoch: [{0:03d}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f} \t'
                  'Accuracy {accuracy.avg:.4f} \t'
                  'Pos {pos.avg:.4f}\t'
                  'Neg {neg.avg:.4f} \t'.format
                  # (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                  #  loss=losses, accuracy=accuracy, pos=dist_ap, neg=dist_an))
                  (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                  loss=losses, accuracy=accuracy, pos=losses_ap, neg=losses_an))


        if epoch == 0 and i == 0:
            print('-- HA-HA-HA-HA-AH-AH-AH-AH --')
    # print('adle', ap_dim_list_e)
    # print(np.array(an_AP_r_b),np.mean(np.array(pt_b)[:,1]))
    apb = ap_mined_batch
    anb = an_mined_batch
    apmb =np.mean(np.array(ap_mined_batch))
    anmb =np.mean(np.array(an_mined_batch))
    lnlp= np.mean(np.array(pt_b)[:,0])
    # print(lnlp)
    sigmlnlp= np.mean(np.array(pt_b)[:,1])
    # print('[:,2]',np.array(pt_b))
    lnLp= np.mean(np.array(pt_b)[:,2])
    sigmlnLp= np.mean(np.array(pt_b)[:,3])
    lpLp= np.mean(np.array(pt_b)[:,4])

    # lnlpt = np.mean(np.array(po_b)[:,0])
    # sigmlnlpt = np.mean(np.array(po_b)[:,1])
    # lnLpt = np.mean(np.array(po_b)[:,2])
    # sigmlnLpt = np.mean(np.array(po_b)[:,3])
    # lpLpt = np.mean(np.array(po_b)[:,4])
    # ratio_t = [lnlp,sigmlnlp,lnLp,sigmlnLp,lpLp]
    # ratio_o = [lnlpt,sigmlnlpt,lnLpt,sigmlnLpt,lpLpt]
    lnlpt = 1
    sigmlnlpt = 1
    lnLpt = 1
    sigmlnLpt = 1
    lpLpt = 1
    ratio_t = [lnlp, sigmlnlp, lnLp, sigmlnLp, lpLp]
    ratio_o = [lnlpt, sigmlnlpt, lnLpt, sigmlnLpt, lpLpt]
    return (losses, it_100, it_100_g, apb, anb,apmb, anmb, ratio_t, ratio_o)
