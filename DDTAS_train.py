# coding=utf-8
from __future__ import absolute_import, print_function
import time
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import models
import losses
from utils import FastRandomIdentitySampler,RandomIdentitySampler, mkdir_if_missing, logging, display, plot_loss, plot_rank, model_loader, args
from utils.inf_generator import inf_generator
from utils.serialization import save_checkpoint, load_checkpoint
from utils.list_ini import list_ini, list_append
from utils.ckptest import ckptest, test, ckptest_I

# from trainer import train
from trainer_DDTAS import train_DDTAS
from trainer import train

from utils import orth_reg

import DataSet
import numpy as np
import os.path as osp

cudnn.benchmark = True
# from __future__ import absolute_import, print_function
# import argparse
from Model2Feature import Model2Feature
from evaluations import Recall_at_ks, pairwise_similarity
# from utils.serialization import load_checkpoint
import torch
import ast
from fvcore.nn import FlopCountAnalysis, parameter_count_table
# paint
import matplotlib
from matplotlib.ticker import MultipleLocator

matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

use_gpu = True

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def main(args):
    # s_ = time.time()
    print(args.MMSI)
    save_dir = args.save_dir
    mkdir_if_missing(save_dir)

    sys.stdout = logging.Logger(os.path.join(save_dir, 'log.txt'))
    display(args)
    start = 0

    model, start, optimizer, scheduler = model_loader(args, start)
    # print('MPN',parameter_count_table(model))
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    # print('initial model is save at %s' % save_dir)
    if args.OTG == 1:
        #In OTG method, the same DML loss has three functional forms:
        # 1.S--Select Pair (in outer loop)
        criterion_i = losses.create('S' + args.loss, margin=args.margin, alpha=args.alpha, base=args.loss_base).cuda()
        # 2.G--Generated Gradient Weight (in outer loop)
        criterion_m = losses.create('G' + args.loss, margin=args.margin, alpha=args.alpha, base=args.loss_base).cuda()
        # 3.W--calculate the pair-based DML loss by Weight
        criterion = losses.create('W' + args.loss, margin=args.margin, alpha=args.alpha, base=args.loss_base).cuda()
    else:
        # Original DML loss
        criterion = losses.create(args.loss, alpha=args.alpha, base=args.loss_base).cuda()

    data = DataSet.create(args.data, net = args.net,ratio=args.ratio, width=args.width, origin_width=args.origin_width,
                          root=args.data_root)

    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=args.batch_size,
        sampler=FastRandomIdentitySampler(data.train, num_instances=args.num_instances),
        drop_last=True, pin_memory=True, num_workers=args.nThreads)

    if args.OTG == 1:
        valid_loader = torch.utils.data.DataLoader(
            data.train, batch_size=args.val_batch_size,
            sampler=FastRandomIdentitySampler(data.train, num_instances=args.val_num_instances),
            drop_last=True, pin_memory=True, num_workers=args.nThreads)
        valid_loader = inf_generator(valid_loader)

    wmean, losses_p, meta_losses_p, wmax_p, margint_p, wmean_p, wmin_p, Rank1, recall_list, NMI_list = list_ini(
        args)
    Glob = {}


    # Test the initial checkpoint Recall@K
    # ckptest_I(args)
    it_100 = []
    it_100_g = []
    ap_mined_epoch = []
    an_mined_epoch = []
    ap_mined_it_epoch = []
    an_mined_it_epoch = []
    anaprte = []
    anaproe = []
    for epoch in range(start, args.epochs):
        # model.train()

        if epoch <= 1:
            optimizer.param_groups[0]['lr_mul'] = 0.1
        if args.OTG == 1:
            # print('aaaaaaaaaaaaammsiaaaaaaaaaaaaaaaaaaaaa')
            losse, Glob_now ,it_100,it_100_g = train_OTG(epoch=epoch, model=model, criterion=criterion,
                                    criterion_i=criterion_i,
                                    criterion_m=criterion_m,
                                    optimizer=optimizer,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    margin=args.margin,
                                    Glob=Glob,
                                    args=args,
                                    it_100 = it_100, it_100_g = it_100_g)
            Glob = Glob_now
            losse = losse.avg
            losses_p.append(losse)
            if args.net == 'ResNet50_P':
                scheduler.step()

        else:
            losse,it_100,it_100_g, apb,anb,ap_mined_batch, an_mined_batch, anaprtb, anaprob = train(epoch=epoch, model=model, criterion=criterion,
                  optimizer=optimizer, train_loader=train_loader, args=args, it_100 = it_100, it_100_g = it_100_g)
            losse = losse.avg
            losses_p.append(losse)
            if args.net == 'ResNet50_P':
                scheduler.step()
            an_mined_it_epoch += anb
            ap_mined_it_epoch += apb
            ap_mined_epoch.append(ap_mined_batch)
            an_mined_epoch.append(an_mined_batch)
            anaprte.append(anaprtb)
            anaproe.append(anaprob)
            # print(anapre)
            # print(anapret)
            # print(len(ap_mined_epoch), len(an_mined_epoch))

        # losses_p.append(0)
        # print('epk', epoch)

        Rank1 = test(args, epoch, use_gpu, model, save_checkpoint, Rank1, recall_list, losses_p, NMI_list, ap_mined_it_epoch,an_mined_it_epoch,ap_mined_epoch,an_mined_epoch, anaprte, anaproe)



if __name__ == '__main__':
    parser =args()
    main(parser.parse_args())




