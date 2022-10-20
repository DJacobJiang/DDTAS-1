import torch
import numpy as np
from utils.serialization import load_checkpoint
from evaluations import Recall_at_ks, pairwise_similarity, NMI
from Model2Feature import Model2Feature as M2F
import os.path as osp
from utils.plot import plot_loss, plot_rank, heat, hist, mined_num

import torch
import numpy as np
from utils.serialization import load_checkpoint
from evaluations import Recall_at_ks, pairwise_similarity
import os.path as osp
# from utils.plot import plot_loss, plot_rank

from torch.backends import cudnn
from evaluations import extract_features
import models
import DataSet
from utils.serialization import load_checkpoint
cudnn.benchmark = True


def Model2Feature(data, net, checkpoint, dim=512, width=224, root=None, nThreads=16, batch_size=100, pool_feature=False, **kargs):

    model = models.create(net, data=data,dim=dim, pretrained=True)#测试
    # model.eval()
    # resume = checkpoint
    #
    # model_dict = model.state_dict()
    # pretrained_dict = resume
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print("matched parameters: %d/%d" % (len(pretrained_dict), len(model_dict)))
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    # model.load_state_dict(resume['state_dict'])
    model = torch.nn.DataParallel(model).cuda()
    # model.eval()
    data = DataSet.create(data, net = net, width=width, origin_width = 256 ,root=root)
    data_loader = torch.utils.data.DataLoader(data.gallery, batch_size=batch_size,
            shuffle=False, drop_last=False, pin_memory=True,
            num_workers=nThreads)
    features, labels = extract_features(model, data_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
    gallery_feature, gallery_labels = query_feature, query_labels = features, labels
    # print('ql',query_labels)
    return gallery_feature, gallery_labels, query_feature, query_labels



def ckptest_I(args):

    if args.resume is None:
        print('*' * 10, 'IMGN', '*' * 10)
        checkpoint = load_checkpoint('/home/jxr/proj/DDTAS/models/bn_inception-52deb4733.pth')
        # checkpoint = load_checkpoint('/home/jxr/proj/DDTAS/models/resnet50-19c8e357.pth')
        # checkpoint = load_checkpoint(args.resume)
        epoch = 0
        Rank1 = []

        recall_list = np.zeros((args.epochs, 1))
        gallery_feature, gallery_labels, query_feature, query_labels = \
            Model2Feature(data=args.data, root=args.data_root, width=args.width, net=args.net,
                          checkpoint=checkpoint,
                          dim=args.dim, batch_size=args.batch_size_t, nThreads=args.nThreads,
                          pool_feature=args.pool_feature)


        sim_mat = pairwise_similarity(query_feature, gallery_feature)
        if args.gallery_eq_query is True:
            sim_mat = sim_mat - torch.eye(sim_mat.size(0))

        hist(simmat=sim_mat, log_dir=args.save_dir, dataset=args.data, epoch=epoch, query_lbs=query_labels,
             gallery_lbs=gallery_labels)
        heat(x=sim_mat, log_dir=args.save_dir, dataset=args.data, epoch=epoch)
        recall_ks = Recall_at_ks(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)

        Rank1.append(recall_ks[0])

        result = '  '.join(['%.4f' % k for k in recall_ks])
        recall_list[epoch] = recall_ks[0]
        print(recall_list[epoch][0])
        print(np.argmax(recall_list), np.max(recall_list))
        print('Epoch-%d' % epoch, result)
        print('*' * 10, 'IMGN', '*' * 10)
    else:
        print('*' * 10, 'CKP', '*' * 10)
        # checkpoint = load_checkpoint('/home/jxr/proj/Deep_Metric-master/models/ckp_ep1405_Triog_m003_7166.pth.tar')
        checkpoint = load_checkpoint(args.resume)
        epoch = checkpoint['epoch']
        Rank1 = []

        recall_list = np.zeros((args.epochs, 1))
        gallery_feature, gallery_labels, query_feature, query_labels = \
            M2F(data=args.data, root=args.data_root, width=args.width, net=args.net_t,
                          checkpoint=checkpoint,
                          dim=args.dim_t, batch_size=args.batch_size_t, nThreads=args.nThreads,
                          pool_feature=args.pool_feature)

        sim_mat = pairwise_similarity(query_feature, gallery_feature)
        if args.gallery_eq_query is True:
            sim_mat = sim_mat - torch.eye(sim_mat.size(0))

        recall_ks = Recall_at_ks(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)

        Rank1.append(recall_ks[0])

        result = '  '.join(['%.4f' % k for k in recall_ks])
        recall_list[epoch] = recall_ks[0]
        print(recall_list[epoch][0])
        print(np.argmax(recall_list), np.max(recall_list))
        print('Epoch-%d' % epoch, result)
        print('*' * 10, 'CKP', '*' * 10)


    return 0

def ckptest(args):
    print('*' * 10, 'CKP', '*' * 10)
    # checkpoint = load_checkpoint('/home/jxr/proj/Deep_Metric-master/models/ckp_ep1405_Triog_m003_7166.pth.tar')
    checkpoint = load_checkpoint(args.resume)
    epoch = checkpoint['epoch']
    Rank1 = []

    recall_list = np.zeros((args.epochs, 1))
    gallery_feature, gallery_labels, query_feature, query_labels = \
        Model2Feature(data=args.data, root=args.data_root, width=args.width, net=args.net_t,
                      checkpoint=checkpoint,
                      dim=args.dim_t, batch_size=args.batch_size_t, nThreads=args.nThreads,
                      pool_feature=args.pool_feature)

    sim_mat = pairwise_similarity(query_feature, gallery_feature)
    if args.gallery_eq_query is True:
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))

    recall_ks = Recall_at_ks(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)

    Rank1.append(recall_ks[0])

    result = '  '.join(['%.4f' % k for k in recall_ks])
    recall_list[epoch] = recall_ks[0]
    print(recall_list[epoch][0])
    print(np.argmax(recall_list), np.max(recall_list))
    print('Epoch-%d' % epoch, result)
    print('*' * 10, 'CKP', '*' * 10)

    return 0



def test(args,epoch,use_gpu,model,save_checkpoint, Rank1  ,recall_list, losses_p,NMI_list, ap_mined_it_epoch,an_mined_it_epoch,ap_mined_epoch,an_mined_epoch, anapre, anAPre):
    # if epoch == 1:
    #     optimizer.param_groups[0]['lr_mul'] = 0.1
    if (epoch + 1) % args.save_step == 0 or epoch == 0:
        if use_gpu:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        #
        save_checkpoint({
            'state_dict': state_dict,
            'epoch': (epoch + 1),
        }, is_best=False, fpath=osp.join(args.save_dir,'ckp','ckp_ep' + str(epoch + 1) + '.pth.tar'))

        if epoch == 0:
            pass
        else:
            # test
            checkpoint = load_checkpoint(osp.join(args.save_dir,'ckp','ckp_ep' + str(epoch + 1) + '.pth.tar'))
            # print(args.pool_feature)
            epoch = checkpoint['epoch']

            gallery_feature, gallery_labels, query_feature, query_labels = \
                M2F(data=args.data, root=args.data_root, width=args.width, net=args.net_t,
                              checkpoint=checkpoint,
                              dim=args.dim, batch_size=args.batch_size_t, nThreads=args.nThreads,
                              pool_feature=args.pool_feature)
            nmi = NMI(query_feature, query_labels, n_cluster=100)
            print("NMI", nmi)
            # print('ql',query_labels)
            sim_mat = pairwise_similarity(query_feature, gallery_feature)
            if (epoch) % (args.save_step * 1) == 0 or epoch == 0:
                heat(x=sim_mat, log_dir=args.save_dir, dataset=args.data, epoch=epoch)

            if args.gallery_eq_query is True:
                sim_mat = sim_mat - torch.eye(sim_mat.size(0))

            recall_ks = Recall_at_ks(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)
            Rank1.append(recall_ks[0]*100)
            NMI_list.append(nmi)

            recall_list[epoch] = recall_ks[0]
            if (epoch) % (args.save_step * 1) == 0 or epoch == 0:
                hist(simmat=sim_mat, log_dir=args.save_dir, dataset=args.data, epoch=epoch, query_lbs=query_labels,
                     gallery_lbs=gallery_labels)

            if (epoch) % (args.save_step * 1) == 0:
                mined_num(epoch=epoch, log_dir=args.save_dir, dataset=args.data, ap_mined_it_epoch=ap_mined_it_epoch,an_mined_it_epoch=an_mined_it_epoch,ap_mined_epoch=ap_mined_epoch,
                          an_mined_epoch=an_mined_epoch, ptl=anapre, pog=anAPre)
            # hist_t(x1=ap_sim_t.view(-1), x2=an_sim_t.view(-1), log_dir=args.save_dir, dataset=args.data, epoch=epoch)
            # hist_tt(x1=ap_sim.view(-1), x2=an_sim.view(-1), x1_t=ap_sim_t.view(-1), x2_t=an_sim_t.view(-1), log_dir=args.save_dir, dataset=args.data, epoch=epoch)

            print(recall_list[epoch][0])
            print(np.argmax(recall_list), np.max(recall_list))
            print("NMIM",np.argmax(NMI_list), np.max(NMI_list))

            result = '  '.join(['%.4f' % k for k in recall_ks])
            # print('typeresult',type(result))
            print('Epoch-%d' % epoch, result)

            plot_rank(rank=Rank1, log_dir=args.save_dir, dataset=args.data, step=args.save_step, epoch=epoch)
            plot_loss(epoch=epoch,loss=losses_p, log_dir=args.save_dir, dataset=args.data, step=args.save_step, is_log=False)

            # heat(x=sim_mat, log_dir=args.save_dir, dataset=args.data, epoch=epoch)
    return Rank1
            # plot_margin(margin=margint_p,log_dir=args.save_dir, dataset=args.data)


