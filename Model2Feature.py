# coding=utf-8
from __future__ import absolute_import, print_function

import torch
from torch.backends import cudnn
from evaluations import extract_features
import models
import DataSet
from utils.serialization import load_checkpoint
import numpy as np
cudnn.benchmark = True


def Model2Feature(data, net, checkpoint, dim=512, width=224, root=None, nThreads=16, batch_size=100, pool_feature=False, **kargs):
    dataset_name = data
    model = models.create(net, data = dataset_name, pretrained=False, dim=dim)
    weight = checkpoint['state_dict']
    checkpoint = weight
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model = torch.nn.DataParallel(model).cuda()
    data = DataSet.create(data, net=net, width=width, root=root)
    if dataset_name in ['shop', 'jd_test']:
        gallery_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=batch_size, shuffle=False,
            drop_last=False, pin_memory=True, num_workers=nThreads)

        query_loader = torch.utils.data.DataLoader(
            data.query, batch_size=batch_size,
            shuffle=False, drop_last=False,
            pin_memory=True, num_workers=nThreads)

        gallery_feature, gallery_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        query_feature, query_labels = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)

    else:
        data_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=batch_size,
            shuffle=False, drop_last=False, pin_memory=True,
            num_workers=nThreads)
        features, labels = extract_features(model, data_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        gallery_feature, gallery_labels = query_feature, query_labels = features, labels
        # np.random.seed(200)
        # np.random.shuffle(features)
        # np.random.seed(200)
        # np.random.shuffle(labels)
        # # gallery_feature, gallery_labels = query_feature, query_labels = features, labels
        # np.random.seed(200)
        # np.random.shuffle(gallery_feature)
        # np.random.seed(200)
        # np.random.shuffle(gallery_labels)
    return gallery_feature, gallery_labels, query_feature, query_labels

