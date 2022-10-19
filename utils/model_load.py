import models
from utils.serialization import load_checkpoint
from new_edite.dataparallel import DataParallel
from utils.setBNeval import set_bn_eval
import torch
import torch.nn as nn

def model_loader(args,start):
    if args.MMSI == 1:
        args.net = args.net + '_MetaAda'
        model = models.create(args.net, data=args.data, pretrained=True, dim=args.dim)
    else:
        # print('agdddddddd',args.dim)
        model = models.create(args.net, data=args.data, pretrained=True, dim=args.dim)

    if args.resume is None:
        if args.loss == 'MS':
            print('MS')
        #     if args.data == 'car':
        #         resume = '/home/jxr/proj/MMSI/models/MS_CAR_8559ckp.pth.tar'
        #     elif args.data == 'cub':
        #         resume = '/home/jxr/proj/MMSI/models/MS_CUB_6707ckp.pth.tar'
        #     elif args.data == 'product':
        #         resume = '/home/jxr/proj/MMSI/models/MS_SOP_784ckp.pth.tar'
        #     print('load model from {}'.format(resume))
        #     chk_pt = load_checkpoint(resume)
        #     weight = chk_pt['state_dict']
        #     start = chk_pt['epoch']
        #     start = 0
        #     checkpoint = weight  # 获取模型参数
        #     model_dict = model.state_dict()
        #     state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        #     print("matched parameters: %d/%d" % (len(state_dict), len(model_dict)))
        #
        #     model_dict.update(state_dict)
        #     model.load_state_dict(model_dict)
            model_dict = model.state_dict()
        else:
            model_dict = model.state_dict()
    else:
        # resume model
        print('load model from {}'.format(args.resume))
        chk_pt = load_checkpoint(args.resume)
        weight = chk_pt['state_dict']
        start = chk_pt['epoch']
        start = 0

        checkpoint = weight
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        print("matched parameters: %d/%d" % (len(state_dict), len(model_dict)))

        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    # model.eval()
    if args.freeze_BN is True:
        print(40 * '#', '\n BatchNorm frozen')
        model.apply(set_bn_eval)
    else:
        print(40 * '#', 'BatchNorm NOT frozen')

    # freeze BN
    # print('BN',args.net)
    # if args.net == 'BN_Inception' or args.net == 'BN_Inception_MetaAda':
    #     if args.freeze_BN is True:
    #         print(40 * '#', '\n BatchNorm frozen')
    #         model.apply(set_bn_eval)
    #     else:
    #         print(40 * '#', 'BatchNorm NOT frozen')
    # else:
    #     # for module in filter(lambda m: type(m) == nn.BatchNorm2d, model.modules()):
    #     #     module.eval()
    #     #     module.train = lambda _: None
    #     print(40 * '#', 'R_50_BatchNorm NOT frozen')

    # print(model)
    # new_param_ids = set(map(id, model.module.classifier.parameters()))
    #
    # new_params = [p for p in model.module.parameters() if id(p) in new_param_ids]
    # base_params = [p for p in model.module.parameters() if id(p) not in new_param_ids]
    # param_groups = [{'params': base_params, 'lr_mult': 0.0}, {'params': new_params, 'lr_mult': 1.0}]
    if args.net == 'BN_Inception':
        new_param_ids = set(map(id, model.module.classifier.parameters()))
        new_params = [p for p in model.module.parameters() if id(p) in new_param_ids]
        base_params = [p for p in model.module.parameters() if id(p) not in new_param_ids]
        param_groups = [{'params': base_params, 'lr_mult': 0.0}, {'params': new_params, 'lr_mult': 1.0}]
    elif args.net == 'BNAt_Inception':
        print('Attention_param_loading......')
        new_param_ids = set(list(map(id, model.module.classifier.parameters())) + list(map(id, model.module.sge.parameters())))
        # print(new_param_ids)
        # new_param_ids = set(map(id, model.module.classifier.parameters()))
        # att_param_ids = set(map(id, model.module.mViT_v2.parameters()))
        new_params = [p for p in model.module.parameters() if id(p) in new_param_ids]
        # print(new_params)
        # att_params = [p for p in model.module.parameters() if id(p) in att_param_ids]
        base_params = [p for p in model.module.parameters() if id(p) not in new_param_ids]
        param_groups = [{'params': base_params, 'lr_mult': 0.0},
                        {'params': new_params, 'lr_mult': 1.0}]

    else:
        new_param_ids = set(map(id, model.module.model.classifier.parameters()))

        new_params = [p for p in model.module.model.parameters() if id(p) in new_param_ids]
        base_params = [p for p in model.module.model.parameters() if id(p) not in new_param_ids]
        param_groups = [{'params': base_params, 'lr_mult': 0.0}, {'params': new_params, 'lr_mult': 1.0}]
        # param_groups = model.module.parameters()
    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                     weight_decay=args.weight_decay)




    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    return model, start, optimizer, scheduler

def model_loader_sop(args,start):
    print('SOP_dataset')
    if args.MMSI == 1:
        args.net = args.net + '_MetaAda'
        model = models.create(args.net, data=args.data, pretrained=True, dim=args.dim)
    else:
        model = models.create(args.net, data=args.data, pretrained=True, dim=args.dim)

    if args.resume_sop is None:
        model_dict = model.state_dict()
    else:
        # resume model
        print('load model from {}'.format(args.resume_sop))
        chk_pt = load_checkpoint(args.resume_sop)
        weight = chk_pt['state_dict']
        start = chk_pt['epoch']
        start = 0

        checkpoint = weight
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        print("matched parameters: %d/%d" % (len(state_dict), len(model_dict)))

        model_dict.update(state_dict)
        model.load_state_dict(model_dict) 

    model = torch.nn.DataParallel(model)
    model = model.cuda()




    # freeze BN
    if args.freeze_BN is True:
        print(40 * '#', '\n BatchNorm frozen')
        model.apply(set_bn_eval)
    else:
        print(40 * '#', 'BatchNorm NOT frozen')

    new_param_ids = set(map(id, model.module.classifier.parameters()))

    new_params = [p for p in model.module.parameters() if id(p) in new_param_ids]
    base_params = [p for p in model.module.parameters() if id(p) not in new_param_ids]
    param_groups = [{'params': base_params, 'lr_mult': 0.0}, {'params': new_params, 'lr_mult': 1.0}]

    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                 weight_decay=args.weight_decay)

    return model, start, optimizer