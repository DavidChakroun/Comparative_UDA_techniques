# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4" #Permet de choisir la carte graphique qu'on veut utiliser
import numpy as np
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, FocalLoss, EFocalLoss, sampler
from model.utils.parser_func import Saito_option, set_dataset_args



if __name__ == '__main__':

    args = Saito_option()
    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name) #roidb contient l'image
    train_size = len(roidb)
    # target dataset
    cfg.TRAIN.USE_FLIPPED = True
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target, training=False)
    train_size_t = len(roidb_t)

    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + "/Saito"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)
    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True) #probl??me ici, image[0]=batchsize
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=False)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    from model.faster_rcnn.vgg16_saito import vgg16
    from model.faster_rcnn.resnet_saito import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, lc=args.lc,
                           gc=args.gc)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,
                            lc=args.lc, gc=args.gc)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, lc=args.lc, gc=args.gc)

    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.preentrainement:
        load_name = 'models/res50/pascal_voc_cycleclipart_car/Saito/saito_target_virat_eta_0.5_local_context_True_global_context_True_gamma_5_session_5_epoch_3_step_9999_wo_DA.pth'
        checkpoint = torch.load(load_name)
        fasterRCNN.load_state_dict(checkpoint['model'])
        print("loaded checkpoint %s" % (load_name))

    if args.resume:
        load_name = 'models/res50/virat/Saito/target_virat_eta_0.1_local_True_global_True_gamma_5_session_5_epoch_3_step_9999_preentrainement_True_freeze_False.pth'
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))


    iters_per_epoch = int(10000 / args.batch_size)
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)

    if args.freeze:
        for p in fasterRCNN.RCNN_base1[:].parameters():
            p.requires_grad = False
        for p in fasterRCNN.RCNN_base2[0].parameters():
            p.requires_grad = False

    count_iter = 0
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()
        if epoch - 1 in  args.lr_decay_step:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)
        for step in range(iters_per_epoch):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)
            #eta = 1.0
            count_iter += 1
            #put source data into variable
            with torch.no_grad():
                im_data.resize_(data_s[0].size()).copy_(data_s[0])
                im_info.resize_(data_s[1].size()).copy_(data_s[1])
                gt_boxes.resize_(data_s[2].size()).copy_(data_s[2])
                num_boxes.resize_(data_s[3].size()).copy_(data_s[3])
            fasterRCNN.zero_grad()
            # print("im_info",im_info)
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, out_d_pixel, out_d = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()


            # domain label
            domain_s = Variable(torch.zeros(out_d.size(0)).long().cuda())
            # global alignment loss
            dloss_s = 0.5 * FL(out_d, domain_s)
            # local alignment loss
            dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)

            #put target data into variable
            im_data.resize_(data_t[0].size()).copy_(data_t[0])
            im_info.resize_(data_t[1].size()).copy_(data_t[1])
            #gt is empty
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
            out_d_pixel, out_d = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=True)
            # domain label
            domain_t = Variable(torch.ones(out_d.size(0)).long().cuda())
            dloss_t = 0.5 * FL(out_d, domain_t)
            # local alignment loss
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)
            if args.dataset == 'sim10k':
                loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p) * 0.1
            else:
                loss += (dloss_s + dloss_t + dloss_s_p + dloss_t_p) * args.eta
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 1000 == 0:
                save_name = os.path.join(output_dir,
                                         'target_{}_eta_{}_local_{}_global_{}_gamma_{}_session_{}_epoch_{}_step_{}_preentrainement_{}_freeze_{}.pth'.format(
                                             args.dataset_t, args.eta,
                                             args.lc, args.gc, args.gamma,
                                             args.session, epoch,
                                             step, args.preentrainement, args.freeze))
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
                print('save model: {}'.format(save_name))


            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    dloss_s = dloss_s.item()
                    dloss_t = dloss_t.item()
                    dloss_s_p = dloss_s_p.item()
                    dloss_t_p = dloss_t_p.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f dloss s: %.4f dloss t: %.4f dloss s pixel: %.4f dloss t pixel: %.4f eta: %.4f" \
                    % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, dloss_s, dloss_t, dloss_s_p, dloss_t_p,
                       args.eta))

        save_name = os.path.join(output_dir,
                                 'target_{}_eta_{}_local_{}_global_{}_gamma_{}_session_{}_epoch_{}_step_{}_preentrainement_{}_freeze_{}.pth'.format(
                                     args.dataset_t,args.eta,
                                     args.lc, args.gc, args.gamma,
                                     args.session, epoch,
                                     step, args.preentrainement,args.freeze))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))


