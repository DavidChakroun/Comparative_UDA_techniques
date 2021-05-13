from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5" #Permet de choisir la carte graphique qu'on veut utiliser
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, sampler
from model.utils.parser_func import HSU_option, set_dataset_args
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb
from model.faster_rcnn.vgg16_HSU import vgg16
from model.faster_rcnn.resnet_HSU import resnet as resnet
import pprint
import numpy as np
import time
import random
import json


if __name__ == '__main__':

    args = HSU_option()

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
    random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)

    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True)
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



    # load network
    if args.net == 'vgg16':
        fasterRCNN = vgg16()
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False)
    else:
        raise NotImplementedError

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

    if args.resume:
        args.load_name='models/res50/mot20_scene_2/HSU_target_mot20_scene_1_eta_0.1_session_2_epoch_6_step_10000.pth'
        checkpoint = torch.load(args.load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (args.load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch = int(10000 / args.batch_size)


#### On charge le fichier des poids si on est sur des images synthetiques:
    if "fake" in args.dataset:
        with open("fichier_poids/poids_netD_B.json", "r") as read_file:
            weight_score= json.load(read_file)

    bceLoss_func = nn.BCEWithLogitsLoss()
    # if args.use_tfboard:
    #     from tensorboardX import SummaryWriter
        # logger = SummaryWriter("logs")
    count_iter = 0
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()

        count_step = 0
        loss_temp_last = 1
        loss_temp = 0
        loss_rpn_cls_temp = 0
        loss_rpn_box_temp = 0
        loss_rcnn_cls_temp = 0
        loss_rcnn_box_temp = 0

        start = time.time()
        # if epoch % (args.lr_decay_step + 1) == 0:
        if epoch - 1 in  args.lr_decay_step:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)

        for step in range(1, iters_per_epoch + 1):
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
            # "utilise pour la phase 1, on recup
            if 'fake' in args.dataset:
                synth_weight=weight_score[data_s[4][0][-10:]]
            else:
                synth_weight = 1

            source_label = 0
            target_label = 1

            ### Put source data into variable

            im_data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.resize_(data_s[3].size()).copy_(data_s[3])

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, D_img_out = fasterRCNN(im_data, im_info, gt_boxes, num_boxes) #meme perte que dans HSU original
            #dans l'origianl ils ne font qu'une addition des pertes, pas une moyenne
            count_step += 1
            ######################### loss source #####################################
            loss_s = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

            loss_temp += loss_s.item()
            loss_rpn_cls_temp += rpn_loss_cls.mean().item()
            loss_rpn_box_temp += rpn_loss_box.mean().item()
            loss_rcnn_cls_temp += RCNN_loss_cls.mean().item()
            loss_rcnn_box_temp += RCNN_loss_bbox.mean().item()

            # domain label
            domain_s = Variable(torch.zeros(D_img_out.size(0)).long().cuda())
            # global alignment loss
            loss_D_img_S = bceLoss_func(D_img_out,Variable(torch.FloatTensor(D_img_out.data.size()).fill_(source_label)).cuda())

            total_loss_S = loss_s + (0.05) * loss_D_img_S

            #put target data into variable
            im_data.resize_(data_t[0].size()).copy_(data_t[0])
            im_info.resize_(data_t[1].size()).copy_(data_t[1])
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()

            D_img_out = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=True)
            ######################### loss target #####################################
            loss_D_img_T = bceLoss_func(D_img_out,
                                        Variable(torch.FloatTensor(D_img_out.data.size()).fill_(source_label)).cuda())
            total_loss_T = (0.05) * loss_D_img_T

            total_loss = total_loss_S + total_loss_T


            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():

                print("s_name: {}".format(data_s[4]))
                print("s_gt_boxes:{}".format(data_s[2]))
                print("s_im_info:{}".format(data_s[1]))
                print("s_num_bixes:{}".format(data_s[3]))

                print("t_name: {}".format(data_t[4]))
                print("t_gt_boxes:{}".format(data_t[2]))
                print("t_im_info:{}".format(data_t[1]))
                print("t_num_bixes:{}".format(data_t[3]))
                raise ValueError

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()

                loss_temp /= count_step
                loss_rpn_cls_temp /= count_step
                loss_rpn_box_temp /= count_step
                loss_rcnn_cls_temp /= count_step
                loss_rcnn_box_temp /= count_step


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
                    loss_D_img_S = loss_D_img_S.item()
                    loss_D_img_T = loss_D_img_T.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e, step: %3d, count: %3d" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr, count_step, count_iter))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f loss_D_img_S s: %.4f loss_D_img_T t: %.4f eta: %.4f" \
                    % (loss_rpn_cls_temp, loss_rpn_box_temp, loss_rcnn_cls_temp, loss_rcnn_box_temp, loss_D_img_S, loss_D_img_T,
                       args.eta))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls_temp,
                        'loss_rpn_box': loss_rpn_box_temp,
                        'loss_rcnn_cls': loss_rcnn_cls_temp,
                        'loss_rcnn_box': loss_rcnn_box_temp
                    }
                    # logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                    #                    (epoch - 1) * iters_per_epoch + step)
                    logger.add_scalars(args.log_ckpt_name, info,
                                       (epoch - 1) * iters_per_epoch + step)

                count_step = 0
                loss_temp_last = loss_temp
                loss_temp = 0
                loss_rpn_cls_temp = 0
                loss_rpn_box_temp = 0
                loss_rcnn_cls_temp = 0
                loss_rcnn_box_temp = 0

                start = time.time()


        save_name = os.path.join(output_dir,
                                 'HSU_target_{}_eta_{}_session_{}_epoch_{}_step_{}.pth'.format(
                                     args.dataset_t,args.eta,
                                     args.session, epoch,
                                     step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()

