# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import pprint
import time
import numpy as np
from sklearn.manifold import TSNE
import os
from matplotlib import pyplot as plt


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Permet de choisir la carte graphique qu'on veut utiliser
import cv2

import torch
from torch.autograd import Variable
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import sampler
from model.utils.parser_func import faster_option, set_dataset_args

from model.faster_rcnn.resnet import resnet

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':
    image_pour_tsne=200
    Saito=False
    HTCN=True
    load_name = 'models/res50/mot20_scene_1/Faster/mot_scene_1_faster_rcnn_1_7_7591.pth'#'models/res50/pascal_voc_cycleclipart/faster/pascal_voc_cycleclipart_faster_rcnn_1_7_9999.pth'
    load_name_da = 'models/res50/mot20_scene_1/HTCN/target_mot20_scene_2_eta_0.1_local_True_global_True_gamma_3_session_15_epoch_7_step_10000_prenetrainement_True_freeze_False.pth'
    args = faster_option()
    print('Called with args:')
    print(args)


    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)


    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name) #roidb contient l'image
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target, training=False)
    train_size_t = len(roidb_t)
    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)
    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=False)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=False)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)


#### tsne avec source et target sans da
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN.create_architecture()
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    print('load model successfully!')
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
        cfg.CUDA = True
        fasterRCNN.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)


    start = time.time()
    data_iter = iter(dataloader_s)
    fasterRCNN.eval()
    for i in range(image_pour_tsne):
        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            basefeat_source = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, tsne=True)
            basefeat = basefeat_source.view(1, -1)
            try :
                basefeatCat_source=torch.cat((basefeatCat_source,basefeat),0)
            except :
                basefeatCat_source=torch.clone(basefeat)
            print(i)
    end = time.time()
    print("test time: %0.4fs" % (end - start))



    data_iter = iter(dataloader_t)
    fasterRCNN.eval()
    for i in range(image_pour_tsne):
        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            basefeat_source = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, tsne=True)
            basefeat = basefeat_source.view(1, -1)
            basefeatCat_source = torch.cat((basefeatCat_source, basefeat), 0)
            print(i)
    end = time.time()
    print("test time: %0.4fs" % (end - start))

    basefeatCat_source=basefeatCat_source.cpu().detach().numpy()


    ### meme chose mais avec adaptation de domaine

    if Saito==True:
        from model.faster_rcnn.resnet_saito import resnet as resnet_saito

    if HTCN==True:
        from model.faster_rcnn.resnet_HTCN import resnet as resnet_HCTN
        fasterRCNN = resnet_HCTN(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic,
                                lc=True, gc=True, la_attention = True, mid_attention = True)
    fasterRCNN.create_architecture()
    print("load checkpoint %s" % (load_name_da))
    checkpoint = torch.load(load_name_da)
    fasterRCNN.load_state_dict(checkpoint['model'])
    print('load model successfully!')
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
        cfg.CUDA = True
        fasterRCNN.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    start = time.time()
    data_iter = iter(dataloader_s)
    fasterRCNN.eval()
    for i in range(image_pour_tsne):
        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            basefeat_cible = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, tsne=True)
            basefeat = basefeat_cible.view(1, -1)
            try:
                basefeatCat_cible = torch.cat((basefeatCat_cible, basefeat), 0)
            except:
                basefeatCat_cible = torch.clone(basefeat)
            print(i)
    end = time.time()
    print("test time: %0.4fs" % (end - start))

    data_iter = iter(dataloader_t)
    fasterRCNN.eval()
    for i in range(image_pour_tsne):
        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            basefeat_cible = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, tsne=True)
            basefeat = basefeat_cible.view(1, -1)
            basefeatCat_cible = torch.cat((basefeatCat_cible, basefeat), 0)
            print(i)

    end = time.time()
    print("test time: %0.4fs" % (end - start))

    basefeatCat_cible = basefeatCat_cible.cpu().detach().numpy()
    basefeatCat=np.concatenate((basefeatCat_source,basefeatCat_cible),0)
    torch.cuda.empty_cache()
    print("test time: %0.4fs" % (end - start))
    print(basefeatCat.shape)
    print("maintenant, tnse")
    start = time.time()
    X_embedded = TSNE(n_components=2).fit_transform(basefeatCat)
    np.save('mot1_mot2',X_embedded)
    end = time.time()
    print("test time: %0.4fs" % (end - start))
    print('X_embedded',X_embedded.shape)
    source=0
    cible=source+image_pour_tsne+1
    sourceda=cible+image_pour_tsne+1
    cibleda=sourceda+image_pour_tsne+1
    plt.scatter(X_embedded[source:cible-1, 0], X_embedded[source:cible-1, 1], s = 60,marker = 'x', c='red')
    plt.scatter(X_embedded[cible:sourceda-1, 0], X_embedded[cible:sourceda-1, 1], s = 60,marker = '+', c = 'mediumblue')
    plt.scatter(X_embedded[sourceda:cibleda-1, 0], X_embedded[sourceda:cibleda-1, 1], s = 60,marker = 'x', c='orange')
    plt.scatter(X_embedded[cibleda:, 0], X_embedded[cibleda:, 1], s = 60,marker = '+', c = 'darkorchid')

    plt.savefig("mot1_mot2.jpg")