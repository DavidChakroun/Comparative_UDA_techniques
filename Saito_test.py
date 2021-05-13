# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Permet de choisir la carte graphique qu'on veut utiliser
import sys
import numpy as np
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections
from model.faster_rcnn.vgg16_saito import vgg16
from model.faster_rcnn.resnet_saito import resnet
from model.utils.parser_func import Saito_option,set_dataset_args

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3



lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = Saito_option()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args, test=True)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs_target is not None:
        cfg_from_list(args.set_cfgs_target)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name_target, training=False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))


    # initilize the network here.
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
    load_names =[
       'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_1_step_1000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_1_step_2000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_1_step_3000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_1_step_4000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_1_step_5000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_1_step_6000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_1_step_7000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_1_step_8000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_1_step_9000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_1_step_9999_preentrainement_False_freeze_False.pth',

'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_2_step_1000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_2_step_2000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_2_step_3000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_2_step_4000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_2_step_5000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_2_step_6000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_2_step_7000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_2_step_8000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_2_step_9000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_2_step_9999_preentrainement_False_freeze_False.pth',

'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_3_step_1000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_3_step_2000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_3_step_3000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_3_step_4000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_3_step_5000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_3_step_6000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_3_step_7000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_3_step_8000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_3_step_9000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_3_step_9999_preentrainement_False_freeze_False.pth',

'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_4_step_1000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_4_step_2000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_4_step_3000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_4_step_4000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_4_step_5000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_4_step_6000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_4_step_7000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_4_step_8000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_4_step_9000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_4_step_9999_preentrainement_False_freeze_False.pth',

'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_5_step_1000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_5_step_2000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_5_step_3000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_5_step_4000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_5_step_5000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_5_step_6000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_5_step_7000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_5_step_8000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_5_step_9000_preentrainement_False_freeze_False.pth',
'models/res50/pascal_voc_cycleclipart_car_500/Saito/target_virat_eta_1_local_True_global_True_gamma_5_session_99_epoch_5_step_9999_preentrainement_False_freeze_False.pth',
        ]
    for load_name in load_names:
        checkpoint = torch.load(load_name)
        fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        print("load checkpoint %s" % (load_name))
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

        if args.cuda:
            fasterRCNN.cuda()

        start = time.time()
        max_per_image = 100

        vis = args.vis

        if vis:
            thresh = 0.05
        else:
            thresh = 0.0
        if args.mGPUs:
            fasterRCNN = nn.DataParallel(fasterRCNN)
        num_images = len(imdb.image_index)
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]

        output_dir = get_output_dir(imdb, 'save_name')
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                                 imdb.num_classes, training=False, normalize=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 shuffle=False, num_workers=0,
                                                 pin_memory=False)

        data_iter = iter(dataloader)

        _t = {'im_detect': time.time(), 'misc': time.time()}
        det_file = os.path.join(output_dir, 'detections.pkl')

        fasterRCNN.eval()
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
        for i in range(num_images):

            data = next(data_iter)

            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label,d_pred,_ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]
            d_pred = d_pred.data
            # path = data[4]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2].item()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()
            if vis:
                im = cv2.imread(imdb.image_path_at(i))
                im2show = np.copy(im)
            for j in xrange(1, imdb.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis:
                        im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                          for j in xrange(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in xrange(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

            if vis:
                path_image=data[4][0]
                path_image=path_image.split('/')
                nom_image=path_image[-1]
                atrlan='virat_1'
                nameImageVis=('resultat/Saito/'+atrlan+'/'+nom_image)
                if not os.path.exists('resultat/Saito/'+atrlan):
                    os.makedirs('resultat/Saito/'+atrlan)
                cv2.imwrite(nameImageVis, im2show)


        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        imdb.evaluate_detections(all_boxes, output_dir)

        end = time.time()
        print("test time: %0.4fs" % (end - start))



    bob=False
    if bob==True:
    ##### test de toutes les bases

        imdb, roidb, ratio_list, ratio_index = combined_roidb('mot20_test_camera_01', training=False) ##test_scene_2 'test_camera_03', 'test_camera_04', 'test_camera_05'
        imdb.competition_mode(on=True)
        print('{:d} roidb entries'.format(len(roidb)))
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                                 imdb.num_classes, training=False, normalize=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,  pin_memory=False)

        data_iter = iter(dataloader)
        num_images = len(imdb.image_index)
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]
        _t = {'im_detect': time.time(), 'misc': time.time()}
        for i in range(num_images):

            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label,d_pred,_ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]
            d_pred = d_pred.data
            # path = data[4]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2].item()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()
            if vis:
                im = cv2.imread(imdb.image_path_at(i))
                im2show = np.copy(im)
            for j in xrange(1, imdb.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis:
                        im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                          for j in xrange(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in xrange(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

            if vis:
                cv2.imwrite('result.png', im2show)
                pdb.set_trace()


        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections test_camera_01')
        imdb.evaluate_detections(all_boxes, output_dir)

        end = time.time()
        print("test time: %0.4fs" % (end - start))


    ##### test de toutes les blabla
    # 'test_scene_1','test_camera_01','test_camera_02', 'test_camera_07'
        print("test_camera_02")
        imdb, roidb, ratio_list, ratio_index = combined_roidb("mot20_test_camera_02", training=False) ##test_scene_2 'test_camera_03', 'test_camera_04', 'test_camera_05'
        imdb.competition_mode(on=True)

        print('{:d} roidb entries'.format(len(roidb)))
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                                 imdb.num_classes, training=False, normalize=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 shuffle=False, num_workers=0,
                                                 pin_memory=False)

        data_iter = iter(dataloader)

        num_images = len(imdb.image_index)
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]
        _t = {'im_detect': time.time(), 'misc': time.time()}
        for i in range(num_images):

            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label,d_pred,_ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]
            d_pred = d_pred.data
            # path = data[4]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2].item()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()
            if vis:
                im = cv2.imread(imdb.image_path_at(i))
                im2show = np.copy(im)
            for j in xrange(1, imdb.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis:
                        im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                          for j in xrange(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in xrange(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

            if vis:
                cv2.imwrite('result.png', im2show)
                pdb.set_trace()


        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections test_camera_02')
        imdb.evaluate_detections(all_boxes, output_dir)

        end = time.time()
        print("test time: %0.4fs" % (end - start))


    ##### test de toutes les blabla
    # 'test_scene_1','test_camera_01','test_camera_02', 'test_camera_07'
        print("_______ test_camera_07_______________")
        imdb, roidb, ratio_list, ratio_index = combined_roidb("mot20_test_camera_07", training=False) ##test_scene_2 'test_camera_03', 'test_camera_04', 'test_camera_05'
        imdb.competition_mode(on=True)

        print('{:d} roidb entries'.format(len(roidb)))
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                                 imdb.num_classes, training=False, normalize=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 shuffle=False, num_workers=0,
                                                 pin_memory=False)

        data_iter = iter(dataloader)

        num_images = len(imdb.image_index)
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]
        _t = {'im_detect': time.time(), 'misc': time.time()}
        for i in range(num_images):

            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label,d_pred,_ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]
            d_pred = d_pred.data
            # path = data[4]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2].item()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()
            if vis:
                im = cv2.imread(imdb.image_path_at(i))
                im2show = np.copy(im)
            for j in xrange(1, imdb.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis:
                        im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                          for j in xrange(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in xrange(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

            if vis:
                cv2.imwrite('result.png', im2show)
                pdb.set_trace()


        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections test_camera_07')
        imdb.evaluate_detections(all_boxes, output_dir)

        end = time.time()
        print("test time: %0.4fs" % (end - start))