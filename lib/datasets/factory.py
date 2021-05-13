# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}

from datasets.cityscape_car import cityscape_car
from datasets.cityscape import cityscape
from datasets.foggy_cityscape import foggyCityscape
from datasets.kitti_car import kitti_car
# from datasets.kitti_person import kitti_person
from datasets.mot20 import mot20
from datasets.fake_kitti_car import fake_kitti_car

from datasets.pascal_voc_cycleclipart import pascal_voc_cycleclipart,pascal_voc_cycleclipart_car
from datasets.clipart import clipart
from datasets.wildtrack import wildtrack
from datasets.videowave import videowave
from datasets.virat import virat

from datasets.sim10k import sim10k_car
#set up sim10k
for split in ['train','test']:
    name = 'sim10k_car_{}'.format(split)
    __sets[name] = (lambda split=split : sim10k_car(split))

#set up virat
for split in ['all','train_all','test_all','scene01','scene02','train_scene04','val_scene04','scene06','train_scene07','train_scene08',
              'test_scene04','test_scene07','test_scene08', 'scene10','train']:
    name = 'virat_{}'.format(split)
    __sets[name] = (lambda split=split : virat(split))


# Set up videowave
for split in ['all','train','test']:
    name = 'videowave_{}'.format(split)
    __sets[name] = (lambda split=split : videowave(split))
# Set up wildtrack
for split in ['camera01','camera02','camera03','camera04','camera05','camera06','camera07','all']:
    name = 'wildtrack_{}'.format(split)
    __sets[name] = (lambda split=split : wildtrack(split))

# Set up mot20
for split in ['train1', 'scene_1', "train_scene_1",'scene_2','train_scene_2', 'scene_3',
              'train_camera_03', 'test_camera_03', 'train_camera_04','test_camera_04', 'train_camera_05','test_camera_05',
              'test_scene_2','test_scene_1', 'train_camera_01','train_camera_02','test_camera_01','test_camera_02', 'test_camera_07']:
    name = 'mot20_{}'.format(split)
    __sets[name] = (lambda split=split : mot20(split))

# Set up foggyCityscape
for split in ['train', 'test', 'munster', 'lindau', 'frankfurt', 'val']:
    name = 'foggyCityscape_{}'.format(split)

    __sets[name] = (lambda split=split : foggyCityscape(split))


# Set up cityscape_car
for split in ['train', 'test', 'munster', 'lindau', 'frankfurt', 'val']:
    name = 'cityscape_car_{}'.format(split)

    __sets[name] = (lambda split=split : cityscape_car(split))
# Set up cityscape_car
for split in ['train', 'test', 'munster', 'lindau', 'frankfurt', 'val']:
    name = 'cityscape_{}'.format(split)

    __sets[name] = (lambda split=split : cityscape(split))

# Set up kitti_car
for split in ['train', 'test', 'all']:
    name = 'kitti_car_{}'.format(split)
    __sets[name] = (lambda split=split : kitti_car(split))

# Set up fake_kitti_car
for split in ['train', 'test', 'all']:
    name = 'fake_kitti_car_{}'.format(split)
    __sets[name] = (lambda split=split : fake_kitti_car(split))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test', '500']:
      name = 'voc_cycleclipart_car_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc_cycleclipart_car(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_cycleclipart_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc_cycleclipart(split, year))
for year in ['2007']:
  for split in ['trainval', 'test','train']:
    name = 'clipart_{}'.format(split)
    __sets[name] = (lambda split=split : clipart(split,year))
###########################################


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
