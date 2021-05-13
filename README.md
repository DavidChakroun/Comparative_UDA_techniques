A Pytorch Implementation for DA comparaison

## Introduction
This respository was created to compare different Unsupervised Domain Adaptation Techniques : [Saito](https://github.com/VisionLearningGroup/DA_Detection), [HSU](https://github.com/kevinhkhsu/DA_detection) and [HTCN](https://github.com/chaoqichen/HTCN).
These techniques have been updated to PyTorch 1.1
Please follow [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) respository to setup the environment. In this project, we use Pytorch 1.1 and CUDA version is 10.2

## Additional Information

### Datasets Format
All codes are written to fit for the **format of PASCAL_VOC**.  
If you want to use this code on your own dataset, please arrange the dataset in the format of PASCAL, make dataset class in ```lib/datasets/```, and add it to ```lib/datasets/factory.py```. Then, add the dataset option to ```lib/model/utils/parser_func.py```.


## Models
### Pre-trained Models
In our experiments, we used two pre-trained models on ImageNet, i.e., VGG16 and ResNet101. Please download these two models from:
* **ResNet50:** Download the model (the one for BGR) from [link](https://github.com/ruotianluo/pytorch-resnet)

### Be careful
The Pre-trained Models are pretrained with Caffe, so the image need to be in BGR and in range of \[0-255] (The code do it itself).
You can't use PyTorch pretrained model without code modification

