import argparse

def basic_option():
    parser=advance_option()
    parser.add_argument('--dataset', dest='dataset', help='source training dataset',
                        default='pascal_voc', type=str)

    parser.add_argument('--dataset_t', dest='dataset_t', help='target training dataset or testing',
                        default='clipart', type=str) #ou virat_val_scene_04

    parser.add_argument('--net', dest='net', help='vgg16, res101, res50',
                        default='res50', type=str)

    parser.add_argument('--preentrainement', dest='preentrainement',help='preentrainement sans domaine adaptation',
                        default=False, type=bool)
    parser.add_argument('--freeze', dest='freeze', help='Savoir si on freeze la base 1 apres entrainement sans DA',
                        default=False, type=bool)
    parser.add_argument('--epochs', dest='max_epochs',help='number of epochs to train',
                        default=7, type=int)
    parser.add_argument('--eta', dest='eta',
                        help='trade-off parameter between detection loss and domain-alignment loss',
                        default=30, type=float)

    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=98, type=int)
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--vis', dest='vis',
                        help='visualisation',
                        default=False, type=bool)
    return parser
def advance_option():
    #Option qu'on touche peu
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--train_wo_da', dest='train_wo_da',help='number of epochs to pretrain',
                        default=4, type=int)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', help='step to do learning rate decay, unit is epoch',
                        default=[5], type=int)

    parser.add_argument('--start_epoch', dest='start_epoch', help='starting epoch',
                        default=1, type=int)

    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)

    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)

    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--load_name', dest='load_name',
                        help='path to load models', default="",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        default= True)

    parser.add_argument('--detach', dest='detach',
                        help='whether use detach',
                        action='store_false')

    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')


    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)

    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        default= False)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")

    return parser
def Saito_option():
    parser = basic_option()

    ################################
    ### Saito
    ################################

    parser.add_argument('--lc', dest='lc',
                        help='whether use context vector for pixel level',
                        default= True)
    parser.add_argument('--gc', dest='gc',
                        help='whether use context vector for global level',
                        default= True)
    parser.add_argument('--gamma', dest='gamma',
                        help='value of gamma',
                        default=5, type=float)
    parser.add_argument('--ef', dest='ef',
                        help='whether use exponential focal loss',
                        action='store_true')
    args = parser.parse_args()
    return args
def HTCN_option():
    parser = basic_option()

    ###################################
    ###    HTCN
    ####################################

    parser.add_argument('--lc', dest='lc',
                        help='whether use context vector for pixel level',
                        default= True)
    parser.add_argument('--gc', dest='gc',
                        help='whether use context vector for global level',
                        default= True)
    parser.add_argument('--LA_ATT', dest='LA_ATT',
                        help='whether to use local attention',
                        default=True)
    parser.add_argument('--MID_ATT', dest='MID_ATT',
                        help='whether to use middle attention',
                        default=True)
    parser.add_argument('--gamma', dest='gamma',
                        help='value of gamma',
                        default=3, type=float)
    parser.add_argument('--ef', dest='ef',
                        help='whether use exponential focal loss',
                        action='store_true')
    args = parser.parse_args()
    return args
def HSU_option():
    parser = basic_option()

    ###################################
    ###    HSU
    ####################################
    args = parser.parse_args()
    return args
def faster_option():
    parser = basic_option()

    ###################################
    ###    faster
    ####################################
    args = parser.parse_args()
    return args



def set_dataset_args(args, test=False):
    if not test:
        if args.dataset == "mot20_scene_1":
            args.imdb_name = "mot20_train_scene_1"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '60']

        elif args.dataset == "mot20_camera_02":
            args.imdb_name = "mot20_train_camera_02"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '40']

        elif args.dataset == "mot20_camera_34":
            args.imdb_name = "mot20_train_camera_03+mot20_train_camera_04"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '60']
        # 'train_camera_01','train_camera_02','test_camera_01','test_camera_02']:
        elif args.dataset == "mot20_camera_05":
            args.imdb_name = "mot20_train_camera_05"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '60']
### virat
        elif args.dataset == "virat":
            args.imdb_name = "virat_scene01+virat_scene02+virat_scene06+virat_scene10"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "virat_limite_haute":
            args.imdb_name = "virat_train_scene04"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
### sim10k
        elif args.dataset == "sim10k_car":
            args.imdb_name = "sim10k_car_train+sim10k_car_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "kitti_car":
            args.imdb_name = "kitti_car_all"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cityscape_car":
            args.imdb_name = "cityscape_car_train+cityscape_car_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "fake_kitti_car":
            args.imdb_name = "fake_kitti_car_all"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cityscape":
            args.imdb_name = "cityscape_train+cityscape_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "wildtrack":
            args.imdb_name = "wildtrack_camera01"#+wildtrack_camera02+wildtrack_camera03+wildtrack_camera04+wildtrack_camera05"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '40']
        elif args.dataset == "videowave":
            args.imdb_name = "videowave_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '5']


        elif args.dataset == "pascal_voc":
            args.imdb_name = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "pascal_voc_cycleclipart_car":
            args.imdb_name = "voc_cycleclipart_car_2007_trainval+voc_cycleclipart_car_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "pascal_voc_cycleclipart_car_500":
            args.imdb_name = "voc_cycleclipart_car_2012_500"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        ############################ target ################################################

        if args.dataset_t == "mot20_camera_02":
            args.imdb_name_target = "mot20_train_camera_02"
        if args.dataset_t == "mot20_camera_05":
            args.imdb_name_target = "mot20_train_camera_05"
        elif args.dataset_t == "mot20_scene_2":
            args.imdb_name_target = "mot20_train_scene_2"


        elif args.dataset_t == "virat":
            args.imdb_name_target = "virat_train_scene04"

        elif args.dataset_t == "virat_all":
            args.imdb_name_target = "virat_train_all"
        elif args.dataset_t == "cityscape_car":
            args.imdb_name_target = "cityscape_car_train"
        elif args.dataset_t == "cityscape":
            args.imdb_name_target = "cityscape_train"
        elif args.dataset_t == "kitti_car":
            args.imdb_name_target = "kitti_car_test"

        elif args.dataset_t == "fake_kitti_car":
            args.imdb_name_target = "fake_kitti_car_all"

        elif args.dataset_t == "clipart":
            args.imdb_name_target = "clipart_trainval"

        elif args.dataset_t == "clipart_train":
            args.imdb_name_target = "clipart_train"

        elif args.dataset_t == "wildtrack":
            args.imdb_name_target = "wildtrack_camera06+wildtrack_camera07"
        elif args.dataset_t == "videowave":
            args.imdb_name_target = "videowave_train"


    else:
        if args.dataset_t == "mot20_scene_2":
            args.imdb_name_target = "mot20_test_scene_2" #test_scene_2 'test_camera_03', 'test_camera_04', 'test_camera_05'4
            #'train_camera_01','train_camera_02','test_camera_01','test_camera_02']:
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '60']
        elif args.dataset_t == "mot20_scene_1":
            args.imdb_name_target = "mot20_test_scene_1"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '40']
        elif args.dataset_t == "mot20_camera_02":
            args.imdb_name_target = "mot20_test_camera_02"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '40']
        elif args.dataset_t == "mot20_camera_05":
            args.imdb_name_target = "mot20_test_camera_05"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '60']


        elif args.dataset_t == "virat":
            args.imdb_name_target = "virat_test_scene04"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '20']
        elif args.dataset_t == "virat_all":
            args.imdb_name_target = "virat_test_all"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '20']
        elif args.dataset_t == "virat_val_scene_04":
            args.imdb_name_target = "virat_train_scene04"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '20']
        elif args.dataset_t == "virat_val2_scene_04":
            args.imdb_name_target = "virat_val_scene04"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '20']

        elif args.dataset_t == "cityscape_car":
            args.imdb_name_target = "cityscape_car_test"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES','20']

        elif args.dataset_t == "cityscape":
            args.imdb_name_target = "cityscape_test"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "kitti_car":
            args.imdb_name_target = "kitti_car_test"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES','20']

        elif args.dataset_t == "fake_kitti_car":
            args.imdb_name_target = "fake_kitti_car_test"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "clipart":
            args.imdb_name_target = "clipart_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '20']
        elif args.dataset_t == "wildtrack":
            args.imdb_name_target = "wildtrack_camera06"#+wildtrack_camera07"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '40']

        elif args.dataset_t == "videowave":
            args.imdb_name_target = "videowave_test"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '5']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    return args
