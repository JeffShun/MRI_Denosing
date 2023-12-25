import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
from custom.dataset.dataset import MyDataset
from custom.dataset.data_tools import *
from custom.model.backbones.ResUnet import ResUnet
from custom.model.model_head import Model_Head
from custom.model.model_network import Model_Network
from custom.model.model_loss import *

class network_cfg:
    # img
    patch_size = (160, 160)
    # network
    network = Model_Network(
        backbone = ResUnet(
            in_ch=1, 
            channels=32, 
            blocks=2
        ),
        head = Model_Head(
            in_channels=32,
            num_class=1
        ),
        apply_sync_batchnorm=False,
    )

    # loss function
    train_loss_f = LossCompose([
        LossDropoutWrapper(SSIMLoss(win_size = 7, k1 = 0.01, k2 = 0.03, reduce=False), drop_p=0.5, beta=5),
        LossDropoutWrapper(MSELoss(reduce=False), drop_p=0.5, beta=5),
        ])
    
    valid_loss_f = LossCompose([
        SSIMLoss(win_size = 7, k1 = 0.01, k2 = 0.03, reduce=True),
        MSELoss(reduce=True)
        ])


    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/train.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            random_flip(axis=1, prob=0.5),
            random_flip(axis=2, prob=0.5),
            random_rotate90(k=1, prob=0.5),
            random_crop(patch_size),
            resize(patch_size),
            ])
        )
    valid_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/valid.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            random_crop(patch_size),
            resize(patch_size),
            ])
        )
    
    # dataloader
    batchsize = 4
    shuffle = True
    num_workers = 4
    drop_last = False

    # optimizer
    lr = 1e-4
    weight_decay = 5e-4

    # scheduler
    milestones = [20,40,80]
    gamma = 0.5
    warmup_factor = 0.1
    warmup_iters = 1
    warmup_method = "linear"
    last_epoch = -1

    # debug
    valid_interval = 1
    log_dir = work_dir + "/Logs/ResUnet"
    checkpoints_dir = work_dir + '/checkpoints/ResUnet'
    checkpoint_save_interval = 1
    total_epochs = 200
    load_from = work_dir + '/checkpoints/None/50.pth'

    # others
    device = 'cuda'
    dist_backend = 'nccl'
    dist_url = 'env://'
