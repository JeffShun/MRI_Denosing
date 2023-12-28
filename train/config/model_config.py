import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
from custom.dataset.dataset import MyDataset
from custom.dataset.data_tools import *
from custom.model.backbones.Cascade_Arch import Cascade_Arch
from custom.model.backbones.ResUnet import ResUnet
from custom.model.backbones.MWResUNet import MWResUnet
from custom.model.backbones.DnCNN import DnCNN
from custom.model.backbones.SwinIR import SwinIR
from custom.model.backbones.Restormer import Restormer
from custom.model.backbones.MWCNN import MWCNN
from custom.model.model_head import Model_Head
from custom.model.model_network import Model_Network
from custom.model.model_loss import *

class network_cfg:
    # img
    patch_size = (256, 256)
    # network
    network = Model_Network(
        # backbone = DnCNN(
        #     depth=17, 
        #     n_channels=64, 
        #     image_channels=1, 
        #     kernel_size=3
        #     ),
        # backbone = MWCNN(
        #     in_channels=1, 
        #     out_channels=1, 
        #     n_feats=32
        #     ),
        backbone = MWResUnet(
            in_chans=1, 
            out_chans=1, 
            num_chans=32, 
            n_res_blocks=2, 
            global_residual=True,
            ),
        # backbone = Cascade_Arch(
        #     base_model = 
        #         ResUnet(
        #             in_chans=1, 
        #             out_chans=1, 
        #             num_chans=32, 
        #             n_res_blocks=2, 
        #             global_residual=True,
        #         ),
        #     num_iter=4,
        #     share_paras=True,
        #     ),
        # backbone = Restormer(        
        #     inp_channels=1, 
        #     out_channels=1, 
        #     dim = 48,
        #     num_blocks = [4,6,6,8], 
        #     num_refinement_blocks = 3,
        #     heads = [1,2,4,8],
        #     ffn_expansion_factor = 2.66,
        #     bias = False,
        #     LayerNorm_type = 'WithBias',   
        #     dual_pixel_task = False        
        #     ),

        # backbone = SwinIR(
        #     in_chans=1, 
        #     upscale=1, 
        #     img_size=patch_size,
        #     window_size=8, 
        #     img_range=1., 
        #     depths=[4, 6, 6, 8],
        #     embed_dim=48, 
        #     num_heads=[1, 2, 4, 8], 
        #     mlp_ratio=2, 
        #     upsampler='pixelshuffledirect'
        #     ),
        head = Model_Head(),
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
        dst_list_file = work_dir + "/train_data/processed_data_para2/train.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            random_flip(axis=1, prob=0.5),
            random_flip(axis=2, prob=0.5),
            random_rotate90(k=1, prob=0.5),
            random_center_crop(crop_size=(256,256), shift_range=(30,30), prob=0.5),
            resize(patch_size),
            ])
        )
    valid_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data_para2/valid.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
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
    log_dir = work_dir + "/Logs/Cascade_ResUnet"
    checkpoints_dir = work_dir + '/checkpoints/Cascade_ResUnet'
    checkpoint_save_interval = 1
    total_epochs = 150
    load_from = work_dir + '/checkpoints/None/50.pth'

    # others
    device = 'cuda'
    dist_backend = 'nccl'
    dist_url = 'env://'
