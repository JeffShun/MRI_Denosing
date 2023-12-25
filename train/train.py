import warnings
warnings.filterwarnings('ignore')
import os
from config.model_config import network_cfg
import torch
from torch import optim
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
import time
from custom.utils.common_tools import *
from tensorboardX import SummaryWriter

# os.environ['CUDA_VISIBLE_DEVICES'] = network_cfg.gpus
logger_dir = network_cfg.log_dir
os.makedirs(logger_dir, exist_ok=True)
logger = Logger(logger_dir+"/trainlog.txt", level='debug').logger
writer = SummaryWriter(logger_dir)
create_tar_archive("./", logger_dir+"/project.tar")

def train():
    net = network_cfg.network.cuda()
    # 定义损失函数
    train_loss_f = network_cfg.train_loss_f
    valid_loss_f = network_cfg.valid_loss_f
    if os.path.exists(network_cfg.load_from):
        net.load_state_dict(torch.load(network_cfg.load_from))
    net.train()
    train_dataset = network_cfg.train_dataset
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=network_cfg.batchsize, 
                                shuffle=network_cfg.shuffle,
                                num_workers=network_cfg.num_workers, 
                                drop_last=network_cfg.drop_last)
    valid_dataset = network_cfg.valid_dataset
    valid_dataloader = DataLoader(valid_dataset, 
                                batch_size=network_cfg.batchsize, 
                                shuffle=False,
                                num_workers=network_cfg.num_workers, 
                                drop_last=network_cfg.drop_last)
    
    optimizer = optim.Adam(params=net.parameters(), lr=network_cfg.lr, weight_decay=network_cfg.weight_decay)
    scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                milestones=network_cfg.milestones,
                                gamma=network_cfg.gamma,
                                warmup_factor=network_cfg.warmup_factor,
                                warmup_iters=network_cfg.warmup_iters,
                                warmup_method=network_cfg.warmup_method,
                                last_epoch=network_cfg.last_epoch)
    time_start=time.time()
    for epoch in range(network_cfg.total_epochs): 
        #Training Step!
        for ii, (src_img, tgt_img) in enumerate(train_dataloader):
            src_img = V(src_img).cuda()
            tgt_img = V(tgt_img).cuda()
            optimizer.zero_grad()
            t_out = net(src_img)
            t_loss = train_loss_f(t_out, tgt_img)
            loss_all = V(torch.zeros(1)).cuda()
            loss_info = ""
            for loss_item, loss_val in t_loss.items():
                loss_all += loss_val
                loss_info += "{}={:.4f}\t ".format(loss_item,loss_val.item())
                writer.add_scalar('TrainLoss/{}'.format(loss_item),loss_val.item(), epoch*len(train_dataloader)+ii+1)
            time_temp=time.time()
            eta = ((network_cfg.total_epochs-epoch)+(1-(ii+1)/len(train_dataloader)))/(epoch+(ii+1)/len(train_dataloader))*(time_temp-time_start)/60
            if eta < 60:
                eta = "{:.1f}min".format(eta)
            else:
                eta = "{:.1f}h".format(eta/60.0)
            logger.info('Epoch:[{}/{}]\t Iter:[{}/{}]\t Eta:{}\t {}'.format(epoch+1 ,network_cfg.total_epochs, ii+1, len(train_dataloader), eta, loss_info))
            loss_all.backward()
            optimizer.step()
        writer.add_scalar('LR', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        scheduler.step()

        # Valid Step!
        if (epoch+1) % network_cfg.valid_interval == 0:
            valid_loss = dict()
            for ii, (src_img, tgt_img) in enumerate(valid_dataloader):
                src_img = V(src_img).cuda()
                tgt_img = V(tgt_img).cuda()
                with torch.no_grad():
                    v_out = net(src_img)
                    v_loss = valid_loss_f(v_out, tgt_img)
                loss_all = V(torch.zeros(1)).cuda()
                for loss_item, loss_val in v_loss.items():
                    if loss_item not in valid_loss:
                        valid_loss[loss_item] = loss_val
                    else:
                        valid_loss[loss_item] += loss_val  
            loss_info = ""              
            for loss_item, loss_val in valid_loss.items():
                valid_loss[loss_item] /= (ii+1)
                loss_info += "{}={:.4f}\t ".format(loss_item,valid_loss[loss_item])
                writer.add_scalar('ValidLoss/{}'.format(loss_item),valid_loss[loss_item], (epoch+1)*len(train_dataloader))
            logger.info('Validating Step:\t {}'.format(loss_info))
            
            
        os.makedirs(network_cfg.checkpoints_dir,exist_ok=True)
        if (epoch+1) % network_cfg.checkpoint_save_interval == 0:
            torch.save(net.state_dict(), network_cfg.checkpoints_dir+"/{}.pth".format(epoch+1))
    writer.close()

if __name__ == '__main__':
	train()
