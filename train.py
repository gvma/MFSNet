import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
import torch.nn.functional as F
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import img_as_ubyte
import skimage.io as io

from avg.avg_meter import AvgMeter
from mfsnet import MFSNet
from datasets.skin_dataset import get_loader

def clip_gradient(optimizer, grad_clip):
    
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edges = model(images)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    save_path = 'Snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'MFSNet.pth')
        print('[Saving Snapshot:]', save_path + 'MFSNet.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=20, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.05, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=25, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='train', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='MFSNet')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = MFSNet().cuda()

    # ---- flops and params ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(150):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)

    # model.eval()
    # os.makedirs('test/outputs', exist_ok=True)
    # image_root = '/home/guilherme/Documents/projects/MFSNet/data/test/images/'
    # gt_root = '/home/guilherme/Documents/projects/MFSNet/data/test/masks/'
    # test_loader = test_dataset(image_root, 1)
    
    # for i in range(test_loader.size):
    #     image, name = test_loader.load_data()

    #     image = image.cuda()

    #     print(image.size())
    #     print(name)
    #     lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)

    #     res = lateral_map_2
        
    #     res = res.sigmoid().data.cpu().numpy().squeeze()
    #     lateral_edge=lateral_edge.data.cpu().numpy().squeeze()
    #     inv_map=lateral_map_4.max()-lateral_map_4
    #     inv_map=inv_map.sigmoid().data.cpu().numpy().squeeze()
    #     lateral_map_4=lateral_map_4.sigmoid().data.cpu().numpy().squeeze()
    #     lateral_map_3=lateral_map_3.data.cpu().numpy().squeeze()
    #     lateral_map_5=lateral_map_5.data.cpu().numpy().squeeze()
    #     res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
    #     io.imsave('test/outputs'+name, img_as_ubyte(res))
