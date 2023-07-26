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
from torchmetrics import Dice
from sklearn.metrics import jaccard_score

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
    # save_path = 'Snapshots/{}/'.format(opt.train_save)
    # os.makedirs(save_path, exist_ok=True)
    # if (epoch+1) % 10 == 0:
    #     torch.save(model.state_dict(), save_path + 'MFSNet.pth')
    #     print('[Saving Snapshot:]', save_path + 'MFSNet.pth')

def test(test_loader, model):
    print
    best_jaccard_score = -1
    best_dice_score = -1
    avg_dice_score = 0
    avg_jaccard_score = 0

    for i, pack in enumerate(test_loader, start=1):
            image, mask =  pack
            mask_im_arr = np.array(mask)
            mask_im_arr = np.reshape(mask_im_arr, (352, 352))
            mask_im_arr = mask_im_arr.flatten()
            mask_im_arr = mask_im_arr.astype(np.uint8)

            mask = mask.cuda(0)
            mask = mask.type(torch.int64)

            image_cpu = image.numpy().squeeze()

            image = image.cuda()

            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)

            res = lateral_map_2
            
            res_gpu = res.sigmoid().data
            res_gpu[res_gpu >= 0.5] = 1
            res_gpu[res_gpu < 0.5] = 0
            
            res = res.sigmoid().data.cpu().numpy().squeeze()
            lateral_edge=lateral_edge.data.cpu().numpy().squeeze()
            inv_map=lateral_map_4.max()-lateral_map_4
            inv_map=inv_map.sigmoid().data.cpu().numpy().squeeze()
            lateral_map_4=lateral_map_4.sigmoid().data.cpu().numpy().squeeze()
            lateral_map_3=lateral_map_3.data.cpu().numpy().squeeze()
            lateral_map_5=lateral_map_5.data.cpu().numpy().squeeze()

            res[res >= 0.5] = 1
            res[res < 0.5] = 0
            res = res.flatten()
            res = res.astype(np.uint8)
            
            j_score = jaccard_score(mask_im_arr, res, average='micro')
            if j_score > best_jaccard_score:
                best_jaccard_score = j_score
            
            dice = Dice(average='micro').to(torch.device("cuda", 0))
            d_score = dice(res_gpu, mask)
            dice_score = d_score.item()
            if dice_score > best_dice_score:
                best_dice_score = dice_score
            
            avg_dice_score += dice_score
            avg_jaccard_score += j_score
    print('Average Dice score: ', avg_dice_score/len(test_loader))
    print('Average Jaccard socre: ', avg_jaccard_score/len(test_loader))
    return avg_dice_score, avg_jaccard_score

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
                        default='data/train', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='MFSNet')
    parser.add_argument('--test_path', type=str,
                        default='data/test')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = MFSNet().cuda()

    # ---- flops and params ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root_train= '{}/images/'.format(opt.train_path)
    gt_root_train = '{}/masks/'.format(opt.train_path)

    image_root_test= '{}/images/'.format(opt.test_path)
    gt_root_test = '{}/masks/'.format(opt.test_path)

    train_loader = get_loader(image_root_train, gt_root_train, batchsize=opt.batchsize, trainsize=opt.trainsize)
    test_loader = get_loader(image_root_test, gt_root_test,batchsize=opt.batchsize,trainsize=opt.trainsize)
    total_step = len(train_loader)
    eval_step = 10
    print("#"*20, "Start Training", "#"*20)

    save_path = "weights/"
    best_dice = 0
    for epoch in range(150):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
        if epoch % eval_step == 0:
            print("#"*20, "Start Eval", "#"*20)
            dice, jaccard = test(test_loader, model)
            if dice > best_dice:
                best_dice = dice
                torch.save(model.state_dict(), save_path + 'best_epoch.pth')
                print('[Saving Actual Best Model:]', save_path + 'best_epoch.pth')   
