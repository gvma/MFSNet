import torch
import os, argparse
import torch.utils.data as data
import skimage.io as io
from skimage import img_as_ubyte
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import jaccard_score

from models.mfsnet import MFSNet
from models.resnet import res2net50_v1b_26w_4s
from datasets.test_dataset import TestDatasetLoader

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}

if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2net50_v1b_26w_4s(pretrained=True)
    model = model.cuda(0)

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--model_path', type=str, default='./Snapshots/MFSNet/MFSNet.pth')
parser.add_argument('--data_path', type=str, default='test', help='Directory of test images')
parser.add_argument('--save_path', type=str, default='test/outputs', help='Directory where prediction masks will be saved.')


opt = parser.parse_args()
data_path = opt.data_path
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

model = MFSNet()
model.load_state_dict(torch.load(opt.model_path))
model.cuda()
model.eval()

os.makedirs(save_path, exist_ok=True)
image_root = '{}/images/'.format(data_path)
gt_root = '{}/masks/'.format(data_path)
test_loader = TestDatasetLoader(image_root, opt.testsize)

for i in range(test_loader.size):
        image, mask, name = test_loader.load_data()
        mask_im_arr = np.array(mask)
        image_cpu = image.numpy().squeeze()

        image = image.cuda()

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)

        res = lateral_map_2
        
        res = res.sigmoid().data.cpu().numpy().squeeze()
        lateral_edge=lateral_edge.data.cpu().numpy().squeeze()
        inv_map=lateral_map_4.max()-lateral_map_4
        inv_map=inv_map.sigmoid().data.cpu().numpy().squeeze()
        lateral_map_4=lateral_map_4.sigmoid().data.cpu().numpy().squeeze()
        lateral_map_3=lateral_map_3.data.cpu().numpy().squeeze()
        lateral_map_5=lateral_map_5.data.cpu().numpy().squeeze()
        
        image_cpu = img_as_ubyte((image_cpu - image_cpu.min()) / (image_cpu.max() - image_cpu.min() + 1e-8))
        res = img_as_ubyte((res - res.min()) / (res.max() - res.min() + 1e-8))
        
        # (3, 352, 352)
        # (352, 352)

        print(jaccard_score(image_cpu[0], res, average='micro'))
        print('Salvando em ' + save_path+name)
        io.imsave(save_path+'/'+name, res)
