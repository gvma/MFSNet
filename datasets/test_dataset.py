import os

from PIL import Image

import torchvision.transforms as transforms

class TestDatasetLoader:
    def __init__(self, image_root, mask_root, testsize):
          self.testsize = testsize
          self.images = [(image_root + f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
          self.images = sorted(self.images)
          self.masks = [(mask_root + f) for f in os.listdir(mask_root) if f.endswith('.jpg') or f.endswith('.png')]
          self.masks = sorted(self.masks)
          # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
          # self.gts = sorted(self.gts)
          self.img_transform = transforms.Compose([
              transforms.Resize((self.testsize, self.testsize)),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])
          self.mask_transform = transforms.Compose([
              transforms.Resize((self.testsize, self.testsize)),
              transforms.ToTensor()])
          # self.gt_transform = transforms.ToTensor()
          self.size = len(self.images)
          self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        mask = self.binary_loader(self.masks[self.index])
        mask = self.mask_transform(mask).unsqueeze(0)
        
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        return image, mask, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')