import random
import math
import numbers
import collections
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.from_numpy(np.array(pic, np.uint8))
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class Scale(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.abc.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size[::-1], self.interpolation)

    def randomize_parameters(self):
        pass


class RandomHorizontalFlip(object):

    def __call__(self, img):
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.p = random.random()


class RandomCrop(object):
    def __init__(self, size, p=0.5, interpolation=Image.BILINEAR):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.height, self.width = self.size
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if not self.cropping:
            return img.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(self.tl_x * x_maxrange))
        y1 = int(round(self.tl_y * y_maxrange))
        return resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))

    def randomize_parameters(self):
        self.cropping = random.uniform(0, 1) < self.p
        self.tl_x = random.random()
        self.tl_y = random.random()


class RandomErasing(object):
    """ 
    Randomly selects a rectangle region in an image and erases its pixels.

    Reference:
        Zhong et al. Random Erasing Data Augmentation. arxiv: 1708.04896, 2017.
        
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, height=256, width=128, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.height = height
        self.width = width
       
    def __call__(self, img):
        if self.re:
            return img

        if img.size()[0] == 3:
            img[0, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[0]
            img[1, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[1]
            img[2, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[2]
        else:
            img[0, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[0]
        return img

    def randomize_parameters(self):
        self.re = random.uniform(0, 1) < self.probability
        self.h, self.w, self.x1, self.y1 = 0, 0, 0, 0
        whether_re = False
        if self.re:
            for attempt in range(100):
                area = self.height*self.width

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)

                self.h = int(round(math.sqrt(target_area * aspect_ratio)))
                self.w = int(round(math.sqrt(target_area / aspect_ratio)))
                if self.w < self.width and self.h < self.height:
                    self.x1 = random.randint(0, self.height - self.h)
                    self.y1 = random.randint(0, self.width - self.w)
                    whether_re = True
                    break

        self.re = whether_re