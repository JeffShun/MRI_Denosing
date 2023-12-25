
import random
import torch

"""
数据预处理工具
1、所有数据预处理函数都包含两个输入: img 、label
2、img、label的输入维度为3维[C,H,W]，第一个维度是通道数
"""

class TransformCompose(object):

    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class to_tensor(object):
    def __call__(self, img, label):
        img_o = torch.from_numpy(img)
        label_o = torch.from_numpy(label)
        return img_o, label_o

class random_flip(object):
    def __init__(self, axis=1, prob=0.5):
        assert isinstance(axis, int) and axis in [1,2]
        self.axis = axis
        self.prob = prob

    def __call__(self, img, label):
        img_o, label_o = img, label
        if random.random() < self.prob:
            img_o = torch.flip(img, [self.axis])
            label_o = torch.flip(label, [self.axis])
        return img_o, label_o

class random_rotate90(object):
    def __init__(self, k=1, prob=0.5):
        assert isinstance(k, int) and k in [1,2,3]
        self.k = k
        self.prob = prob

    def __call__(self, img, label):
        img_o, label_o = img, label
        if random.random() < self.prob:
            img_o = torch.rot90(img, self.k, [1, 2])
            label_o = torch.rot90(label, self.k, [1, 2])
        return img_o, label_o

class random_crop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, label):
        d, w = img.shape[1:]
        crop_x_start = random.randint(0, max(0, d - self.crop_size[0]))
        crop_y_start = random.randint(0, max(0, w - self.crop_size[1]))
        img_o = img[:, crop_x_start:crop_x_start+self.crop_size[0], crop_y_start:crop_y_start+self.crop_size[1]]
        label_o = label[:, crop_x_start:crop_x_start+self.crop_size[0], crop_y_start:crop_y_start+self.crop_size[1]]            
        return img_o, label_o


class resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, label):
        if tuple(img.shape[1:]) == tuple(label.shape[1:]) and tuple(img.shape[1:]) == tuple(self.size):
            return img, label
        img_o = torch.nn.functional.interpolate(img[None], size=self.size, mode="trilinear") 
        label_o = torch.nn.functional.interpolate(label[None], size=self.size, mode="trilinear")
        img_o = img_o.squeeze(0)
        label_o = label_o.squeeze(0)
        return img_o, label_o
    
class normlize(object):
    def __init__(self, win_clip=None):
        self.win_clip = win_clip

    def __call__(self, img, label):  
        if self.win_clip is not None:
            img = torch.clip(img, self.win_clip[0], self.win_clip[1])
            label = torch.clip(label, self.win_clip[0], self.win_clip[1])
        img_o = self._norm(img)
        label_o = self._norm(label)      
        return img_o, label_o
    
    def _norm(self, img):
        ori_shape = img.shape
        img_flatten = img.reshape(ori_shape[0], -1)
        img_min = img_flatten.min(dim=-1,keepdim=True)[0]
        img_max = img_flatten.max(dim=-1,keepdim=True)[0]
        img_norm = (img_flatten - img_min)/(img_max - img_min)
        img_norm = img_norm.reshape(ori_shape)
        return img_norm