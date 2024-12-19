import sys

from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy

def read_image(img_list):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if type(img_list) ==type("This is a str"):
        img_path = img_list
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                #判断图像的宽度是256的几倍，如果是三倍，则进行下面的代码，否则只裁剪前两个
                img3 = [img.crop((256 * i, 0, 256 * (i + 1), 128)) for i in range(img.size[0] // 256)]
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
    else:
        img3 = []
        for i in img_list:
            img_path = i
            got_img = False
            if not osp.exists(img_path):
                raise IOError("{} does not exist".format(img_path))
            while not got_img:
                try:
                    img = Image.open(img_path)
                    img3.append(img)
                    got_img = True
                except IOError:
                    print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                    pass
    return img3


def read_image_(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks =[], [], []

        for _, pid, camid,trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]

        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)

        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)

        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  image_train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  image_query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor
class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
        normalize_transform = T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        # self.transformT = T.Compose([
        #     T.Resize((256,128), interpolation = 'bicubic'),
        #     # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        #     T.ToTensor(),
        #     normalize_transform,
        #     # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        # ])
        # print(self.transform)
        # print(self.transformT)
        # raise


    def __len__(self):
        return len(self.dataset)

    def get_exposure_fake_label(self,r,n,t):
        r = to_tensor(r)
        n = to_tensor(n)
        rhist = torch.histc(r)
        nhist = torch.histc(n)
        # thist = torch.histc(t)
        ratior = torch.sum(rhist[98:])/torch.sum(rhist)
        ration =torch.sum(nhist[98:])/torch.sum(nhist)
        if ratior.ge(0.095) and ration.ge(0.095):
            overexposure = 1
        else:
            overexposure = 0
        return overexposure

   
        
    def __getitem__(self, index):
        img_path, pid, camid,trackid = self.dataset[index] # original
        imgs = read_image(img_path)

        # img1 = Image.open(img_path[0]).convert('RGB')
        # img2 = Image.open(img_path[1]).convert('RGB')
        # img3 = Image.open(img_path[2]).convert('RGB')
        img1 = imgs[0]
        img2 = imgs[1]
        img3 = imgs[2]
        exposure_label = self.get_exposure_fake_label(img1,img2,img3)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)


        return img1, img2, img3, pid, camid,trackid, img_path, exposure_label
        