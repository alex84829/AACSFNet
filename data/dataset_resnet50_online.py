import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import random
# from data.video_transforms import *
from config import opt
from PIL import Image


# online
class UNBCdataset(Dataset):
    def __init__(self, id, mode="train"):
        super(UNBCdataset, self).__init__()

        mean = [0.48766823, 0.34021825, 0.30166676]
        std = [0.15309834, 0.13826898, 0.12866308]

        if mode == "train":
            self.flag = "train"
            if id == -1:
                raise (" validatation_number can not be -1! ")

            random_order = random.sample(range(0, 25), 25)  # 随机25个(0, 25)范围内不重复的数字
            # random_order = random.sample(range(0,5), 5)  # 随机25个(0, 25)范围内不重复的数字，测试使用
            #
            random_order.remove(id)
            # random_order.remove(1)

            train_images = np.load('./data/data_pre/%d/%d_image.npy' % (random_order[0], random_order[0]),
                                   allow_pickle=True)
            train_labels = np.load('./data/data_pre/%d/%d_label1.npy' % (random_order[0], random_order[0]),
                                   allow_pickle=True)

            for i in random_order[0:]:
                images = np.load('./data/data_pre/%d/%d_image.npy' % (i, i), allow_pickle=True)
                train_images = np.vstack((train_images, images))

                labels = np.load('./data/data_pre/%d/%d_label1.npy' % (i, i), allow_pickle=True)
                train_labels = np.concatenate((train_labels, labels))

            self.images = train_images
            self.labels = train_labels

            self.transform = T.Compose([
                T.Resize(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

        else:
            self.flag = "val"
            self.images = np.load('./data/data_pre/%d/%d_image.npy' % (id, id),
                                  allow_pickle=True)
            self.labels = np.load('./data/data_pre/%d/%d_label1.npy' % (id, id),
                                  allow_pickle=True)

            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize(224),
                T.Normalize(mean=mean, std=std),
            ])

        self.num_frames = 16
        self.interval = 1
        # self.samples_weight = self.make_weights_for_balanced_classes()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        label = self.labels[index]
        img = self.images[index]
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = self.transform(img)

        return img, int(label)


if __name__ == '__main__':
    data = UNBCdataset(id=0, mode="test")
    dataloader = DataLoader(data, batch_size=32, shuffle=False)

    for idx, (a, b, c) in enumerate(dataloader):
        import pdb
        pdb.set_trace()
