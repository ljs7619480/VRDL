import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from PIL import Image
import numpy as np
import os
import csv


def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        print(e)
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', (224, 224), 'white')
    return img


class StanfordCarsDataset(Dataset):
    def __init__(self, path, imglist, train=True, transform=None, dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader
        self.datapath = path
        if train:
            with open(path+'/../../training_labels_seri.csv') as csvfile:
                cin = csv.reader(csvfile)
                self.meta = {k: v for (k, _, v) in cin}
        self.train = train
        self.imglist = imglist

    def __getitem__(self, index):
        image_id = self.imglist[index]
        image_path = os.path.join(self.datapath, image_id)
        image_id = image_id[:-4]
        img = self.dataloader(image_path)
        img = self.transform(img)
        if not self.train:
            return [img, image_id]

        label = int(self.meta[image_id])
        label = torch.LongTensor([label])
        return [img, label]

    def __len__(self):
        return len(self.imglist)


if __name__ == "__main__":
    import torchvision.transforms as transforms
    sett = StanfordCarsDataset('dataset/training_data/training_data',
                               imglist=os.listdir(
                                   'dataset/training_data/training_data'),
                               train=True,
                               transform=transforms.Compose([
                                   transforms.RandomAffine(
                                       degrees=30, translate=(0.2, 0.2), scale=(1, 1.2)),
                                   transforms.RandomPerspective(0.2, 1),
                                   # transforms.RandomCrop(224, pad_if_needed=True, padding_mode='symmetric'),
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   # transforms.RandomVerticalFlip(),
                                   # transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0)),
                                   transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                               ]))
    img = sett[1][0].numpy().transpose((1, 2, 0))
    import matplotlib.pyplot as plt
    plt.imshow(img/np.max(img))
    plt.show()
