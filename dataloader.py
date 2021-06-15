import torch
import os
import numpy as np
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, noise_scale=0):
        super(Dataset, self).__init__()
        img_folder = './dataset/%s_img/' % phase
        label_folder = './dataset/%s_label' % phase

        self.noise_scale = noise_scale

        file_names = os.listdir(img_folder)
        file_names = sorted(file_names, key=lambda x: int(x.replace('.png', '')))
        print(file_names)
        images = []
        labels = []

        for fn in file_names:
            img = os.path.join(img_folder, fn)
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)

        for fn in file_names:
            label = os.path.join(label_folder, fn)
            label = cv2.imread(label)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            label = np.where(label > 0, 2, 1)
            # if phase == 'train':
            #     assert np.all(label == np.load('./dataset/train_label/%s.npy' % fn))
            if phase == 'test' and fn == '0.png':
                print('Rotation 180 degree for the label of test set 0.png')
                label = label[::-1, ::-1]
            labels.append(label)

        self.num_img = len(images)
        self.images = np.stack(images, 0)
        self.labels = np.stack(labels, 0)
        print('# number of %s images: %d' % (phase, self.num_img))
        print('# images %s' % list(self.images.shape))
        print('# labels %s' % list(self.labels.shape))
        print('Load %s dataset complete' % phase)

        self.labels -= 1
        self.loss_weight = [np.mean(self.labels == i) for i in range(2)]
        self.loss_weight = -np.log(self.loss_weight)
        if phase == 'train':
            print('# Loss weight %s' % self.loss_weight.tolist())

    def __len__(self):
        return self.num_img

    def __getitem__(self, item):
        image = self.images[item]
        image = image + np.random.randn(*image.shape) * self.noise_scale
        label = self.labels[item]
        return image, label


def get_dataloader(phase, bz, noise_scale=0):
    dataset = Dataset(phase, noise_scale)
    shuffle = (phase == 'train')
    return torch.utils.data.DataLoader(dataset, batch_size=bz, num_workers=4, shuffle=shuffle)
