from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import glob

root = './archive/processed'


class dataset(Dataset):
    def __init__(self, mode, transform=None):
        self.root = './archive/processed'
        self.x = f'x_{mode}'
        self.y = f'y_{mode}'
        self.xdirec = os.path.join(self.root, self.x)
        self.ydirec = os.path.join(self.root, self.y)

        self.input = glob.glob(self.xdirec + '*/*.*')
        self.output = glob.glob(self.ydirec + '*/*.*')
        self.transform = transforms.Compose(transform)

    def __getitem__(self, index):
        self.a = np.load(self.input[index])
        self.b = np.load(self.output[index])

        self.a = self.a.astype(np.float)
        self.b = self.b.astype(np.float)

        self.a = self.transform(self.a)
        self.b = self.transform(self.b)
        #         self.a=self.a[0,:,:].unsqueeze(0)
        #         self.b=self.b[0,:,:].unsqueeze(0)

        return {'input': self.a, 'output': self.b}

    def __len__(self):
        return len(self.input)

transform = [
    #     transforms.ToPILImage(),
    #     transforms.Grayscale(),
    transforms.ToTensor(),
]

train_dataset = dataset('train', transform=transform)
validation_dataset = dataset('val', transform=transform)
test_dataset = dataset('test', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

