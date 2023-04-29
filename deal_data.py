import numpy as np
import torch
import scipy.io as sio

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


class GestureDataset(Dataset):
    def __init__(self,file):
        if isinstance(file,str):
            self.data = sio.loadmat(f"./clean_data/{file}")['data']
        elif isinstance(file,list):
            np_list = []
            for f in file:np_list.append(sio.loadmat(f"./clean_data/{f}")['data'])
            self.data = np.concatenate(np_list)
            # print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx_data = self.data[index]
        # print(idx_data)
        gesture = torch.FloatTensor(idx_data[:-1]).reshape(-1,65)
        label = torch.LongTensor(idx_data[-1:])[0]
        return gesture, label  # [65] [1]

def get_dataloader():
    dataset = GestureDataset("gesture.mat")  #
    # print(len(dataset))
    train_num = int(0.8 * len(dataset))
    lengths = [train_num, len(dataset) - train_num]
    trainset, validset = random_split(dataset, lengths)
    # print(len(trainset),len(validset))
    """
    drop_last：告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留 
    pin_memory: 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存
    collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能。
    """
    train_loader = DataLoader(trainset, batch_size=512, shuffle=True, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(validset, batch_size=5000,shuffle=True, drop_last=True, pin_memory=True)
    testset = GestureDataset("test.mat")
    test_loader = DataLoader(testset, batch_size=5000,shuffle=True, drop_last=True, pin_memory=True)
    return train_loader, valid_loader,test_loader


if __name__ == '__main__':
    dataset = GestureDataset()
    print(len(dataset), dataset[0])
    # train_loader, valid_loader,test_loader = get_dataloader()
    # # print(len(train_loader),len(valid_loader))
