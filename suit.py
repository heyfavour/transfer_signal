import os
import torch
from torch import nn
from deal_data import GestureDataset
from torch.utils.data import DataLoader, random_split
from net import Classifier
from torch.optim import AdamW


def get_dataloader():
    dataset = GestureDataset(["G01.mat","G02.mat","G03.mat","G04.mat","G05.mat","G06.mat","G07.mat","G08.mat","G09.mat","G10.mat","G11.mat","G12.mat"])
    train_num = int(0.05 * len(dataset))
    lengths = [train_num, len(dataset) - train_num]
    trainset, validset = random_split(dataset, lengths)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(validset, batch_size=1000, shuffle=True, drop_last=True, pin_memory=True)
    print(len(train_loader),len(valid_loader))
    return train_loader, valid_loader


def test(model, file, device):
    model.eval()
    dataset = GestureDataset(file)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True, drop_last=True, pin_memory=True)
    with torch.no_grad():
        for index, (gesture, label) in enumerate(dataloader):
            gesture, label = gesture.to(device), label.to(device)
            output = model(gesture)
            loss = criterion(output, label)
            accuracy = torch.mean((output.argmax(1) == label).float()).item()
            print(f"[{file.upper()}] [NUM {idx:0>4d}][accuracy:{accuracy:.2f}][loss {loss.sum().item():.4f}]")
            break

def valid(model, dataloader, device):
    with torch.no_grad():
        for index, (gesture, label) in enumerate(dataloader):
            gesture, label = gesture.to(device), label.to(device)
            output = model(gesture)
            loss = criterion(output, label)
            accuracy = torch.mean((output.argmax(1) == label).float()).item()
            print(f"[VALID] [NUM {idx:0>4d}][accuracy:{accuracy:.2f}][loss {loss.sum().item():.4f}]")
            break


def load_model(model, name):
    path = f"./model/{name}.pkl"
    if os.path.exists(path): model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = get_dataloader()
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    epoch_num = 100000
    name = "S01_S06_timeseq_hdemg"
    load_model(model, name)
    for epoch in range(epoch_num):
        model.train()
        for idx, (gesture, label) in enumerate(train_loader):
            optimizer.zero_grad()
            gesture, label = gesture.to(device), label.to(device)
            output = model(gesture)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            accuracy = torch.mean((output.argmax(1) == label).float()).item()
            if idx and (idx+1) % 5 == 0:
                print(f"[TRAIN {epoch:0>3d}/{epoch_num:0>3d}][NUM {idx:0>3d}][accuracy:{accuracy:.4f}][loss {loss.sum().item():.4f}]")
            if idx and (idx+1) % 5 == 0:
                valid(model,valid_loader,device)
                test(model, "gesture", device)
                for i in range(1,13,1):test(model,f"G{i:02d}",device)
