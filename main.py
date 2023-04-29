import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from deal_data import get_dataloader,GestureDataset
from net import Classifier
from torch.utils.data import DataLoader


def valid(model, valid_loader, device):
    model.eval()
    with torch.no_grad():
        for index, (gesture, label) in enumerate(valid_loader):
            if index>3:break
            gesture, label = gesture.to(device), label.to(device)
            output = model(gesture)
            loss = criterion(output, label)
            accuracy = torch.mean((output.argmax(1) == label).float()).item()
            print(f"[VALID] [NUM {idx:0>4d}][accuracy:{accuracy:.2f}][loss {loss.sum().item():.4f}]")

def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for index, (gesture, label) in enumerate(test_loader):
            if index>3:break
            gesture, label = gesture.to(device), label.to(device)
            output = model(gesture)
            loss = criterion(output, label)
            accuracy = torch.mean((output.argmax(1) == label).float()).item()
            print(f"[TEST] [NUM {idx:0>4d}][accuracy:{accuracy:.2f}][loss {loss.sum().item():.4f}]")

def test_by_gesture(model,device):
    model.eval()
    for i in range(1,13):
        _testset = GestureDataset(f"G{i:02d}.mat")
        _loader = DataLoader(_testset, batch_size=1000,shuffle=True, drop_last=True,pin_memory=True)
        with torch.no_grad():
            for index, (gesture, label) in enumerate(_loader):
                gesture, label = gesture.to(device), label.to(device)
                output = model(gesture)
                loss = criterion(output, label)
                accuracy = torch.mean((output.argmax(1) == label).float()).item()
                print(f"[TEST BY GESTURE {i:02d}] [NUM {idx:0>4d}][accuracy:{accuracy:.2f}][loss {loss.sum().item():.4f}]")
                break

def test_by_subject(model,device):
    model.eval()
    for i in [18,19,20]:
        _testset = GestureDataset(f"test{i}.npy")
        _loader = DataLoader(_testset, batch_size=1000,shuffle=True, drop_last=True,pin_memory=True)
        with torch.no_grad():
            for index, (gesture, label) in enumerate(_loader):
                gesture, label = gesture.to(device), label.to(device)
                output = model(gesture)
                loss = criterion(output, label)
                accuracy = torch.mean((output.argmax(1) == label).float()).item()
                print(f"[TEST BY SUBJECT {i:02d}] [NUM {idx:0>4d}][accuracy:{accuracy:.2f}][loss {loss.sum().item():.4f}]")
                break

def save_model(model,name):
    path = f"./model/{name}.pkl"
    torch.save(model.state_dict(), path)

def load_model(model,name):
    path = f"./model/{name}.pkl"
    if os.path.exists(path):model.load_state_dict(torch.load(path))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader = get_dataloader()
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-4)
    epoch_num = 100
    name = "S01_S20_transfomer_hdemg"
    # load_model(model,name)
    for epoch in range(epoch_num):
        for idx, (gesture, label) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            gesture, label = gesture.to(device), label.to(device)
            output = model(gesture)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            accuracy = torch.mean((output.argmax(1) == label).float()).item()
            if idx and idx%10==0:
                print(f"[TRAIN {epoch:0>3d}/{epoch_num:0>3d}][NUM {idx:0>3d}][accuracy:{accuracy:.4f}][loss {loss.sum().item():.4f}]")
            # if idx and idx % 100 == 0: valid(model, valid_loader, device)
            # if idx and idx % 100 == 0: test(model, test_loader, device)
            if idx and idx % 100 == 0: test_by_subject(model, device)
        save_model(model,name)
