import scipy.io as sio
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def analysis_dirty_data(file):
    print(f"LOADING FILE {file}")
    data = sio.loadmat(file)["data"]#108000 66
    print(np.max(data),np.min(data),np.mean(data),np.std(data))
    # for i in range(len(data)):
    #     print(data[i][:65])

def analysis_hdemg_data():
    file = f'./dirty_data/S01/data/hdEMG.mat'
    print(f"LOADING FILE {file}")
    data = sio.loadmat(file)
    data = data['Data']
    assert data.shape == (368640, 91)  # 3组6次10s 1s采样 2048
    for i in range(len(data)):
        print(data[i][:65])



def plt_tsne():
    file = f'./clean_data/tsne.mat'
    data = sio.loadmat(file)
    data = data['data']
    x = data[:,:-1]
    y = data[:,-1:].reshape(-1)
    tsne = TSNE(n_components=2, perplexity=30)
    train_x_tsne = tsne.fit_transform(x)
    print(train_x_tsne)
    print(y)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'black', 'cyan', 'magenta']
    for i in range(len(colors)):
        ax.scatter(train_x_tsne[y == i, 0], train_x_tsne[y == i, 1], c=colors[i], label=i)
    ax.legend()
    plt.xticks([]), plt.yticks([])
    plt.show()



def analysis_norm():
    file_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    for no in file_list:
        file = f"./dirty_data/norm/S{no:02d}/hdemg_norm.mat"
        data = sio.loadmat(file)["data"][:,:65]  # 108000 66
        print(np.max(data), np.min(data), np.mean(data), np.std(data))


if __name__ == '__main__':
    #analysis_dirty_data()
    #analysis_norm()
    plt_tsne()


