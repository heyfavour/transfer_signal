import gc
import time

import scipy.io as sio
import numpy as np


def normalization(data):
    _range = np.max(data[:, :-1]) - np.min(data[:, :-1]).astype(float)
    data[:, :-1] = (data[:, :-1] - np.min(data[:, :-1])) / _range
    return data


def standardization(data):
    data[:, :-1] = (data[:, :-1] - np.mean(data[:, :-1])) / np.std(data[:, :-1])
    return data


def train_data(time_window):
    start, end = time_window
    clean_data = []
    for j in range(1, 4):
        for i in range(12):
            file = f'./dirty_data/S0{j}/G{(i + 1):02d}.mat'
            print(f"LOADING FILE {file}")
            data = sio.loadmat(file)
            data = data['Data']
            assert data.shape == (368640, 91)  # 3组6次10s 1s采样 2048
            for t in range(18):  # 18次
                start_step = int(t * 10 * 2048 + start * 2048)
                end_step = int((t + 1) * 10 * 2048 - (10 - end) * 2048)
                # new_x = data[start_step:end_step, :65]
                row = []
                for i in range(0, 64): row.append(data[start_step + i:end_step + i, :65])
                new_x = np.concatenate(row, axis=1)
                row.clear()

                y = np.ones((len(new_x), 1), dtype=np.int64) * i
                new_x = np.c_[new_x, y]
                clean_data.append(new_x)
                del row, new_x, y
                gc.collect()
                print(f"[{j}][{t}]===============")
        # subject_data = np.concatenate(subject_data)
        # subject_data = np.abs(subject_data)
        # subject_data = normalization(subject_data)
        # all_subject.append(subject_data)
    # merged_normalization = np.concatenate(all_subject)
    np_array = np.concatenate(clean_data)
    np_array = np.abs(np_array)
    np_array = standardization(np_array)
    print(np_array[0])
    print(np.shape)
    sio.savemat('clean_data/gesture.mat', {'data': np_array})  # Saving .mat File of MYO


def test_data(time_window):
    start, end = time_window
    clean_data = []
    for i in range(12):
        file = f'./dirty_data/S04/G{(i + 1):02d}.mat'
        print(f"LOADING FILE {file}")
        data = sio.loadmat(file)
        data = data['Data']
        assert data.shape == (368640, 91)  # 3组6次10s 1s采样 2048
        for t in range(18):  # 18次
            start_step = int(t * 10 * 2048 + start * 2048)
            end_step = int((t + 1) * 10 * 2048 - (10 - end) * 2048)
            new_x = data[start_step:end_step, :65]
            y = np.ones((len(new_x), 1), dtype=np.int64) * i
            new_x = np.c_[new_x, y]
            clean_data.append(new_x)
    np_array = np.concatenate(clean_data)
    np_array = np.abs(np_array)
    np_array = standardization(np_array)
    # subject_data = np.abs(subject_data)
    # merged_normalization = normalization(subject_data)
    sio.savemat('clean_data/test.mat', {'data': np_array})  # Saving .mat File of MYO


def test_by_gesture_data(time_window):
    start, end = time_window
    for i in range(12):
        clean_data = []
        file = f'./dirty_data/S/G{(i + 1):02d}.mat'
        print(f"LOADING FILE {file}")
        data = sio.loadmat(file)
        data = data['Data']
        assert data.shape == (368640, 91)  # 3组6次10s 1s采样 2048
        for t in range(18):  # 18次
            start_step = int(t * 10 * 2048 + start * 2048)
            end_step = int((t + 1) * 10 * 2048 - (10 - end) * 2048)
            new_x = data[start_step:end_step, :65]
            y = np.ones((len(new_x), 1), dtype=np.int64) * i
            new_x = np.c_[new_x, y]
            clean_data.append(new_x)
        merged_array = np.concatenate(clean_data)
        merged_abs = np.abs(merged_array)
        merged_normalization = normalization(merged_abs)
        sio.savemat(f'clean_data/test{i + 1:02d}.mat', {'data': merged_normalization})  # Saving .mat File of MYO


def wash_hdemg_data(time_window, out, dirty_list):
    # 清洗hdEMG数据
    start, end = time_window
    clean_data = []
    for j in dirty_list:
        file = f'./dirty_data/data/S{j:02d}/hdEMG.mat'
        print(f"LOADING FILE {file}")
        data = sio.loadmat(file)  # 3组6次10s 1s采样 300
        x = data['x']  # (108000, 65)
        y = data['y_static'].reshape(-1, 1) - 1  # (1, 108000)
        for t in range(0, 108000, 500):  # 216=3组6次12个
            start_step = t + int((500 / 10) * start)
            end_step = t + int((500 / 10) * end)
            # print(start_step,end_step)
            # split_x = x[start_step:end_step, :]
            split_y = y[start_step:end_step, :]
            row = []
            for i in range(0, 5): row.append(x[start_step + i:end_step + i, :])
            split_x = np.concatenate(row, axis=1)
            row.clear()
            new_x = np.c_[split_x, split_y]
            clean_data.append(new_x)
        # print(np.max(x),np.min(x),np.mean(x),np.std(x))
        # clean_data.append(np.c_[x, y])
    train_data = np.concatenate(clean_data)
    sio.savemat(f'clean_data/{out}.mat', {'data': train_data})  # Saving .mat File of MYO


def wash_norm_hdemg_data(time_window, out, dirty_list):
    # 清洗hdEMG_norm数据
    start, end = time_window
    clean_data = []
    for j in dirty_list:
        file = f'./dirty_data/norm/S{j:02d}/hdEMG_norm.mat'
        print(f"LOADING FILE {file}")
        data = sio.loadmat(file)["data"]  # 108000 66
        for t in range(0, 108000, 500):  # 18次
            start_step = t + int((500 / 10) * start)
            end_step = t + int((500 / 10) * end)
            split_y = data[start_step:end_step, -1:] - 1
            row = []
            for i in range(0, 5): row.append(data[start_step + i:end_step + i, :65])
            split_x = np.concatenate(row, axis=1)
            new_x = np.c_[split_x, split_y]
            clean_data.append(new_x)
    train_data = np.concatenate(clean_data)
    sio.savemat(f'clean_data/{out}.mat', {'data': train_data})  # Saving .mat File of MYO


def wash_tsne_hdemg_data(time_window, prefix, suffix, dirty_list):
    # 该数据lr=1e-4跑批合适
    start, end = time_window
    clean_data = []
    for j in dirty_list:
        file = f'{prefix}/S{j:02d}/{suffix}.mat'
        print(f"LOADING FILE {file}")
        data = sio.loadmat(file)  # 3组6次10s 1s采样 300
        if "norm" in suffix:
            x = data['data'][:, :65]
            y = data['data'][:, -1:].reshape(-1, 1) - 1
        else:
            x = data['x']  # (108000, 65)
            y = data['y_static'].reshape(-1, 1) - 1  # (1, 108000)
        for t in range(0, 108000, 500):  # 216=3组6次12个
            start_step = t + int((500 / 10) * start)
            end_step = t + int((500 / 10) * end)
            split_x = x[start_step:end_step, :]
            split_y = y[start_step:end_step, :]
            new_x = np.c_[split_x, split_y]
            clean_data.append(new_x)
        # print(np.max(x),np.min(x),np.mean(x),np.std(x))
        # clean_data.append(np.c_[x, y])
    train_data = np.concatenate(clean_data)
    sio.savemat('clean_data/tsne.mat', {'data': train_data})  # Saving .mat File of MYO


if __name__ == '__main__':
    # train_data((1.5, 8.5))

    # wash_hdemg_data((2.5,7.5),"gesture",[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    # wash_hdemg_data((2.5,7.5),"test",[20])

    # wash_norm_hdemg_data((3.5,6.5),"gesture",[1,2,3,4,5])
    # wash_norm_hdemg_data((3.5,6.5),"test",[6])
    wash_norm_hdemg_data((2.5, 7.5), "gesture", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    wash_norm_hdemg_data((2.5, 7.5), "test", [20])
    # wash_tsne_hdemg_data((2.5, 7.5), "./dirty_data/norm", "hdEMG_norm", [20, ])
    # wash_tsne_hdemg_data((2.5, 7.5), "./dirty_data/norm", "hdEMG_norm", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20])
