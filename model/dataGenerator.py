import numpy as np
from sklearn.utils import shuffle


def gen_sub_data(x, y, labels, now_label, count):
    temp_xx = np.array([x[i] for i in range(x.shape[0]) if y[i] in labels])
    np.random.shuffle(temp_xx)
    temp_x = temp_xx[:count]
    temp_y = np.full(len(temp_x), now_label)
    ril = [i for i in range(x.shape[0]) if x[i] in temp_x]
    x = np.delete(x, ril, axis=0)
    y = np.delete(y, ril)
    temp_x = temp_x.reshape((temp_x.shape[0], 1, temp_x.shape[1]))
    return x, y, temp_x, temp_y


def gen_data(x, y, num_labels):
    label_count = []
    out_x = []
    out_y = []
    for i in range(num_labels):
        label_count.append(np.sum(y == i))
    now_label = 0
    x, y, temp_x_p, temp_y_p = gen_sub_data(x, y, [0], now_label, int(label_count[0] / num_labels))
    now_label = now_label + 1
    x, y, temp_x_n, temp_y_n = gen_sub_data(x, y, set(range(num_labels)) - {0}, now_label,
                                            int(label_count[0] / num_labels))
    temp_x = np.append(temp_x_p, temp_x_n)
    temp_x.resize((temp_x_p.shape[0] * 2, 1, temp_x_p.shape[-1]))
    temp_y = np.append(temp_y_p, temp_y_n)
    now_label = now_label + 1
    temp_x, temp_y = shuffle(temp_x, temp_y)
    out_x.append(temp_x)
    out_y.append(temp_y)
    x, y = shuffle(x, y)
    for i in range(2, num_labels + 1):
        x_p_s = []
        x_n_s = []
        for j in range(int(label_count[i - 1] / num_labels)):
            x_p = []
            x_n = []
            for m in range(i):
                for k in range(x.shape[0]):
                    if y[k] == m:
                        x_p.append(x[k])
                        x = np.delete(x, k, axis=0)
                        y = np.delete(y, k)
                        break
            x_n = x_p.copy()
            while (x_p[0] == x_n[0]).all() and (x_p[i - 1] == x_n[i - 1]).all():
                np.random.shuffle(x_n)
            x_p_s.append(x_p)
            x_n_s.append(x_n)
            if i > 2:
                x_p_d = x_p.copy()
                count = np.random.randint(1, i - 1)
                il = np.random.randint(1, i-1, count)
                for index in il:
                    x_p_d.pop(index)
                x_n_d = x_p_d.copy()
                while (x_p_d[0] == x_n_d[0]).all() and (x_p_d[len(x_p_d) - 1] == x_n_d[len(x_n_d) - 1]).all():
                    np.random.shuffle(x_n_d)
                x_p_s.append(x_p_d)
                x_n_s.append(x_n_d)
        y_p = np.full(len(x_p_s), now_label)
        now_label = now_label + 1
        y_n = np.full(len(x_n_s), now_label)
        now_label = now_label + 1
        temp_x = np.array(x_p_s + x_n_s)
        temp_y = np.append(y_p, y_n)
        temp_x, temp_y = shuffle(temp_x, temp_y)
        out_x.append(temp_x)
        out_y.append(temp_y)
    return out_x, out_y


if __name__ == "__main__":
    x = np.zeros((64, 2))
    y = np.zeros(64)
    for i in range(4):
        for j in range(16):
            x[i * 16 + j] = np.array([(i + 1) * 100 + j, (i + 1) * 100 + j])
            y[i * 16 + j] = i
    print(x)
    print(y)
    out_x, out_y = gen_data(x, y, 4)
    out_x = np.array(out_x)
    out_y = np.array(out_y)
    print(out_x.shape)
    print('----------------------------')
    print(out_y.shape)