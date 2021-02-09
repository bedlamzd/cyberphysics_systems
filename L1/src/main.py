from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

img_dir = 'L1/img/'

data_path = 'L1/data/testLab1Var6.csv'
columns = ['time', 'current', 'voltage']


def estimate_params(voltage: np.ndarray, current: np.ndarray, Td: float) -> Tuple[float, float, float, np.ndarray]:
    X = np.c_[voltage, current]
    Y = current

    k = np.linalg.inv(X.T @ X) @ X.T @ Y
    R = (1 - k[1]) / k[0]
    Te = -Td / np.log(k[1])
    L = T * R

    return (R, L, Te, k)


if __name__ == '__main__':
    ## Given
    T = 0.1  # Period
    Td = 1e-3  # sampling period
    data = np.genfromtxt(data_path, names=columns, delimiter=',')

    ## Display all data
    fig, (cur_ax, vol_ax) = plt.subplots(2, 1, sharex=True)
    cur_ax.plot(data['time'], data['current'])
    cur_ax.set(xlabel='Время t, с',
               ylabel='Сила тока, А',
               title='Сила тока и напряжение от времени')
    cur_ax.grid()

    vol_ax.plot(data['time'], data['voltage'])
    vol_ax.set(xlabel='Время t, с',
               ylabel='Напряжение, В')
    vol_ax.grid()

    fig.savefig(img_dir + 'current-voltage-all.png')

    ## Display only N periods
    N = 2
    t_final = N * T
    time = data['time'][data['time'] < t_final]
    current = data['current'][data['time'] < t_final]
    voltage = data['voltage'][data['time'] < t_final]

    fig, (cur_ax, vol_ax) = plt.subplots(2, 1, sharex=True)
    cur_ax.plot(time, current)
    cur_ax.set(xlabel='Время t, с',
               ylabel='Сила тока, А',
               title='Сила тока и напряжение от времени')
    cur_ax.grid()

    vol_ax.plot(time, voltage)
    vol_ax.set(xlabel='Время t, с',
               ylabel='Напряжение, В')
    vol_ax.grid()
    fig.savefig(img_dir + f'current-voltage-{N}T.png')

    ## Estimate parameters
    X = np.c_[data['voltage'][1:-1], data['current'][1:-1]]
    R, L, Te, k = estimate_params(X[:, 0], X[:, 1], Td)

    ## Compare model and given data
    current_est = (X @ k)[:time.size - 1]

    fig, ax = plt.subplots()
    ax.plot(time, current, label='Исходные данные')
    ax.plot(time[1:], current_est, label='Аппроксимированные данные')
    ax.set(xlabel='Время, с',
           ylabel='Сила тока, А',
           title='Сравнение модели и данных')
    ax.grid()
    ax.legend()
    fig.savefig(img_dir + 'current_estimation_comparison.png')

    ## Mean and std of parameters
    R_est = []
    L_est = []
    n = 1000
    for i in range(n):
        indices = (i * T <= data['time']) & (data['time'] <= (i + 1) * T)
        curr = data['current'][indices]
        volt = data['voltage'][indices]

        R, L, Te, k = estimate_params(volt[1:-1], curr[1:-1], Td)

        R_est.append(R)
        L_est.append(L)

    print(
            f'Mean R: {np.mean(R_est, where=np.isfinite(R_est))}, Ohm\n'
            f'Standard deviation R: {np.std(R_est, where=np.isfinite(R_est))}\n'
            f'Mean L: {np.mean(L_est, where=np.isfinite(L_est))}, Hn\n'
            f'Standard deviation L: {np.std(L_est, where=np.isfinite(L_est))}\n'
            )
