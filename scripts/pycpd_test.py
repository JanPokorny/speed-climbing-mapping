from functools import partial
import matplotlib.pyplot as plt
import pycpd
import numpy as np
import sklearn
import sklearn.cluster
import sklearn.preprocessing


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():
    Y = np.loadtxt('all_detection_points.txt')
    X = np.loadtxt('reference/wall_points.txt')

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg = pycpd.AffineRegistration(X=X, Y=Y)
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    main()