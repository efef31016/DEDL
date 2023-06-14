import numpy as np
import matplotlib.pyplot as plt
import random

def grad_plot(grad_terms, layers):

    '''
    Organize the historical records of gradient changes with order. (from fisrt layer and up to down (think this by NN graph))

    grad_terms : a list which has shape like grad[迭代次數][層數][梯度,參數值].shape = 該層的梯度和參數的shape
    '''

    grad_plot = []

    for layer_ in range(len(layers)-1):

        grad_num = layers[layer_] * layers[layer_+1]

        for i in range(grad_num):

            temp = []

            for times in range(len(grad_terms)):

                temp.append(grad_terms[times][layer_].reshape(-1)[i])

            grad_plot.append(temp)

    return np.array(grad_plot)


# 更全面查看 grad
def visual_grad(layersi, layerso, grad, start, n=30):

    '''
    plotting n subplot to see the change of gradients.

    grad : history of gradient for each parameters with order (from fisrt layer and up to down (think this by NN graph))
    start : which number of parameter to start (from fisrt layer and up to down (think this by NN graph))
    n : how many graph we may plot
    '''

    num = layersi * layerso

    # if there are too many terms, we are going to choose appropriate number of grad_terms
    rdm_idx = random.sample(range(start, start+num), n)
    sorted_idx = sorted(rdm_idx)
    grad = [grad[i] for i in sorted_idx]

    fig, axes = plt.subplots(nrows=n//10, ncols=10, figsize=(18, 5))

    for idx, ax in enumerate(axes.ravel()):
        try:
            ax.plot(grad[idx])
            ax.set_title("%d"%sorted_idx[idx])
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
        except:
            pass

    fig.subplots_adjust(hspace=0.8, wspace=0.8)
    plt.show()

    return 0

'''
Example of visual_grad
print("======================================LAYER_1======================================")
start = 0
visual_grad(layers[0],layers[1], grad_plot_v, start)
start += 30
print("======================================LAYER_2======================================")
visual_grad(layers[1],layers[2], grad_plot_v, start)
start += 30*30
print("======================================LAYER_3======================================")
visual_grad(layers[2],layers[3], grad_plot_v, start)
'''


def grad_hist(grad_r, grad_ub, bins, layers, vw, folder_path, date_string):

    n = len(layers)-1
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(15, 5))
    grad_r_ = grad_plot(grad_r, layers)
    grad_ub_ = grad_plot(grad_ub, layers)

    for hid_num in range(n):
        grad_r_tmp = grad_r_[:layers[hid_num]*layers[hid_num+1]].reshape(-1)
        grad_r_ = grad_r_[layers[hid_num]*layers[hid_num+1]:]
        grad_ub_tmp = grad_ub_[:layers[hid_num]*layers[hid_num+1]].reshape(-1)
        grad_ub_ = grad_ub_[layers[hid_num]*layers[hid_num+1]:]
        axes.ravel()[hid_num].hist(grad_r_tmp, bins=bins)
        axes.ravel()[hid_num].hist(grad_ub_tmp, bins=bins, alpha=0.7)
        axes.ravel()[hid_num].legend(["$L_r(θ)$","$L_{ub}(θ)$"])
        axes.ravel()[hid_num].set_title("Layer%d"%(hid_num+1))
    plt.savefig(folder_path + '/%s/grad_terms_%s.jpg'%(date_string,vw))
    plt.show()

    return 0