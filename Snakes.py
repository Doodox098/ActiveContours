import argparse
from asyncio.windows_events import NULL
import math
import copy
import numpy as np
import skimage.io
from scipy.signal import convolve
import matplotlib.pyplot as plt
import skimage.io as skio

DEBUG = True   
def display_image_in_actual_size(img, dpi = 80):
    height = img.shape[0]
    width = img.shape[1]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(img, cmap='gray')

#     plt.show()
    
    return fig, ax

def save_mask(fname, snake, img):
    plt.ioff()
    fig, ax = display_image_in_actual_size(img)
    ax.fill(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    fig.savefig(fname, pad_inches=0, bbox_inches='tight', dpi='figure')
    plt.close(fig)
    
    mask = skio.imread(fname)
    blue = ((mask[:,:,2] == 255) & (mask[:,:,1] < 255) & (mask[:,:,0] < 255)) * 255
    skio.imsave(fname, blue.astype(np.uint8))
    plt.ion()
    
def display_snake(img, init_snake, result_snake):
    fig, ax = display_image_in_actual_size(img)
    ax.plot(init_snake[:, 0], init_snake[:, 1], '-r', lw=2)
    ax.plot(result_snake[:, 0], result_snake[:, 1], '-b', lw=2)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()


def IoU(x, y):
    
    return (x * y > 0).sum() / (x + y > 0).sum()

def line_force(img, sigma):
    res = np.zeros_like(img)
    rad = math.ceil(3 * sigma)

    x = (np.repeat(np.arange(0, 2 * rad + 1)[None, :], 2 * rad + 1, axis=0) - rad) ** 2
    y = x.T

    gauss_weights = np.exp(-(x + y) / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2)
    
    h, w = gauss_weights.shape
    line = convolve(np.pad(img, ((h // 2, h - h // 2), (w // 2, w - w // 2)), mode='edge'), gauss_weights, mode='valid')
    return -line


def edge_force(img, sigma):
    res = np.zeros_like(img)
    rad = math.ceil(3 * sigma)

    _x = (np.repeat(np.arange(0, 2 * rad + 1)[None, :], 2 * rad + 1, axis=0) - rad)
    x = _x ** 2
    y = x.T

    gauss_weights = np.exp(-(x + y) / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2)
    h, w = gauss_weights.shape
    
    grad_x = convolve(np.pad(img, ((h // 2, h - h // 2), (w // 2, w - w // 2)), mode='edge'), _x * gauss_weights, mode='valid')
    grad_y = convolve(np.pad(img, ((h // 2, h - h // 2), (w // 2, w - w // 2)), mode='edge'), _x.T * gauss_weights, mode='valid')
    grad = (grad_x) ** 2 + (grad_y) ** 2
    
    return -grad


def inv_matrix(a, b, tau, size):
    A = np.zeros((size, size), dtype=float)
    for i, x in enumerate([b, - a - 4 * b, 2 * a + 6 * b, - a - 4 * b, b]):
        A += np.roll(np.eye(size, size) * x, i - 2, axis=1)
    
    return np.linalg.inv(np.eye(size, size) + tau * A)
    

def f_external(snake, grad_x, grad_y):
    eps = 1e-10
    snake -= eps
    snake[snake == 0] += 2 * eps
    w = grad_y.shape[1]
    grad_y = grad_y.ravel()
    Q11, Q12, Q21, Q22 = [np.zeros_like(snake.shape[0])] * 4
    Q11 = grad_y[np.floor(snake).astype(int)[:, 1] * w + np.floor(snake).astype(int)[:, 0]]
    Q12 = grad_y[np.ceil(snake).astype(int)[:, 1] * w + np.floor(snake).astype(int)[:, 0]]
    Q21 = grad_y[np.floor(snake).astype(int)[:, 1] * w + np.ceil(snake).astype(int)[:, 0]]
    Q22 = grad_y[np.ceil(snake).astype(int)[:, 1] * w + np.ceil(snake).astype(int)[:, 0]]
    
    R1 = ((np.ceil(snake).astype(int)[:, 0] - snake[:, 0]) * Q11 + (snake[:, 0] - np.floor(snake).astype(int)[:, 0]) * Q21) / (np.ceil(snake).astype(int)[:, 0] - np.floor(snake).astype(int)[:, 0])
    R2 = ((np.ceil(snake).astype(int)[:, 0] - snake[:, 0]) * Q12 + (snake[:, 0] - np.floor(snake).astype(int)[:, 0]) * Q22) / (np.ceil(snake).astype(int)[:, 0] - np.floor(snake).astype(int)[:, 0])
    Py = ((np.ceil(snake).astype(int)[:, 1] - snake[:, 1]) * R1 + (snake[:, 1] - np.floor(snake).astype(int)[:, 1]) * R2) / (np.ceil(snake).astype(int)[:, 1] - np.floor(snake).astype(int)[:, 1])
    
    w = grad_x.shape[1]
    grad_x = grad_x.ravel()
    Q11, Q12, Q21, Q22 = [np.zeros_like(snake.shape[0])] * 4
    Q11 = grad_x[np.floor(snake).astype(int)[:, 1] * w + np.floor(snake).astype(int)[:, 0]]
    Q12 = grad_x[np.ceil(snake).astype(int)[:, 1] * w + np.floor(snake).astype(int)[:, 0]]
    Q21 = grad_x[np.floor(snake).astype(int)[:, 1] * w + np.ceil(snake).astype(int)[:, 0]]
    Q22 = grad_x[np.ceil(snake).astype(int)[:, 1] * w + np.ceil(snake).astype(int)[:, 0]]
    
    R1 = ((np.ceil(snake).astype(int)[:, 0] - snake[:, 0]) * Q11 + (snake[:, 0] - np.floor(snake).astype(int)[:, 0]) * Q21) / (np.ceil(snake).astype(int)[:, 0] - np.floor(snake).astype(int)[:, 0])
    R2 = ((np.ceil(snake).astype(int)[:, 0] - snake[:, 0]) * Q12 + (snake[:, 0] - np.floor(snake).astype(int)[:, 0]) * Q22) / (np.ceil(snake).astype(int)[:, 0] - np.floor(snake).astype(int)[:, 0])
    Px = ((np.ceil(snake).astype(int)[:, 1] - snake[:, 1]) * R1 + (snake[:, 1] - np.floor(snake).astype(int)[:, 1]) * R2) / (np.ceil(snake).astype(int)[:, 1] - np.floor(snake).astype(int)[:, 1])
    
    return np.hstack((Px.reshape(-1, 1), Py.reshape(-1, 1)))


def reparametriation(snake):
    
    N = snake.shape[0]
    length = ((snake - np.roll(snake, -1, axis=0)) ** 2).sum(axis=1) ** 0.5
    h = length[1:].mean()
    
    new_snake = np.zeros_like(snake)
    lsum = 0
    j = 0
    new_snake[0, :] = snake[0, :]
    for i in range(1, N - 1):
        while h * i > lsum:
            lsum = lsum + length[j + 1]
            j += 1
        new_snake[i] = (snake[j] * (h * i - lsum + length[j]) + snake[j - 1] * (lsum - h * i)) / length[j]
    new_snake[N - 1] = snake[N - 1]
    return new_snake


def step_div(snake):
    N = snake.shape[0]
    length = ((snake - np.roll(snake, -1, axis=0)) ** 2).sum(axis=1) ** 0.5
    h = length.mean()

    return np.abs(length - h).mean() 


def snake_opt(img, init_snake, alpha, beta, tau, w_line, w_edge, kappa):   
    max_iter = 4000
    eps = 1e-2
    sigma = 1
    reparametrization_frequency = 5
    k = kappa
    P = w_line * line_force(img, sigma) + w_edge * edge_force(img, sigma)
    
    h, w = P.shape
    
    PP = np.pad(P, 1, mode='edge')
    grad_x = (PP[1: h + 1, 2:w + 2] - PP[1: h + 1, 0:w]) / 2
    grad_y = (PP[2: h + 2, 1:w + 1] - PP[0: h, 1:w + 1]) / 2
    
    grad_x = -k * grad_x / np.abs(grad_x).mean()
    grad_y = -k * grad_y / np.abs(grad_y).mean()

    inv = inv_matrix(alpha, beta, tau, init_snake.shape[0])
    snake = init_snake
    for i in range(max_iter):
        new_snake = np.dot(inv, snake + tau * f_external(snake, grad_x, grad_y))
        new_snake[:, 0] = np.minimum(np.maximum(new_snake[:, 0], 0), w - 1)
        new_snake[:, 1] = np.minimum(np.maximum(new_snake[:, 1], 0), h - 1)
        
        if np.abs(snake - new_snake).mean() < eps:
            break
        
        snake = new_snake
        
        if i % reparametrization_frequency == 0:
            snake = reparametriation(snake)
    
    return snake


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Deconvolution',
        description='',
    )
    parser.add_argument('input_image')
    parser.add_argument('initial_snake')
    parser.add_argument('output_image')
    parser.add_argument('alpha')
    parser.add_argument('beta')
    parser.add_argument('tau')
    parser.add_argument('w_line')
    parser.add_argument('w_edge')
    parser.add_argument('kappa')
    
    args = parser.parse_args()
    
    img = skimage.io.imread(args.input_image)
    img = img.astype(float) / 255
    if len(img.shape) == 3:
        img = img[:, :, 0]

    init_snake = np.loadtxt(args.initial_snake)
    alpha = float(args.alpha)
    beta = float(args.beta)
    tau = float(args.tau)
    w_line = float(args.w_line)
    w_edge = float(args.w_edge)
    kappa = float(args.kappa)
    
    res = snake_opt(img, init_snake, alpha, beta, tau, w_line, w_edge, kappa)
    save_mask(args.output_image, res, img)
    
    if DEBUG:
        ref = skimage.io.imread(f'{args.input_image.split(".")[0]}_mask.png')
        ref = ref.astype(float) / 255
        if len(ref.shape) == 3:
            ref = ref[:, :, 0]
            
        res = skimage.io.imread(args.output_image)
        res = res.astype(float) / 255
        if len(res.shape) == 3:
            res = res[:, :, 0]
            
        print(IoU(ref, res))
    
