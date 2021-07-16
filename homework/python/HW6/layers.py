from builtins import range
import numpy as np


def pre_process(x):
    num_points, new_shape = x.shape[0], np.product(np.array(x.shape[1:]))
    x_prime = x.reshape((num_points, new_shape))
    return x_prime 

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_trans = pre_process(x)
    out = (x_trans @ w) + b  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_trans, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k) (N, D)
    - w: Weights, of shape (D, M)
    - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k) "(N,D)"
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """

    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    x, w, b = cache

    dx, dw, db = dout @ w.T, x.T @ dout, dout.T @ np.ones(dout.shape[0])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.where(x <= 0, 0.0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    #print("out", out.shape, cache.shape, 'cache shape')
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x

    """

    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    relu_prime = np.where(cache <= 0, 0.0, 1.0)
    #print(cache.shape, 'cached', dout.shape, 'dout', "relu_prime", relu_prime.shape)
    dx, x = dout * relu_prime, cache
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x   (dx in R^{N,C}: x_ij score for the jth class for the ith input
    """
    #think about the x with one input ---> what's that loss?
    #maxes = in R^{C}, will ahve the max for every class
    #apply the results componentwise. and take the sum to compute the total loss
    #cross entropy loss is the formulation below
    #= −(sc − m) + log(sum_{k  = 1}^{C} e^{sk−m}
    ###########################################################################
    # TODO: Implement the softmax loss                                        #
    ###########################################################################
    #print(y.shape)
    N = x.shape[0]
    activation = x[np.arange(N), y].reshape(N, 1)
    maxes = np.max(x, axis=1, keepdims=True)
    term_1 = -activation + maxes
    term_2 = np.log(np.sum(np.exp(np.subtract(x, maxes)), axis=1, keepdims=True))
    loss = np.sum(term_1 + term_2)
    
    dx = (np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)) 
    dx[np.arange(N), y] -= 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
