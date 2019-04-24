# Author: Lorna <shiyipaisizuo@gamil.com>
# LICENSE: MIT

import numpy as np


def compute_grad_bgd(beta, x, y):
  """define a function that computes the gradient
  Parameters
  ------------
  beta: initial point
  x:    real data
  y:    real data label

  Return
  ------------
  array
  """
  grad = [0, 0]
  grad[0] = 2. * np.mean(beta[0] + beta[1] * x - y)
  grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x - y))

  return np.array(grad)


def update_beta(beta, lr, grad):
  """define the function that updates beta
  Parameters
  -------------
  beta: initial point
  lr:   learning_rate
  grad: derivative

  Return
  -----------
  array
  """
  return np.array(beta) - lr * grad


def loss(beta, x, y):
  """define the function to calculate SGD loss
  Parameters
  -------------
  beta: initial point
  x:    real data
  y:    real data label

  Return
  ------------
  array
  """
  squared_err = (beta[0] + beta[1] * x - y) ** 2
  res = np.sqrt(np.mean(squared_err))

  return res
