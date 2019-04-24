from sklearn.linear_model import LinearRegression
from ops import compute_grad_bgd, update_beta, loss

import pandas as pd
import numpy as np

# input data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submit = pd.read_csv('data/sample_submit.csv')

# initial setup params
beta = [1, 1]
lr = 0.2
tol_L = 0.1

# normalizes x
max_x = max(train['id'])
x = train['id'] / max_x
y = train['questions']

# do the first calculation
np.random.seed(10)
grad = compute_grad_bgd(beta, x, y)
losses = loss(beta, x, y)
beta = update_beta(beta, lr, grad)
new_losses = loss(beta, x, y)

i = 1
while np.abs(new_losses - losses) > tol_L:
  beta = update_beta(beta, lr, grad)
  grad = compute_grad_bgd(beta, x, y)
  losses = new_losses
  new_losses = loss(beta, x, y)
  i += 1
  print(f'Iter {i} SGD loss {abs(new_losses - losses):.6f}')


val = LinearRegression()
val.fit(train[['id']], train[['questions']])

print(f'Our     Coef: {beta[1] / max_x}')
print(f'Sklearn Coef: {val.coef_[0][0]}')

print(f'Our     Intercept: {beta[0]}')
print(f'Sklearn Intercept: {val.intercept_[0]}')

our_losses = loss(beta, x, y)
sklearn_losses = loss([val.intercept_[0], val.coef_[0][0]], train['id'], y)
print(f'Our     loss: {our_losses}')
print(f"Sklearn loss: {sklearn_losses}")
