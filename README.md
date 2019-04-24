# zero2BGD

![BGD.png](https://images2015.cnblogs.com/blog/764050/201512/764050-20151230190320667-1412088485.png)

Batch gradient descent
Vanilla gradient descent, aka batch gradient descent, computes the gradient of the cost function w.r.t. to the parameters **θ** for the entire training dataset:

**θ=θ−η⋅∇θJ(θ).**

As we need to calculate the gradients for the whole dataset to perform just one update, batch gradient descent can be very slow and is intractable for datasets that don't fit in memory. Batch gradient descent also doesn't allow us to update our model online, i.e. with new examples on-the-fly.

In code, batch gradient descent looks something like this:

```text
for i in range(nb_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad
```
for a pre-defined number of epochs, we first compute the gradient vector params_grad of the loss function for the whole dataset w.r.t. our parameter vector params. Note that state-of-the-art deep learning libraries provide automatic differentiation that efficiently computes the gradient w.r.t. some parameters. If you derive the gradients yourself, then gradient checking is a good idea. (See here for some great tips on how to check gradients properly.)

We then update our parameters in the opposite direction of the gradients with the learning rate determining how big of an update we perform. Batch gradient descent is guaranteed to converge to the global minimum for convex error surfaces and to a local minimum for non-convex surfaces.


**LICENSE**
- [MIT](https://github.com/Lornatang/zero2BGD/blob/master/LICENSE)