# zero2MBGD

Mini-batch gradient descent finally takes the best of both worlds and performs an update for every mini-batch of 
n
 
- training examples:

θ=θ−η⋅∇θJ(θ;x(i:i+n);y(i:i+n)).

This way, it a) reduces the variance of the parameter updates, which can lead to more stable convergence; and b) can make use of highly optimized matrix optimizations common to state-of-the-art deep learning libraries that make computing the gradient w.r.t. a mini-batch very efficient. Common mini-batch sizes range between 50 and 256, but can vary for different applications. Mini-batch gradient descent is typically the algorithm of choice when training a neural network and the term SGD usually is employed also when mini-batches are used. Note: In modifications of SGD in the rest of this post, we leave out the parameters 
x(i:i+n);y(i:i+n)).
 for simplicity.

In code, instead of iterating over examples, we now iterate over mini-batches of size 50:

```text
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
```

**LICENSE**
- [MIT](https://github.com/Lornatang/zero2BGD/blob/master/LICENSE)