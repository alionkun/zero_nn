import random
import math
from tensor import Tensor

# y = a * x + b
real_a = 3.14
real_b = -17.99
# generate dataset
N = 100
Ys = []
Xs = []
for i in range(N):
  x = (random.random() - 0.5) * 2  # (-1, 1)
  y = real_a * x + real_b
  y += (random.random() - 0.5) * 0.01 # add random error
  Xs.append(x)
  Ys.append(y)

# learning
epochs = 100
lr = 0.001
A = Tensor(random.random())
B = Tensor(random.random())
for epoch in range(epochs):
  epoch_loss = 0.0
  for i in range(N):
    # forward pass, compute the loss
    x = Xs[i]
    y_true = Ys[i]
    X = Tensor(x)
    Y_TRUE = Tensor(y_true)
    Y_PRED = A * X + B
    LOSS = (Y_TRUE - Y_PRED) * (Y_TRUE - Y_PRED)
    epoch_loss += LOSS.value
    # backward pass, compute the LOSS gradients w.r.t A and B
    LOSS.reset_gradient()
    LOSS.backward()
    # apply SGD
    A.value -= lr * A.gradient
    B.value -= lr * B.gradient

  print(f'epoch={epoch}/{epochs}, loss={epoch_loss/N}, A={A.value}, B={B.value}')


print(f'real_a={real_a}, real_b={real_b}')
print(f'learned_a={A.value:.2f}, learned_b={B.value:.2f}')

assert math.isclose(A.value, real_a, rel_tol=1e-02) # Python 3.5 or newer
assert math.isclose(B.value, real_b, rel_tol=1e-02)

