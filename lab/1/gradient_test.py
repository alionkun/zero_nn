import math
from tensor import Tensor

# y = a * x * x + b * x + c
# by hande
a = 3.1
x = 2.0
b = -1.2
c = 99.0
y = a * x * x + b * x + c
dy_da = x * x
dy_dx = 2.0 * a * x + b
dy_db = x
dy_dc = 1.0
print(f'by hand:     y={y}, dy_da={dy_da}, dy_dx={dy_dx}, dy_db={dy_db}, dy_dc={dy_dc}')

# by autograd
A = Tensor(a)
X = Tensor(x)
B = Tensor(b)
C = Tensor(c)
Y = A * X * X + B * X + C
Y.backward()
dY_dA = A.gradient
dY_dX = X.gradient
dY_dB = B.gradient
dY_dC = C.gradient
print(f'by autograd: Y={Y.value}, dY_dA={dY_dA}, dY_dX={dY_dX}, dY_dB={dY_dB}, dY_dC={dY_dC}')

assert math.isclose(y, Y.value) # Python 3.5 or newer
assert math.isclose(dy_da, dY_dA)
assert math.isclose(dy_dx, dY_dX)
assert math.isclose(dy_db, dY_dB)
assert math.isclose(dy_dc, dY_dC)

