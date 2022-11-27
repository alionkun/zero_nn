
class Tensor:
  def __init__(self, value):
    assert isinstance(value, (float, int)), f'int and float only, but value={value} is a {type(value)}'
    self.value = value
    self.parent_nodes = []
    self.parent_node_gradient_functions = []
    self.gradient = 0.0

  def __add__(self, other):
    result = Tensor(self.value + other.value)
    result.parent_nodes.append(self)
    result.parent_nodes.append(other)
    result.parent_node_gradient_functions.append(lambda g : g)
    result.parent_node_gradient_functions.append(lambda g : g)
    return result

  def __sub__(self, other):
    result = Tensor(self.value - other.value)
    result.parent_nodes.append(self)
    result.parent_nodes.append(other)
    result.parent_node_gradient_functions.append(lambda g : g)
    result.parent_node_gradient_functions.append(lambda g : -1.0 * g)
    return result

  def __mul__(self, other):
    result = Tensor(self.value * other.value)
    result.parent_nodes.append(self)
    result.parent_nodes.append(other)
    result.parent_node_gradient_functions.append(lambda g : other.value * g)
    result.parent_node_gradient_functions.append(lambda g : self.value * g)
    return result

  def __div__(self, other):
    result = Tensor(self.value / other.value)
    result.parent_nodes.append(self)
    result.parent_nodes.append(other)
    result.parent_node_gradient_functions.append(lambda g : 1.0 / other.value * g)
    result.parent_node_gradient_functions.append(lambda g : self.value * g)
    return result

  def backward(self, g=None):
    g = g or 1.0
    self.gradient += g
    for parent, grad_func in zip(self.parent_nodes, self.parent_node_gradient_functions):
      parent.backward(grad_func(g))

  def reset_gradient(self, back=True):
    self.gradient = 0.0
    for parent in self.parent_nodes:
      parent.reset_gradient(back)

