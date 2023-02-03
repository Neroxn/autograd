import numpy as np
from typing import List, NamedTuple, Callable, Optional, Union

Arrayable = Union[np.ndarray, List, int, float]

class Dependency(NamedTuple):
    """ A basic dependency graph for a tensor."""
    parent: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

def ensure_array(data : Arrayable) -> np.ndarray:
    """ Ensure that data is an arrayable."""
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)

class Tensor:
    """ A basic container for a tensor."""
    def __init__(
        self, 
        data: Arrayable,
        requires_grad: bool = False, 
        depends_on: List[Dependency] = []) -> None:

        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on 
        self.shape = self.data.shape

        self.grad : Optional['Tensor'] = None
        if requires_grad:
            self.zero_grad()


    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, shape = {self.shape})"

    def zero_grad(self) -> None:
        """ Set the gradient to zero."""
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad : 'Tensor' = None) -> None:
        """ Backpropagate the gradient through the graph."""
        assert self.requires_grad, "Can't call backward on a tensor that doesn't require gradients"

        if grad is None: # if 0-dim is specified, gradient is 1.0 by default
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise ValueError("Grad must be specified for non-zero dimensional tensors")
        print(grad)
        self.grad.data += grad.data

        if self.depends_on is not None:
            for dependency in self.depends_on:
                dependency.parent.backward(dependency.grad_fn(grad.data))

    def sum(self) -> 'Tensor':
        return tensor_sum(self)


def tensor_sum(t : Tensor):
    """
    Sum the elements of the tensor t to produce 0-dimensional tensor
    """
    data = t.data.sum()
    requires_grad = t.requires_grad

    # update dependency graph
    if requires_grad:
        def grad_fn(grad : np.ndarray): 
            return np.ones_like(t.data) * grad
        depends_on = [Dependency(parent=t, grad_fn=grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)