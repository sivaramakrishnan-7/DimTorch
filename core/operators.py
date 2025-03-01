from core.tensor import DimTensor
import numpy as np

def id(a):
    return a

def max(a,b):
    a = a if isinstance(a, DimTensor) else DimTensor(a)
    b = a._to_tensor(b)
    result = np.maximum(a.tensor,b.tensor)
    return  DimTensor(result, grad=a.grad or b.grad)

def exp(a):
    return DimTensor(np.exp(a.tensor), grad=a.grad)

def log(a):
    if (a.tensor > 0).all():
        return DimTensor(np.log(a.tensor), grad = a.grad)
    else:
        raise ValueError("Inputs must be positive")

def sigmoid(a):
    sig =  1/(1 + np.exp(-a.tensor))
    return DimTensor(sig, grad=a.grad)

def relu(a):
    return DimTensor(np.maximum(0,a.tensor), grad=a.grad)

def is_close(a,b, tol =1e-5):
    b = a._to_tensor(b)
    return (np.abs(a.tensor - b.tensor) < tol).all()

def inv(a):
    return DimTensor(1/a.tensor, grad = a.grad)

def log_back(a, grad):
    return DimTensor(grad.tensor * (1/a.tensor), grad = a.grad)

def inv_back(a,grad):
    return DimTensor(grad.tensor * (-1/(a.tensor)**2), grad = a.grad)

def relu_back(a,grad):
    return DimTensor(grad.tensor*(a.tensor >0), grad=a.grad)






    


