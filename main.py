from core.tensor import DimTensor
from core.operators import exp,log,max,sigmoid,relu

t1 = DimTensor([1, -3], True)
print(exp(t1))     # [2.718..., 0.049...]
print(log(t1))     # [0, error—needs positive input]
print(sigmoid(t1)) # [0.731..., 0.047...]
print(relu(t1))    # [1, 0]

