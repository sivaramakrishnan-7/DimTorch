import numpy as np 

class DimTensor :

    def __init__(self,tensor:list,grad:bool = False):
        self.tensor = np.array(tensor)
        self.grad = grad

    def _to_tensor(self, other):
        return other if isinstance(other, DimTensor) else DimTensor(other)

    def __add__(self,other):
        other = self._to_tensor(other)
        out = DimTensor(self.tensor + other.tensor, grad=self.grad or other.grad)
        return out
    
    def __neg__(self):
        return DimTensor(self.tensor * -1, grad=self.grad)
    
    def __sub__(self,other):
        other = self._to_tensor(other) 
        out = DimTensor(self.tensor + (-other).tensor, grad=self.grad or other.grad)
        return out


    def __mul__(self, other):
        other = self._to_tensor(other)
        out = DimTensor(self.tensor * other.tensor, grad=self.grad or other.grad)
        return out
    
    def __repr__(self):
        return f"DimTensor({self.tensor}, grad = {self.grad})"

    def __str__(self):
        return str(self.tensor)
    
    def __lt__(self,other):
        other = self._to_tensor(other)
        return (self.tensor < other.tensor).all()
    
    def __eq__(self, other):
        other = self._to_tensor(other)
        return (self.tensor == other.tensor).all()
    

    

        
    
