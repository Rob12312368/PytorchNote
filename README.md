# PytorchNote
This document serves as my pytorch dictionary. You can use ctrl+f to find the syntax in an instant.
# Fundamentals
## Create a tensor
create a tensor `tensor = torch.tensor(n,m)`  
create a tensor with random values `torch.rand(size=(n,m))`  
create a tensor with ones `torch.ones(size=(n,m))`  
create a tensor with zeros `torch.zeros(size=(n,m))`  
create a tensor with series of numbers `torch.arange(start,end,step)`  

## Show details
show count of dimensions (how many paris of square brackets) `tensor.ndim`  
turn tensor back to python int `tensor.item()`  
show length of each dimension `tensor.shape`  
show data type `tensor.dtype`  

## Basic Statistics
find min, max, mean, sum `torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x)` (x == tensor)
find argmax, argmin `tensor.argmax(), tensor.argmin()`

## Change Shape
reshape `torch.reshape(n,m)`  
create tmeporary view (note changing view will affect original tensor) `tensor.view()`  
stack torch horizontally or vertically `torch.stack([tensors, tensors, ...], dim)`  
Returns input with a dimension value of 1 added at dim `torch.unsqueeze()`  
Squeezes input to remove all the dimenions with value 1`torch.squeeze()`  

## Pytorch and Numpy
numpy to pytorch `torch.from_numpy()`  
pytorch to numpy `tensor.numpy()`


## Others
change type `tensor.type(torch.int8)`




