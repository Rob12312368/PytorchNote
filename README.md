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

# WorkFlow
This part is about the generic neural network code in pytorch.

```
# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))

torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 100

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    ### Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_0(X_test)

      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
```


