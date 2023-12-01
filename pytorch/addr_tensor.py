import torch

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

tensor = torch.rand(3,4)

print(type(x_data), type(tensor))
print(x_data)
print(tensor)

print(tensor.shape, tensor.dtype, tensor.device)
print(x_data.shape, x_data.dtype, x_data.device)
