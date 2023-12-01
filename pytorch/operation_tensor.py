import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device, type(device))

tensor = torch.ones(4, 4)
print(tensor)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

tensor[:,1] = 0
tensor[2] = 3
tensor[..., -1] = 9
print(tensor)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)

torch.matmul(tensor, tensor.T, out=y3)

print(y1)
print(y2)
print(y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(z1)
print(z2)
print(z3)

agg = tensor.sum()
print(agg)
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
