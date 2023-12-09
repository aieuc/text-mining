import torch

A = torch.randn(2, 2)
B = torch.randn(2, 2)
C = A + B
print(A)
print(B)
print(C)

A = torch.randn(2, 2)
B = torch.randn(2, 1)
C = torch.matmul(A, B)
print(A)
print(B)
print(C)

 
