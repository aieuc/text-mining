import torch

W_h = torch.randn(20, 20, requires_grad=True)
W_x = torch.randn(20, 10, requires_grad=True)
x = torch.randn(1, 10)
prev_h = torch.randn(1, 20)

h2h = torch.mm(W_h, prev_h.t())
i2h = torch.mm(W_x, x.t())

next_h = h2h + i2h
next_h = next_h.tanh()
loss = next_h.sum()

print(h2h)
print(i2h)
print(next_h)
print(loss)

result = loss.backward()
print(result)
