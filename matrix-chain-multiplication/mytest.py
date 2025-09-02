import torch
import extension_cpp
import time
import random

n = 20
matrices = []
for i in range(n):
    if i == 0:
        x = 128
        y = random.randint(32, 2048)
        a = torch.randn(x, y, dtype=torch.float32, requires_grad=False, device='cpu')
        matrices += [a]
    else:
        x = matrices[-1].size(1)
        y = random.randint(32, 2048)
        a = torch.randn(x, y, dtype=torch.float32, requires_grad=False, device='cpu')
        matrices += [a]

start = time.time()*1000
d1 = None
for i in range(n-1):
    if i == 0:
        d1 = torch.matmul(matrices[i], matrices[i+1])
    else:
        d1 = torch.matmul(d1, matrices[i+1])
end = time.time()*1000
print("Torch CPU Forward Pass Duration = ", end-start)
print("Torch CPU Forward Pass Output\n", d1)
print()

start = time.time()*1000
d2 = extension_cpp.dot_chain_cpu(matrices)
end = time.time()*1000
print("Torch CPU Forward Pass Duration = ", end-start)
print("Torch CPU Forward Pass Output\n", d2)
print()