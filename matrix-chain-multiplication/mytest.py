import torch
import extension_cpp
import time
import random

def get_standard_cost(tensors):
    a:torch.Tensor = tensors[0]
    n, m = a.size(0), a.size(1)
    c = 0.0

    for i in range(1, len(tensors)):
        b:torch.Tensor = tensors[i]
        c += n*m*b.size(1)
        m = b.size(1)

    return c


n = 40
matrices = []
for i in range(n):
    if i == 0:
        x = 1024
        y = random.randint(32, 2048)
        a = torch.randn(x, y, dtype=torch.float32, requires_grad=False, device='cpu')*0.1
        matrices += [a]
    else:
        x = matrices[-1].size(1)
        y = random.randint(32, 2048)
        a = torch.randn(x, y, dtype=torch.float32, requires_grad=False, device='cpu')*0.1
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
print("Torch CPU cost = ", get_standard_cost(matrices))
print()

start = time.time()*1000
d2 = extension_cpp.dot_chain_cpu(matrices)
end = time.time()*1000
print("Custom CPU Forward Pass Duration = ", end-start)
print("Custom CPU Forward Pass Output\n", d2)
print()

max_val = d1.max().abs()

d1 = d1/max_val
d2 = d2/max_val

print(torch.abs(d1-d2).max())

assert torch.allclose(d1, d2, atol=1e-5), "Error in custom matrix impl CPU"

matrices_gpu = [x.to(device='cuda:0') for x in matrices]

start = time.time()*1000
d3 = extension_cpp.dot_chain_gpu(matrices_gpu)
end = time.time()*1000
print("Custom GPU Forward Pass Duration = ", end-start)
print("Custom GPU Forward Pass Output\n", d3)
print()

d3 = d3/max_val

print(torch.abs(d1-d3).max())

assert torch.allclose(d1, d3, atol=1e-5), "Error in custom matrix impl GPU"