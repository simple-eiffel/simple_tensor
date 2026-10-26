"""Independent expected values for simple_tensor tests (T9).
1. AdamW: the algorithm as written in the PyTorch 2.14 torch.optim.AdamW docs (decoupled decay first).
2. rixgpt.py lr_at, copied verbatim from D:/rixgpt/rixgpt.py lines 96-99."""
import math

# --- 1. AdamW hand trace: w0 = [1.0, -0.5], loss = sum(w * x) => grad = x every step
x = [0.5, -2.0]
w = [1.0, -0.5]
m = [0.0, 0.0]
v = [0.0, 0.0]
lr, b1, b2, eps, wd = 0.1, 0.9, 0.99, 1e-8, 0.1
for t in range(1, 4):
    for i in range(2):
        g = x[i]
        w[i] = w[i] - lr * wd * w[i]
        m[i] = b1 * m[i] + (1 - b1) * g
        v[i] = b2 * v[i] + (1 - b2) * g * g
        mh = m[i] / (1 - b1 ** t)
        vh = v[i] / (1 - b2 ** t)
        w[i] = w[i] - lr * mh / (math.sqrt(vh) + eps)
print("adamw after 3 steps:", repr(w[0]), repr(w[1]))

# --- 2. lr_at (rixgpt.py:96-99), with a.lr=1e-3, a.warmup=150, a.iters=5000
class A: pass
a = A(); a.lr = 1e-3; a.warmup = 150; a.iters = 5000
def lr_at(it):
    if it < a.warmup: return a.lr*it/max(a.warmup,1)
    r = (it-a.warmup)/max(a.iters-a.warmup,1)
    return a.lr*0.1 + 0.5*(a.lr - a.lr*0.1)*(1+math.cos(math.pi*min(r,1.0)))
for it in (0, 75, 150, 2575, 5000):
    print("lr_at", it, repr(lr_at(it)))
