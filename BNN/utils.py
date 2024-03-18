import torch
import numpy as np
import math

# ground truth seed (always fixed to 0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cpu')
torch.set_default_tensor_type('torch.FloatTensor')

# one-hidden-layer MLP with ReLU activation
def nn(x, param, d_x, d_h):
    w1, b1, w2 = get_param(param, d_x, d_h)
    z1 = x.unsqueeze(0) @ w1 / math.sqrt(d_x) + b1 * 0.1
    a1 = torch.nn.functional.relu(z1)
    y = a1 @ w2 / math.sqrt(d_h)
    return y[..., 0]

def get_param(param, d_x, d_h):
        w1, b1, w2 = param[:, :d_x*d_h].view(-1, d_x, d_h), param[:, d_x*d_h:(d_x+1)*d_h].view(-1, 1, d_h), param[:, (d_x+1)*d_h:].view(-1, d_h, 1)
        return w1, b1, w2

def get_energy(x_train, y_train, d_x, d_h, s_n):
    def energy(param):
        y_train_pred = nn(x_train, param, d_x, d_h)
        return 0.5 * torch.sum((y_train_pred-y_train)**2, dim=-1) / (s_n**2) + 0.5 * torch.sum(param**2, dim=-1)
    return energy

def get_param_dim(d_x, d_h):
    return d_x * d_h + d_h + d_h

def get_ground_truth_bnn(d_x=20, d_h=25, s_n=0.1):
    dim = get_param_dim(d_x, d_h)

    param_true = torch.randn([dim])
    param_true = param_true.unsqueeze(0)

    with torch.no_grad():
        x_train = torch.randn([500, d_x])
        y_train = nn(x_train, param_true, d_x, d_h)
        y_train = y_train + s_n * torch.randn_like(y_train)

    with torch.no_grad():
        x_test = torch.randn([500, d_x])
        y_test = nn(x_test, param_true, d_x, d_h)
        y_test = y_test + s_n * torch.randn_like(y_test)


    param_true = param_true.to(device)
    x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

    energy = get_energy(x_train, y_train, d_x, d_h, s_n)

    return x_train, x_test, y_train, y_test, param_true, energy

def evaluate_test_nll(samples, x_test, y_test, d_x, d_h, s_n):
    y_test_pred = nn(x_test, samples, d_x, d_h)
    nll = 0.5*torch.log(torch.tensor(2.0) * torch.pi * s_n**2) + 0.5 * torch.mean((y_test_pred - y_test)**2, dim=-1) / (s_n**2)
    return nll.mean().item()
