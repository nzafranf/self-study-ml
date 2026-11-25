import torch
import torch.nn.functional as F


def bow_loop(x):
    B, T, C = x.shape
    out = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            out[b, t] = x[b, : t + 1].mean(0)
    return out


def bow_matmul(x):
    B, T, C = x.shape
    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(1, keepdim=True)
    return wei @ x


def causal_weights(T):
    tril = torch.tril(torch.ones(T, T))
    wei = torch.zeros(T, T).masked_fill(tril == 0, float("-inf"))
    return F.softmax(wei, dim=-1)


def bow_softmax(x):
    return causal_weights(x.shape[1]) @ x
