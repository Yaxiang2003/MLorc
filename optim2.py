import math
import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn
import torch

class RSVD_CM_AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, correct_bias=True, rank=1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        self.rank=rank
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if p.grad.data.dim() != 2:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")


                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["m_A"] = torch.zeros((p.data.shape[0], self.rank), dtype=p.data.dtype, device=p.data.device)
                    state["m_B"] = torch.zeros((self.rank, p.data.shape[1]), dtype=p.data.dtype, device=p.data.device)
                    # Exponential moving average of squared gradient values
                    state["sq_A"] = torch.zeros((p.data.shape[0], self.rank), dtype=p.data.dtype, device=p.data.device)
                    state["sq_B"] = torch.zeros((self.rank, p.data.shape[1]), dtype=p.data.dtype, device=p.data.device)

                m_A, m_B, sq_A, sq_v, sq_s = state["m_A"], state["m_v"], state["m_s"], state["sq_u"], state["sq_v"]
                delta = 1e-8
 
                # computing the inverse matrix
                AA_T = A @ A.T
                    B_TB = B.T @ B
                    AA_T_inv = torch.linalg.pinv(AA_T + delta * torch.eye(A.shape[0]).to(A.device)) 
                    B_TB_inv = torch.linalg.pinv(B_TB + delta * torch.eye(A.shape[0]).to(A.device)) 

                beta1, beta2 = group["betas"]

                state["step"] += 1

                random_matrix = torch.randn(size=(p.data.shape[1], self.rank+5), device=p.data.device)
                Y1 = grad @ ((1-beta1) * random_matrix) + beta1 * m_u @ torch.diag(m_s) @ (m_v @ random_matrix)
                Y2 = (grad * grad) @ ((1-beta2) * random_matrix) + beta2 * sq_u @ torch.diag(sq_s) @ (sq_v @ random_matrix)

                denom = sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, beta1 * m_u @ torch.diag(m_s) @ m_v + (1-beta1) * grad, (beta2 * sq_u @ torch.diag(sq_s) @ sq_v + (1-beta2) * grad * grad).sqrt().add_(group["eps"]))

                m_u, m_s, m_v = randomized_svd(Y1, self.rank)
                sq_u, sq_s, sq_v = randomized_svd(Y2, self.rank)

                 # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
                if state['step']%5 == 1:
                    print(torch.cuda.memory_summary(device=p.data.device, abbreviated=False))
        return loss

                
              
