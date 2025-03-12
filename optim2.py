import math
import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn
import torch

class AdamW_Mon_Add(Optimizer):
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
                beta1, beta2 = group["betas"]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["m_A"] = torch.zeros((p.data.shape[0], self.rank), dtype=p.data.dtype, device=p.data.device)
                    state["m_B"] = torch.zeros((self.rank, p.data.shape[1]), dtype=p.data.dtype, device=p.data.device)
                    # Exponential moving average of squared gradient values
                    state["sq_A"] = torch.zeros((p.data.shape[0], self.rank), dtype=p.data.dtype, device=p.data.device)
                    state["sq_B"] = torch.zeros((self.rank, p.data.shape[1]), dtype=p.data.dtype, device=p.data.device)
                    
                state["step"] += 1
                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, beta1 * state["m_A"] @ state["m_B"] + (1-beta1) * grad, (beta2 * state["sq_A"] @ state["sq_B"] + (1-beta2) * grad * grad).sqrt().add_(group["eps"]))

                if state["step"] == 1:
                    U1, S1, Vh1 = torch.linalg.svd(grad, full_matrices=False)
                    U1_r = U1[:, :r]
                    S1_r = S1[:r] 
                    Vh1_r = Vh1[:r, :]
                    state["m_A"] = U_r1 @ torch.diag(S1_r.sqrt()) 
                    state["m_B"] = torch.diag(S1_r.sqrt()) @ Vh1_r

                    U2, S2, Vh2 = torch.linalg.svd(grad*grad, full_matrices=False)
                    U2_r = U2[:, :r]
                    S2_r = S2[:r] 
                    Vh2_r = Vh2[:r, :]
                    state["sq_A"] = U_r2 @ torch.diag(S2_r.sqrt()) 
                    state["sq_B"] = torch.diag(S1_r.sqrt()) @ Vh2_r

                else:
                    
                    m_A, m_B, sq_A, sq_v, sq_s = state["m_A"], state["m_B"], state["sq_A"], state["sq_B"]
                    delta = 1e-8
 
                    # computing the inverse matrix
                    m_A_TA = m_A.T @ m_A
                    m_BB_T = m_B @ m_B.T
                    m_A_TA_inv = torch.linalg.pinv(m_A_TA + delta * torch.eye(m_A.shape[1]).to(m_A.device)) 
                    m_BB_T_inv = torch.linalg.pinv(m_BB_T + delta * torch.eye(m_B.shape[0]).to(m_B.device)) 
                    g_mA = grad @ m_B.T @ m_BB_T_inv
                    g_mA_projected = g_mA - m_A @ m_A_TA_inv @ (m_A.T @ g_mA)
                    g_mB = m_A_TA_inv @ m_A.T @ grad
                    g_mB_projected = g_mB - (g_mB @ m_B.T) @ m_BB_T_inv @ m_B

                    sq_A_TA = sq_A.T @ sq_A
                    sq_BB_T = sq_B @ sq_B.T
                    sq_A_TA_inv = torch.linalg.pinv(sq_A_TA + delta * torch.eye(sq_A.shape[1]).to(sq_A.device)) 
                    sq_BB_T_inv = torch.linalg.pinv(sq_BB_T + delta * torch.eye(sq_B.shape[0]).to(sq_B.device)) 
                    g_sqA = (grad*grad) @ sq_B.T @ sq_BB_T_inv
                    g_sqA_projected = g_sqA - sq_A @ sq_A_TA_inv @ (sq_A.T @ g_sqA)
                    g_sqB = sq_A_TA_inv @ sq_A.T @ (grad*grad)
                    g_sqB_projected = g_sqB - (g_sqB @ sq_B.T) @ sq_BB_T_inv @ sq_B

                    m_A.mul_(0.5+0.5*beta1).add_(g_mA_projected, alpha=1 - beta1)
                    m_B.mul_(0.5+0.5*beta1).add_(g_mB_projected, alpha=1 - beta1)

                    sq_A.mul_(0.5+0.5*beta2).add_(g_sqA_projected, alpha=1 - beta2)
                    sq_B.mul_(0.5+0.5*beta2).add_(g_sqB_projected, alpha=1 - beta2)
                    

                


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

                
              
