import torch
import numpy
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        # 此处父类将超参数存进param_groups里
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        # 直接修改params
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                # 初始化
                if len(self.state[param]) == 0:
                    self.state[param]['m'] = torch.zeros_like(param.data)
                    self.state[param]['v'] = torch.zeros_like(param.data)
                    self.state[param]['step'] = 0
                # 更新step
                self.state[param]['step'] += 1
                # 获取超参数
                lr: float = group['lr'] # 学习率
                betas: tuple = group['betas'] # beta
                eps: float = group['eps'] # epsilon
                weight_decay: float = group['weight_decay'] # lambda
                # 获取step
                t = self.state[param]['step']
                # 算
                self.state[param]['m'] = betas[0] * self.state[param]['m'] + (1 - betas[0]) * param.grad
                self.state[param]['v'] = betas[1] * self.state[param]['v'] + (1 - betas[1]) * param.grad ** 2
                lr_t: torch.Tensor = lr / (1 - betas[0] ** t) * math.sqrt(1 - betas[1] ** t)
                # 这里必须加上.data 这是裸张量 对data的操作不会被torch追踪
                # Adam更新
                param.data -= lr_t * self.state[param]['m'] / (torch.sqrt(self.state[param]['v']) + eps)
                # Weight decay
                param.data -= lr * weight_decay * param.data

def get_lr_cosine_schedule(step: int, warmup_steps: int, min_lr: float, max_lr: float, ending_decay_step: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps * max_lr
    elif step < ending_decay_step:
        return 0.5 * (max_lr + min_lr) + 0.5 * (max_lr - min_lr) * math.cos((step - warmup_steps) / (ending_decay_step - warmup_steps) * math.pi)
    else:
        return min_lr

def gradient_clipping(params: list[torch.Tensor], max_norm: float) -> None:
    sum_of_squares: float = 0
    params_with_grad: list[torch.Tensor] = []
    for param in params:
        if param.grad is not None and param.requires_grad:
            params_with_grad.append(param)
    for param in params_with_grad:
        sum_of_squares += param.grad.norm() ** 2
    total_norm = math.sqrt(sum_of_squares)
    if total_norm > max_norm:
        for param in params_with_grad:
            param.grad.data *= max_norm / (total_norm + 1e-6)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, filename: str):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }, filename)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str) -> int:
    # weights_only = False是为了把非张量的iteration加载进来
    checkpoint: dict[str, Any] = torch.load(filename, weights_only = False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

         