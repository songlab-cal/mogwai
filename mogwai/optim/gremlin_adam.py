import torch
from torch.optim.optimizer import Optimizer


class GremlinAdam(torch.optim.Adam):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider "
                        "SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # NOTE: This ties together all squared updates unlike in Adam.
                sq_grad = (grad.view(1, -1) @ grad.view(-1, 1)).squeeze(1).squeeze(0)
                exp_avg_sq.mul_(beta2).add_(sq_grad, alpha=1 - beta2)
                p.addcdiv_(
                    exp_avg, (exp_avg_sq.sqrt().add_(group["eps"])), value=-group["lr"]
                )

        return loss
