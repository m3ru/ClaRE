"""Numerically stable PPOTrainer subclass.

Root cause of the NaN collapse: TRL's PPOTrainer.loss() computes
    ratio = exp(logprobs - old_logprobs)
without clamping the log-ratio first. During PPO epochs 2-4, the model's
logprobs can drift far from old_logprobs, producing inf/NaN ratios.
The NaN propagates through backward() into optimizer.step(), permanently
corrupting the model weights.

TRL's ratio_threshold check doesn't help because masked_mean of NaN
ratios returns NaN, and NaN > threshold is False.

Additional TRL bug: self.model_params is a one-shot filter() iterator
that gets consumed by the first call to clip_grad_norm_. All subsequent
minibatches have no gradient clipping and no NaN detection.

Fix: override loss() to clamp the log-ratio and compute in float32,
and override train_minibatch() with a fresh parameter list each time
to detect NaN gradients and skip updates.
"""

import warnings

import torch
import torch.nn.functional as F
from trl import PPOTrainer
from trl.core import (
    clip_by_value,
    entropy_from_logits,
    flatten_dict,
    masked_mean,
    masked_var,
)

LOG_RATIO_CLAMP = 5.0  # exp(5) ≈ 148, exp(-5) ≈ 0.007; safe range for ratios


class StablePPOTrainer(PPOTrainer):
    """PPOTrainer with numerical stability fixes in the loss and update step."""

    def _get_trainable_params(self):
        """Return a fresh list of trainable parameters every time.

        TRL stores self.model_params as a filter() iterator that is exhausted
        after one use. This method always returns a fresh list.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        # Upcast everything to float32 to avoid bfloat16 precision loss
        old_logprobs = old_logprobs.float()
        logprobs = logprobs.float()
        values = values.float()
        vpreds = vpreds.float()
        advantages = advantages.float()
        returns = returns.float()
        mask = mask.float()

        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        # Clamp the log-ratio before exp to prevent inf/NaN
        log_ratio = logprobs - old_logprobs
        log_ratio_clamped = torch.clamp(log_ratio, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
        ratio = torch.exp(log_ratio_clamped)

        # Track how many tokens got clamped for diagnostics
        n_clamped = ((log_ratio < -LOG_RATIO_CLAMP) | (log_ratio > LOG_RATIO_CLAMP)).float()
        frac_clamped = masked_mean(n_clamped, mask)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange
        )

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold "
                f"{self.config.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0

        # If loss is still NaN/Inf (e.g. from NaN inputs), replace with a
        # fresh zero that has a clean autograd graph. NaN * 0.0 = NaN in
        # IEEE 754, so we must create a new tensor.
        if not torch.isfinite(loss):
            warnings.warn(f"[StablePPO] Loss is {loss.item()}, replacing with zero.")
            zero = torch.zeros(1, device=loss.device, dtype=loss.dtype, requires_grad=True)
            pg_loss = zero
            vf_loss = zero
            loss = zero

        entropy = masked_mean(entropy_from_logits(logits.float()), mask)

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
                log_ratio_clamp_frac=frac_clamped.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)

    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        self.model.train()
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        loss = loss_p + loss_v
        self.accelerator.backward(loss)

        # Use a fresh parameter list (self.model_params is a consumed iterator)
        params = self._get_trainable_params()

        # Check for NaN/Inf in gradients before stepping
        has_nan_grad = False
        for p in params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_nan_grad = True
                break

        if has_nan_grad:
            print(
                f"[StablePPO] NaN/Inf gradient detected — skipping optimizer step "
                f"(loss_p={loss_p.item():.6f}, loss_v={loss_v.item():.6f})",
                flush=True,
            )
            self.optimizer.zero_grad()
        else:
            if self.config.max_grad_norm is not None:
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(params, self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return train_stats
