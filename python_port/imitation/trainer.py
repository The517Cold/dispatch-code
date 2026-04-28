import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class BCTrainerConfig:
    epochs: int
    lr: float
    weight_decay: float
    label_smoothing: float
    device: str
    checkpoint_path: str
    rollout_every_n_epochs: int
    log_interval: int = 1
    # If set, each training epoch appends one line (UTF-8) to this file.
    progress_log_path: Optional[str] = None
    # DAgger-lite 复用同一训练器时，可通过不同前缀区分日志来源。
    log_prefix: str = "[BC]"


def is_better_rollout(curr: Dict[str, object], best: Optional[Dict[str, object]]) -> bool:
    """Ranking rule for checkpoint selection: goal first, then quality."""
    if best is None:
        return True
    curr_goal = bool(curr.get("reach_goal", False))
    best_goal = bool(best.get("reach_goal", False))
    if curr_goal != best_goal:
        return curr_goal and (not best_goal)
    if curr_goal:
        curr_ms = float(curr.get("policy_makespan", 1e18))
        best_ms = float(best.get("policy_makespan", 1e18))
        if curr_ms != best_ms:
            return curr_ms < best_ms
        curr_steps = int(curr.get("policy_trans_count", 10 ** 9))
        best_steps = int(best.get("policy_trans_count", 10 ** 9))
        return curr_steps < best_steps
    curr_dist = int(curr.get("goal_distance", 10 ** 9))
    best_dist = int(best.get("goal_distance", 10 ** 9))
    if curr_dist != best_dist:
        return curr_dist < best_dist
    curr_ms = float(curr.get("policy_makespan", 1e18))
    best_ms = float(best.get("policy_makespan", 1e18))
    return curr_ms < best_ms


class BCTrainer:
    def __init__(self, model, config: BCTrainerConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.best_rollout = None
        self.best_epoch = 0

    def _compute_batch(self, batch):
        states = batch["state_features"]
        if hasattr(states, "to"):
            states = states.to(self.device)
        masks = batch["action_mask"].to(self.device)
        targets = batch["expert_action"].to(self.device)
        logits = self.model(states)
        # Illegal actions are suppressed before both loss and metrics are computed.
        masked_logits = logits.masked_fill(~masks, -1e9)
        target_valid = masks.gather(1, targets.unsqueeze(1)).squeeze(1)
        valid_count = int(target_valid.sum().item())
        total_count = int(target_valid.numel())
        invalid_count = total_count - valid_count
        if valid_count > 0:
            valid_logits = masked_logits[target_valid]
            valid_targets = targets[target_valid]
            # Train only on samples whose expert label is still enabled in that state.
            loss = F.cross_entropy(valid_logits, valid_targets, label_smoothing=self.config.label_smoothing)
        else:
            loss = torch.zeros((), dtype=torch.float32, device=self.device, requires_grad=True)
        with torch.no_grad():
            pred_top1 = torch.argmax(masked_logits, dim=1)
            top1_correct = (pred_top1 == targets).float()
            top1_correct = top1_correct[target_valid] if valid_count > 0 else top1_correct[:0]
            k = min(3, masked_logits.shape[1])
            topk_indices = torch.topk(masked_logits, k=k, dim=1).indices
            topk_hit = (topk_indices == targets.unsqueeze(1)).any(dim=1).float()
            topk_hit = topk_hit[target_valid] if valid_count > 0 else topk_hit[:0]
            pred_valid = masks.gather(1, pred_top1.unsqueeze(1)).squeeze(1).float()
            enabled_count = masks.sum(dim=1).float().clamp_min(1.0)
            # Random baseline is useful because some states have very few enabled actions.
            random_acc = (1.0 / enabled_count)
            random_acc = random_acc[target_valid] if valid_count > 0 else random_acc[:0]
            avg_enabled = enabled_count.mean().item() if enabled_count.numel() > 0 else 0.0
        return {
            "loss": loss,
            "valid_count": valid_count,
            "total_count": total_count,
            "invalid_count": invalid_count,
            "top1_sum": float(top1_correct.sum().item()),
            "topk_sum": float(topk_hit.sum().item()),
            "mask_valid_sum": float(pred_valid.sum().item()),
            "random_acc_sum": float(random_acc.sum().item()),
            "avg_enabled_sum": float(avg_enabled * total_count),
        }

    def _run_epoch(self, loader, train: bool):
        self.model.train(mode=train)
        loss_sum = 0.0
        top1_sum = 0.0
        topk_sum = 0.0
        mask_valid_sum = 0.0
        random_acc_sum = 0.0
        avg_enabled_sum = 0.0
        total_valid = 0
        total_samples = 0
        invalid_count = 0
        for batch in loader:
            if train:
                self.optimizer.zero_grad()
            ctx = torch.enable_grad() if train else torch.no_grad()
            with ctx:
                out = self._compute_batch(batch)
                if train:
                    out["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                    self.optimizer.step()
            total_samples += out["total_count"]
            total_valid += out["valid_count"]
            invalid_count += out["invalid_count"]
            loss_sum += float(out["loss"].item()) * max(1, out["valid_count"])
            top1_sum += out["top1_sum"]
            topk_sum += out["topk_sum"]
            mask_valid_sum += out["mask_valid_sum"]
            random_acc_sum += out["random_acc_sum"]
            avg_enabled_sum += out["avg_enabled_sum"]
        denom_valid = max(1, total_valid)
        denom_total = max(1, total_samples)
        return {
            "loss": loss_sum / denom_valid,
            "action_top1_acc": top1_sum / denom_valid,
            "topk_acc@3": topk_sum / denom_valid,
            "mask_valid_rate": mask_valid_sum / denom_total,
            "random_baseline_acc": random_acc_sum / denom_valid,
            "avg_enabled_actions": avg_enabled_sum / denom_total,
            "invalid_label_rate": float(invalid_count) / float(denom_total),
            "valid_samples": total_valid,
            "total_samples": total_samples,
        }

    def _save_checkpoint(self, epoch, train_metrics, val_metrics, rollout_metrics):
        if not self.config.checkpoint_path:
            return
        os.makedirs(os.path.dirname(self.config.checkpoint_path), exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "rollout_metrics": rollout_metrics,
            "best_epoch": self.best_epoch,
        }
        torch.save(payload, self.config.checkpoint_path)

    def fit(
        self,
        train_loader,
        val_loader,
        rollout_fn: Optional[Callable[[], Dict[str, object]]] = None,
    ):
        history = {"train": [], "val": [], "rollout": []}
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)
            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            rollout_metrics = None
            should_rollout = rollout_fn is not None and (
                epoch == 1
                or epoch == self.config.epochs
                or epoch % max(1, self.config.rollout_every_n_epochs) == 0
            )
            if should_rollout:
                # Rollout is the real task metric, so only rollout-improving epochs are checkpointed.
                rollout_metrics = rollout_fn()
                history["rollout"].append({"epoch": epoch, **rollout_metrics})
                if is_better_rollout(rollout_metrics, self.best_rollout):
                    self.best_rollout = rollout_metrics
                    self.best_epoch = epoch
                    self._save_checkpoint(epoch, train_metrics, val_metrics, rollout_metrics)

            line = (
                self.config.log_prefix
                + " epoch="
                + str(epoch)
                + "/"
                + str(self.config.epochs)
                + " train_loss="
                + format(train_metrics["loss"], ".4f")
                + " val_loss="
                + format(val_metrics["loss"], ".4f")
                + " val_top1="
                + format(val_metrics["action_top1_acc"], ".4f")
                + " val_top3="
                + format(val_metrics["topk_acc@3"], ".4f")
                + " invalid_label_rate="
                + format(val_metrics["invalid_label_rate"], ".4f")
            )
            if rollout_metrics is not None:
                line += (
                    " rollout_goal="
                    + ("1" if rollout_metrics.get("reach_goal") else "0")
                    + " rollout_ms="
                    + format(float(rollout_metrics.get("policy_makespan", -1.0)), ".3f")
                    + " rollout_gd="
                    + str(int(rollout_metrics.get("goal_distance", -1)))
                )
            else:
                line += " rollout_goal=na rollout_ms=na rollout_gd=na"
            log_path = self.config.progress_log_path
            if log_path:
                parent = os.path.dirname(log_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            if epoch == 1 or epoch == self.config.epochs or epoch % max(1, self.config.log_interval) == 0:
                print(line, flush=True)
        return {
            "history": history,
            "best_rollout": self.best_rollout,
            "best_epoch": self.best_epoch,
            "checkpoint_path": self.config.checkpoint_path,
        }
