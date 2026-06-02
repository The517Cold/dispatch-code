# LR Schedule Validation

## Change

The previous PPO learning-rate schedule used a linear decay:

```text
lr = max(1e-5, initial_lr * (1 - progress))
```

With the default `initial_lr=3e-4`, this reached `1e-5` at the end of training.
The new default is a slower cosine schedule:

```text
lr = floor_lr + (initial_lr - floor_lr) * 0.5 * (1 + cos(pi * progress / decay_horizon))
floor_lr = initial_lr * lr_min_ratio
```

Default parameters:

```text
GCN_PPO_HQ_LR_SCHEDULE=cosine
GCN_PPO_HQ_LR_MIN_RATIO=0.35
GCN_PPO_HQ_LR_DECAY_HORIZON=1.5
```

The old behavior remains available with:

```text
GCN_PPO_HQ_LR_SCHEDULE=linear
GCN_PPO_HQ_LR_MIN_RATIO=0.03333333333333333
GCN_PPO_HQ_LR_DECAY_HORIZON=1.0
```

## Validation Setup

Both runs used the same seed and short smoke-training configuration:

```text
GCN_PPO_HQ_SEED=123
GCN_PPO_HQ_TRAIN_FILES=1-2-1.txt
GCN_PPO_HQ_AUTO_EVAL=0
GCN_PPO_HQ_MAX_TRAIN_STEPS=512
GCN_PPO_HQ_STEPS_PER_EPOCH=256
GCN_PPO_HQ_MINIBATCH_SIZE=64
GCN_PPO_HQ_PPO_EPOCHS=1
GCN_PPO_HQ_LAMBDA_P=64
GCN_PPO_HQ_LAMBDA_T=32
GCN_PPO_HQ_EXTRA_P2T_ROUNDS=2
GCN_PPO_HQ_USE_DEADLOCK_CONTROLLER=0
GCN_PPO_HQ_REUSE=0
GCN_PPO_HQ_IL_WARMSTART=0
```

## LR Curve Comparison

| Epoch | Progress | Linear LR | Cosine LR |
| --- | ---: | ---: | ---: |
| 1 | 50% | 0.000150 | 0.000251 |
| 2 | 100% | 0.000010 | 0.000154 |

The cosine schedule keeps the LR about 1.67x higher at 50% progress and about
15.4x higher at the end of this run.

## Performance Comparison

| Metric | Linear | Cosine |
| --- | ---: | ---: |
| Epoch 1 train loss | 13.5963 | 13.5963 |
| Epoch 2 train loss | 17.5710 | 17.5706 |
| Epoch 1 avg reward | -9.65 | -9.65 |
| Epoch 2 avg reward | -113.42 | -113.42 |
| Epoch 1 eval makespan | 62429 | 62429 |
| Epoch 2 eval makespan | 62429 | 62476 |
| Final makespan | 56791 | 56791 |
| Final seen-pool success | 1.0 | 1.0 |
| Final seen-pool avg makespan | 56791 | 56791 |

## Result

The LR curve now decays substantially more slowly while preserving the smoke-run
training behavior under a fixed seed. The 512-step run is intentionally short, so
it verifies schedule correctness and basic stability rather than long-horizon
quality. A longer run should be used to judge whether the higher late-training LR
improves convergence or final makespan on the full training pool.
