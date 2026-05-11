#!/bin/bash
# tmux_train_all.sh
# 同时训练 2Staticobs (GPU 0) 和 2Dynamicobs (GPU 1)
# 针对 2×5090 (32GB) + 208 CPU 核心优化
#
# 使用方法:
#   export WANDB_API_KEY="your_wandb_key"
#   bash tmux_train_all.sh
#
# 进入 tmux 后:
#   Ctrl+B, 0  -> 查看 2Staticobs (GPU 0) 窗口
#   Ctrl+B, 1  -> 查看 2Dynamicobs (GPU 1) 窗口
#   Ctrl+B, d  -> detach 离开 (训练在后台继续)

SESSION="her_training"
cd "$(dirname "$0")"

# W&B 配置（可从环境变量读取，有默认值）
ENTITY="${WANDB_ENTITY:-1906402677-nanchang-university}"
PROJECT="${WANDB_PROJECT:-her}"

# ===================== 性能调参区 =====================
# 208 CPU 核 + 32GB 显存，可以开很多 MPI worker 并行采样
# 每张 GPU 上的 MPI 进程数。建议 4~16，越大采样越快，但 MPI 通信开销会增加
NP_PER_GPU=8

# 每个 MPI 进程每 cycle 采集的 episode 数
# 总采样量 = NP_PER_GPU × NUM_ROLLOUTS_PER_MPI × T (每 episode 步数)
NUM_ROLLOUTS_PER_MPI=2

# 训练超参数
N_EPOCHS=200
N_CYCLES=50
N_BATCHES=40
OBSTACLE_WEIGHT=0.2
SAFE_DISTANCE=0.10
DENSE_WEIGHT=0.0
COLLISION_PEN=0.0
COLLISION_DIST=0.02
GIF_INTERVAL=10
SEED=123
# ====================================================

# 检查 WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY 未设置。如果已用 'wandb login' 登录过，可忽略此警告。"
    echo "否则请执行: export WANDB_API_KEY=your_key"
    sleep 3
fi

# 清理旧 session
tmux kill-session -t $SESSION 2>/dev/null

echo "============================================"
echo "Hardware: 2×5090(32GB) + 208 CPU cores"
echo "MPI workers per GPU: $NP_PER_GPU"
echo "Rollouts per worker: $NUM_ROLLOUTS_PER_MPI"
echo "Total rollouts/cycle/GPU: $((NP_PER_GPU * NUM_ROLLOUTS_PER_MPI))"
echo "============================================"
echo ""
echo "Attach  : tmux attach -t $SESSION"
echo "Detach  : Ctrl+B, d"
echo "Switch  : Ctrl+B, 0  or  Ctrl+B, 1"
echo ""

# ============================================
# Window 0: 2Staticobs  on GPU 0
# ============================================
tmux new-session -d -s $SESSION -n "2Staticobs"
tmux send-keys -t $SESSION:0 "
set -e
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=\${WANDB_API_KEY}
export WANDB_ENTITY=$ENTITY
export WANDB_PROJECT=$PROJECT

echo '[2Staticobs] GPU visible: '\$CUDA_VISIBLE_DEVICES

for ENV in FetchReach-v4 FetchPush-v4 FetchPickAndPlace-v4 FetchSlide-v4; do
  echo ''
  echo '========================================'
  echo '[2Staticobs] Starting '\$ENV' on GPU 0'
  echo '========================================'

  conda run -n her mpirun -np $NP_PER_GPU python -u train.py \\
    --env-name \${ENV} \\
    --env-config 2Staticobs \\
    --n-epochs $N_EPOCHS \\
    --n-cycles $N_CYCLES \\
    --n-batches $N_BATCHES \\
    --seed $SEED \\
    --num-rollouts-per-mpi $NUM_ROLLOUTS_PER_MPI \\
    --obstacle-reward-weight $OBSTACLE_WEIGHT \\
    --obstacle-safe-distance $SAFE_DISTANCE \\
    --dense-reward-weight $DENSE_WEIGHT \\
    --collision-penalty $COLLISION_PEN \\
    --collision-distance $COLLISION_DIST \\
    --save-gif-interval $GIF_INTERVAL \\
    --wandb-entity $ENTITY \\
    --wandb-project $PROJECT \\
    --wandb-mode online \\
    --cuda

  echo '[2Staticobs] Finished '\$ENV
  echo 'Policy GIFs -> figures/2Staticobs/'\$(echo \$ENV | sed 's/Fetch//;s/-v4//')'/seed${SEED}/policy_gifs/'
done

echo ''
echo '[2Staticobs] All training done. Generating W&B plots...'
WANDB_API_KEY=\${WANDB_API_KEY} conda run -n her python plot_results.py --entity $ENTITY --project $PROJECT --env-config 2Staticobs
echo '[2Staticobs] Plots saved to figures/'
" C-m

# ============================================
# Window 1: 2Dynamicobs on GPU 1
# ============================================
tmux new-window -t $SESSION -n "2Dynamicobs"
tmux send-keys -t $SESSION:1 "
set -e
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=\${WANDB_API_KEY}
export WANDB_ENTITY=$ENTITY
export WANDB_PROJECT=$PROJECT

echo '[2Dynamicobs] GPU visible: '\$CUDA_VISIBLE_DEVICES

for ENV in FetchReach-v4 FetchPush-v4 FetchPickAndPlace-v4 FetchSlide-v4; do
  echo ''
  echo '========================================'
  echo '[2Dynamicobs] Starting '\$ENV' on GPU 1'
  echo '========================================'

  conda run -n her mpirun -np $NP_PER_GPU python -u train.py \\
    --env-name \${ENV} \\
    --env-config 2Dynamicobs \\
    --n-epochs $N_EPOCHS \\
    --n-cycles $N_CYCLES \\
    --n-batches $N_BATCHES \\
    --seed $SEED \\
    --num-rollouts-per-mpi $NUM_ROLLOUTS_PER_MPI \\
    --obstacle-reward-weight $OBSTACLE_WEIGHT \\
    --obstacle-safe-distance $SAFE_DISTANCE \\
    --dense-reward-weight $DENSE_WEIGHT \\
    --collision-penalty $COLLISION_PEN \\
    --collision-distance $COLLISION_DIST \\
    --save-gif-interval $GIF_INTERVAL \\
    --wandb-entity $ENTITY \\
    --wandb-project $PROJECT \\
    --wandb-mode online \\
    --cuda

  echo '[2Dynamicobs] Finished '\$ENV
  echo 'Policy GIFs -> figures/2Dynamicobs/'\$(echo \$ENV | sed 's/Fetch//;s/-v4//')'/seed${SEED}/policy_gifs/'
done

echo ''
echo '[2Dynamicobs] All training done. Generating W&B plots...'
WANDB_API_KEY=\${WANDB_API_KEY} conda run -n her python plot_results.py --entity $ENTITY --project $PROJECT --env-config 2Dynamicobs
echo '[2Dynamicobs] Plots saved to figures/'
" C-m

# 训练已在后台 tmux session 中启动
# 手动 attach: tmux attach -t $SESSION
echo "Training started in tmux session: $SESSION"
