#!/bin/bash
# tmux_train_free.sh
# 同时用 GPU 0 和 GPU 1 训练 free 环境
#
# 使用方法:
#   export WANDB_API_KEY="your_wandb_key"
#   bash tmux_train_free.sh

SESSION="her_training_free"
cd "$(dirname "$0")"

ENTITY="${WANDB_ENTITY:-1906402677-nanchang-university}"
PROJECT="${WANDB_PROJECT:-her}"

NP_PER_GPU=8
NUM_ROLLOUTS_PER_MPI=2
N_EPOCHS=200
N_CYCLES=50
N_BATCHES=40
GIF_INTERVAL=10
SEED=123

if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY 未设置。"
    sleep 3
fi

tmux kill-session -t $SESSION 2>/dev/null

echo "============================================"
echo "Free environment training on 2 GPUs"
echo "============================================"

# ============================================
# Window 0: GPU 0 -> FetchReach, FetchPush
# ============================================
tmux new-session -d -s $SESSION -n "free_gpu0"
tmux send-keys -t $SESSION:0 "
set -e
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=\${WANDB_API_KEY}
export WANDB_ENTITY=$ENTITY
export WANDB_PROJECT=$PROJECT

echo '[Free GPU0] Starting FetchReach + FetchPush'

for ENV in FetchReach-v4 FetchPush-v4; do
  echo ''
  echo '========================================'
  echo '[Free] Starting '\$ENV' on GPU 0'
  echo '========================================'

  conda run -n her mpirun -np $NP_PER_GPU python -u train.py \\
    --env-name \${ENV} \\
    --env-config free \\
    --n-epochs $N_EPOCHS \\
    --n-cycles $N_CYCLES \\
    --n-batches $N_BATCHES \\
    --seed $SEED \\
    --num-rollouts-per-mpi $NUM_ROLLOUTS_PER_MPI \\
    --save-gif-interval $GIF_INTERVAL \\
    --wandb-entity $ENTITY \\
    --wandb-project $PROJECT \\
    --wandb-mode disabled \\
    --cuda

  echo '[Free] Finished '\$ENV
  echo 'Policy GIFs -> figures/free/'\$(echo \$ENV | sed 's/Fetch//;s/-v4//')'/seed${SEED}/policy_gifs/'
done

echo '[Free GPU0] Done.'
" C-m

# ============================================
# Window 1: GPU 1 -> PickAndPlace, Slide
# ============================================
tmux new-window -t $SESSION -n "free_gpu1"
tmux send-keys -t $SESSION:1 "
set -e
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=\${WANDB_API_KEY}
export WANDB_ENTITY=$ENTITY
export WANDB_PROJECT=$PROJECT

echo '[Free GPU1] Starting PickAndPlace + Slide'

for ENV in FetchPickAndPlace-v4 FetchSlide-v4; do
  echo ''
  echo '========================================'
  echo '[Free] Starting '\$ENV' on GPU 1'
  echo '========================================'

  conda run -n her mpirun -np $NP_PER_GPU python -u train.py \\
    --env-name \${ENV} \\
    --env-config free \\
    --n-epochs $N_EPOCHS \\
    --n-cycles $N_CYCLES \\
    --n-batches $N_BATCHES \\
    --seed $SEED \\
    --num-rollouts-per-mpi $NUM_ROLLOUTS_PER_MPI \\
    --save-gif-interval $GIF_INTERVAL \\
    --wandb-entity $ENTITY \\
    --wandb-project $PROJECT \\
    --wandb-mode online \\
    --cuda

  echo '[Free] Finished '\$ENV
  echo 'Policy GIFs -> figures/free/'\$(echo \$ENV | sed 's/Fetch//;s/-v4//')'/seed${SEED}/policy_gifs/'
done

echo '[Free GPU1] Done.'
echo '[Free] All training done. Generating local plots...'
conda run -n her python plot_local_results.py --env-config free --output-dir figures/20250511_free_green
echo '[Free] Plots saved to figures/20250511_free_green/'
" C-m

echo "Free training started in tmux session: $SESSION"
echo "Attach : tmux attach -t $SESSION"
