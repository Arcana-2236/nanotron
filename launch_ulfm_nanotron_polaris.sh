#!/bin/bash -l
#PBS -N ulfm_pytorch
#PBS -A TensorCompress
#PBS -q prod
#PBS -l select=32:ncpus=64:ngpus=4:mpiprocs=4
#PBS -l walltime=6:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe

# set -euo pipefail

# --- Environment ---
source ~/.bash_profile
ompi2-pytorch

which mpiexec

cd /eagle/TensorCompress/$USER/project/pytorch/mpi_ulfm_extension/nanotron

CONFIG=${CONFIG:-"config_llama_7b_ulfm"}
DP=${DP:-16}

# --- Failure simulator env vars (unset → flag omitted → Python default) ---
# FSIM_SEED, FSIM_DESIRED_FAILURES, FSIM_TOTAL_MINIBATCHES,
# FSIM_TARGET_RANKS, FSIM_CONFIG, FSIM_START_MINIBATCH
FSIM_ARGS=()
[[ -n "${FSIM_SEED:-}" ]]               && FSIM_ARGS+=(--failure-sim-seed "$FSIM_SEED")
[[ -n "${FSIM_DESIRED_FAILURES:-}" ]]   && FSIM_ARGS+=(--failure-sim-desired-failures "$FSIM_DESIRED_FAILURES")
[[ -n "${FSIM_TOTAL_MINIBATCHES:-}" ]]  && FSIM_ARGS+=(--failure-sim-total-minibatches "$FSIM_TOTAL_MINIBATCHES")
[[ -n "${FSIM_TARGET_RANKS:-}" ]]       && FSIM_ARGS+=(--failure-sim-target-ranks "$FSIM_TARGET_RANKS")
[[ -n "${FSIM_CONFIG:-}" ]]             && FSIM_ARGS+=(--failure-sim-config "$FSIM_CONFIG")
[[ -n "${FSIM_START_MINIBATCH:-}" ]]    && FSIM_ARGS+=(--failure-sim-start-minibatch "$FSIM_START_MINIBATCH")

# --- Run CoLA Nanotron ---
LOGDIR="/eagle/TensorCompress/$USER/project/pytorch/mpi_ulfm_extension/nanotron/.logging/$(date +%Y%m%d)"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/ulfm_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID}.log"
echo "LOGFILE=$LOGFILE"

export TMPDIR=/tmp

# Resolve MASTER_ADDR to the first node in the PBS allocation so
# the TCPStore (needed for NCCL subgroup ncclUniqueId exchange) is
# reachable from all nodes.
export MASTER_ADDR=$(head -1 "$PBS_NODEFILE")
export MASTER_PORT=$(( RANDOM + 1000 ))

NCCL_PP_TIMEOUT_SECONDS=20 NCCL_MP_TIMEOUT_SECONDS=120 NCCL_PG_TIMEOUT_SECONDS=20 \
    TORCH_NCCL_PROPAGATE_ERROR=0 TORCH_NCCL_DUMP_ON_TIMEOUT=0 \
    mpiexec --mca pml ob1 --mca btl tcp,self,sm -n $((NNODES * NPROC_PER_NODE)) --hostfile "$PBS_NODEFILE" --with-ft ulfm --map-by ppr:4:node:PE=8 --bind-to core \
    python run_train_ulfm.py --config-file examples/$CONFIG.yaml --dp $DP "${FSIM_ARGS[@]}" \
    > "$LOGFILE" 2>&1