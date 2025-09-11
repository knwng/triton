#! /bin/bash

# export TRITON_LLVM_DEBUG_ONLY="tritonamdgpu-stream-pipeline"
# export TRITON_LLVM_DEBUG_ONLY="tritonamdgpu-block-pingpong"
# export TRITON_INTERPRET=1
# export MLIR_ENABLE_DUMP=1
# export LLVM_IR_ENABLE_DUMP=1
# export TRITON_ENABLE_LLVM_DEBUG=1

rm -rf ~/.triton/cache/

# amd-smi monitor -putmvq

RUN=""

path="mi350_runs"
seq_q_l=4096
seq_kv_l=4096
num_heads_q=64
num_heads_k=8
batch_size=2
window_size=0
block_size=64
repeat=500
export TRITON_HIP_USE_ASYNC_COPY=1
export TRITON_HIP_USE_BLOCK_PINGPONG=1

mkdir -p $path
rocprofv3 -o results_attn.csv --output-format csv --runtime-trace -- python run_kernel.py --window_size $window_size --block_size $block_size \
                                       --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $batch_size \
                                       --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --repeat $repeat

python collect_results.py --window_size $window_size --block_size $block_size \
                          --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $batch_size \
                          --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l --path $path --repeat $repeat \
                          --kernel_names kernel_unified_attention_3d reduce_segments kernel_unified_attention_2d
