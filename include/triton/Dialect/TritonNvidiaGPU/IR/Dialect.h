/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_IR_DIALECT_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

// TritonNvidiaGPU depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h.inc"

namespace mlir::triton::nvidia_gpu::impl {
LogicalResult verifyMMAv5Op(Operation *op);
} // namespace mlir::triton::nvidia_gpu::impl

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.h.inc"

#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/Ops.h.inc"

namespace mlir::triton::nvidia_gpu {

struct TensorMemory : public SideEffects::Resource::Base<TensorMemory> {
  StringRef getName() final { return "<TensorMemory>"; }
};

struct TMemAllocation {
  TMemAllocation(int numCols, int numRows)
      : numCols(numCols), numRows(numRows) {}
  int numCols;
  int numRows;
};

TMemAllocation getTmemAllocSizes(gpu::MemDescType memDescType);

gpu::DistributedEncodingTrait getTmemCompatibleLayout(unsigned M, unsigned N,
                                                      RankedTensorType oltType,
                                                      unsigned numWarps);
gpu::DistributedEncodingTrait
getTmemLoadLayoutSplitLongM(RankedTensorType tensorType,
                            gpu::MemDescType memType, int numWarps);
SmallVector<gpu::DistributedEncodingTrait>
getTmemCompatibleLayouts(Operation *op, RankedTensorType tensorType,
                         gpu::MemDescType memType);

bool isDistributedLayoutTMemCompatible(Operation *op,
                                       RankedTensorType tensorType,
                                       gpu::MemDescType memType);
bool isDistributedLayoutSplitMTmemLoadStore(RankedTensorType tensorType,
                                            gpu::MemDescType memType,
                                            int numWarps);

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_IR_DIALECT_H_
