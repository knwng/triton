/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
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
#include "../PatternTritonGPUOpToLLVM.h"
#include "../TritonAMDGPUToLLVM/SchedInstructions.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#include <sstream>

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::LLVM::AMD::scaleDotElemTypeToMLIRType;
using ::mlir::LLVM::AMD::shuffleXor;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::LinearEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

using ValueTable = std::map<std::array<int, 3>, Value>;

/// Get matrix format flag passed through BLGP/CBSZ args in V_MFMA_*_F8F6F4
/// instructions.
///
/// Values:
/// - 0: E4M3(FP8)
/// - 1: E5M2(BF8)
/// - 2: E2M3(FP6)
/// - 3: E3M2(BF6)
/// - 4: E2M1(FP4)
static inline int32_t getMfmaF8F6F4MatrixFormat(mlir::Type t) {
  if (llvm::isa<Float8E4M3FNUZType>(t)) {
    return 0;
  }
  if (llvm::isa<Float8E5M2FNUZType>(t)) {
    return 1;
  }
  if (llvm::isa<Float6E3M2FNType>(t)) {
    return 2;
  }
  if (llvm::isa<Float6E2M3FNType>(t)) {
    return 3;
  }
  if (llvm::isa<Float4E2M1FNType>(t)) {
    return 4;
  }
  return -1;
}

struct DotOpMFMAConversionHelper {
  AMDMfmaEncodingAttr mfmaLayout;

  ConversionPatternRewriter &rewriter;
  const LLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConversionHelper(AMDMfmaEncodingAttr mfmaLayout,
                                     ConversionPatternRewriter &rewriter,
                                     const LLVMTypeConverter *typeConverter,
                                     Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(mfmaLayout.getContext()) {}

  Value getThreadId() const {
    auto llvmIndexTy = typeConverter->getIndexType();
    auto tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x);
    return rewriter.create<arith::TruncIOp>(loc, i32_ty, tid);
  }

  Value generateMFMAOp(StringRef mfmaInsnName, Value valA, Value valB,
                       Value valC) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resType = valC.getType();
    Value zeroFlag = b.i32_val(0);
    OperationState loweredOp(loc, mfmaInsnName);
    loweredOp.addTypes(resType);
    loweredOp.addOperands({valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    return rewriter.create(loweredOp)->getResult(0);
  }

  int getNumSubmatrices(Type elementType, int mDim, int nDim) const {
    if ((mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64))
      return 1;
    assert(mDim == nDim);
    switch (mDim) {
    case 32:
    case 16:
      return 1;
      break;
    case 4:
      assert(elementType.getIntOrFloatBitWidth() <= 32 &&
             "fp64 is not supported yet");
      assert(elementType.getIntOrFloatBitWidth() != 8 ||
             elementType.isInteger(8) && "fp8 is not supported yet");
      return 16;
      break;
    default:
      llvm::report_fatal_error("unsupported nonKDim in MFMA dot");
    }
    return -1;
  }

  Value processSubBlocks(int numSubBlocks, Value acc, bool reduceSubBlocks,
                         bool zeroSubBlocks) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert((numSubBlocks & (numSubBlocks - 1)) == 0 &&
           "numSubBlocks in not pow 2!");
    if (numSubBlocks == 1)
      return acc;
    constexpr int warpSize = 64;
    int subBlockSize = warpSize / numSubBlocks;
    Value laneId = getThreadId();
    laneId = b.and_(laneId, b.i32_val(warpSize - 1));
    auto vecTy = dyn_cast<VectorType>(acc.getType());
    auto elemType = vecTy.getElementType();
    assert(elemType.getIntOrFloatBitWidth() == 32);
    int numScalars = vecTy.getNumElements();
    std::vector<Value> accScalar(numScalars);
    for (int i = 0; i < numScalars; ++i)
      accScalar[i] = b.extract_element(elemType, acc, b.i32_val(i));

    if (reduceSubBlocks) {
      while (subBlockSize < warpSize) {
        for (int i = 0; i < numScalars; ++i) {
          Value other_acc =
              shuffleXor(loc, rewriter, accScalar[i], subBlockSize);
          if (elemType.isInteger(32))
            accScalar[i] = b.add(accScalar[i], other_acc);
          else
            accScalar[i] = b.fadd(accScalar[i], other_acc);
        }
        subBlockSize *= 2;
      }
    }
    if (zeroSubBlocks) {
      Value zero;
      if (elemType.isInteger(32))
        zero = b.i32_val(0);
      else
        zero = b.f32_val(0.0);
      auto cond = b.icmp_ult(laneId, b.i32_val(subBlockSize));
      for (int i = 0; i < numScalars; ++i)
        accScalar[i] = b.select(cond, accScalar[i], zero);
    }

    Value reducedAcc = b.undef(vecTy);
    for (int i = 0; i < numScalars; ++i)
      reducedAcc =
          b.insert_element(vecTy, reducedAcc, accScalar[i], b.i32_val(i));
    return reducedAcc;
  }

  /// @brief MFMA 4x4 is computes 16 matrix multiplications, this functions adds
  /// these 16 matrices to get final 4x4 matrix
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value reduceSubBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, true, false);
  }

  /// @brief Zeroes out redundant values in all sub-blocks except first one
  ///
  /// Every warp in mfma 4x4 layout holds only 4 unique values(scalar or
  /// vectors) in blocks of 4 consecutive threads, There are 16 copies of these
  /// 4 values across all threads of the warp. Need to zero out 15 copies to use
  /// accumulator between dot operations.
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value zeroAuxiliarBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, false, true);
  }

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    // Check if this dot has come with priority set by setprio.
    auto setPrioOp = dyn_cast_or_null<ROCDL::SetPrioOp>(op->getPrevNode());

    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();
    auto mfmaVersion = mfmaLayout.getVersionMajor();
    assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
           (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));

    Value a = op.getA();
    Value b = op.getB();
    Value d = op.getD();
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());
    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();

    bool allowXF32 =
        op.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
    StringRef mfmaInsnName;
    auto maybeMfmaInsn = MfmaInsn::selectMfma(mDim, nDim, elemTyA, elemTyB,
                                              mfmaVersion, allowXF32);
    if (failed(maybeMfmaInsn))
      llvm::report_fatal_error("No match found in MFMA database\n");

    mfmaInsnName = maybeMfmaInsn->getInsnName();
    unsigned kBase = maybeMfmaInsn->getKBase();

    auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
    int kWidth = aEncoding.getKWidth();

    // If we are using XF32, the kWidth (and kBase) is double that of F32.
    if (aTensorTy.getElementType().isF32() && allowXF32)
      kWidth *= 2;

    auto rank = aTensorTy.getShape().size();
    const auto kDimOperandSize = aTensorTy.getShape()[rank - 1];
    const auto kDimInstrSize = mfmaLayout.getInstrShapeForOperand(kWidth, 0)[1];

    auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), kWidth, 0);
    auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), kWidth, 1);

    assert(repA[2] == repB[1]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[1];
    auto numRepN = repB[2];
    auto numRepK = repA[2];
    auto numRepB = repA[0];
    assert(repA[0] == repB[0]);

    auto operandA = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepB, numRepM, numRepK, kWidth, kBase,
        aTensorTy.getElementType(), allowXF32);
    auto operandB = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepB, numRepN, numRepK, kWidth, kBase,
        aTensorTy.getElementType(), allowXF32);

    auto dstElemTy = dTensorTy.getElementType();
    auto fc = unpackLLElements(loc, loadedC, rewriter);

    unsigned warpSize = triton::gpu::getWarpSize(mfmaLayout);
    // compute number of output elements that each thread holds for one MFMA
    // instruction.
    const int subBlocks =
        getNumSubmatrices(aTensorTy.getElementType(), mDim, nDim);
    auto elemsPerVec = mDim * nDim * subBlocks / warpSize;

    Value firstMfma;
    auto setFirstMfma = [&](Value mfma) {
      if (!firstMfma)
        firstMfma = mfma;
    };

    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int b = 0; b < numRepB; ++b) {
      for (int m = 0; m < numRepM; ++m) {
        for (int n = 0; n < numRepN; ++n) {
          Value acc = tb.undef(vecTy);
          for (unsigned v = 0; v < elemsPerVec; ++v) {
            acc = tb.insert_element(
                vecTy, acc,
                fc[b * numRepM * numRepN * elemsPerVec +
                   m * numRepN * elemsPerVec + n * elemsPerVec + v],
                tb.i32_val(v));
          }
          acc = zeroAuxiliarBlocks(subBlocks, acc);
          for (int k = 0; k < numRepK; k++) {
            for (int kPack = 0; kPack < kWidth / kBase; ++kPack) {
              acc =
                  mfmaLayout.getIsTransposed()
                      ? generateMFMAOp(mfmaInsnName, operandB[kPack][{b, n, k}],
                                       operandA[kPack][{b, m, k}], acc)
                      : generateMFMAOp(mfmaInsnName, operandA[kPack][{b, m, k}],
                                       operandB[kPack][{b, n, k}], acc);
              setFirstMfma(acc);
            }
          }
          acc = reduceSubBlocks(subBlocks, acc);
          for (unsigned v = 0; v < elemsPerVec; ++v) {
            Value accElem = tb.extract_element(dstElemTy, acc, tb.i32_val(v));
            // Dot operand layout minimal tile is kDimInstrSize elements across
            // K dimension. If dot operand K dimension is smaller, layout
            // assigns tensor elements to multiple different hardware locations.
            // In this case mfma instruction adds elements in accumulator
            // multiple times.
            //
            // Let say A=[1,2]; B=[3,4], C = A*B = 1*3+2*4 = 11
            // Consider instruction K size is 4,
            // in this case operands will be duplicated:
            // A' = [1,2,1,2] B' = [3,4,3,4]
            // C' = (1*3+2*4) + (1*3+2*4) = 22
            //
            // Following code adjusts accumulator values in such cases.
            // If accumulator is integer, shift accumulator right by
            // log2(duplicationRate). If accumulator is float, multiply accum
            // with 1/duplicationRate constant.
            if (kDimInstrSize > kDimOperandSize) {
              assert(kDimInstrSize % kDimOperandSize == 0);
              int duplicationRate = kDimInstrSize / kDimOperandSize;
              assert(llvm::isPowerOf2_32(duplicationRate));
              if (dstElemTy.isInteger()) {
                auto shiftSize = llvm::Log2_32(duplicationRate);
                assert(!accElem.getType().isUnsignedInteger() &&
                       "MFMA uses signed accumulator");
                accElem = tb.ashr(accElem, tb.i32_val(shiftSize));
              } else {
                auto multiplierAttr =
                    rewriter.getFloatAttr(dstElemTy, 1.0 / duplicationRate);
                auto multiplierVal = rewriter.create<LLVM::ConstantOp>(
                    loc, dstElemTy, multiplierAttr);
                accElem = tb.fmul(accElem, multiplierVal);
              }
            }
            auto linearIdx = b * numRepM * numRepN * elemsPerVec +
                             m * numRepN * elemsPerVec + n * elemsPerVec + v;
            fc[linearIdx] = accElem;
          }
        }
      }
    }

    // Originally, setprio (high) is set to the high-level dot op. After dot is
    // being lowered to the series of mfma operations, it should be moved next
    // to the first mfma leaving the first mfma staying at the low priority. In
    // this way, incoming warp can be effectively waiting on the first mfma
    // instruction (low priority) while the other warp is executing mfma with
    // high priority. Otherwise, incoming warp can break the cluster.
    if (setPrioOp && firstMfma)
      setPrioOp->moveAfter(firstMfma.getDefiningOp());

    // replace with new packed result
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), dstElemTy));
    Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

    Type elemtTy = elemTyA;
    const size_t mmaCount =
        numRepB * numRepM * numRepN * numRepK * kWidth / kBase;
    setNumGeneratedMMAs(op, mmaCount, maybeMfmaInsn->getMDim(),
                        maybeMfmaInsn->getNDim(), maybeMfmaInsn->getKDim(),
                        elemtTy);

    rewriter.replaceOp(op, res);

    return success();
  }

  /// Extract vector from rawElems based on kWidth and kBase
  /// rawElems is a vector of kWidth elements. We need to prepare vector(s) of
  /// kBase elements for each mfma instruction
  virtual SmallVector<Value> extractOperands(Value rawElems, int kWidth,
                                             int kBase, Type type) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int kpack = kWidth / kBase;
    SmallVector<Value> results;
    auto vecTy = vec_ty(type, kBase);
    if (type.isBF16())
      vecTy = vec_ty(i16_ty, kBase);
    for (int k = 0; k < kpack; ++k) {
      Value vec = b.undef(vecTy);
      for (int elemId = 0; elemId < kBase; ++elemId) {
        auto val =
            b.extract_element(type, rawElems, b.i32_val(elemId + k * kBase));
        if (type.isBF16()) {
          // rocdl.mfma.f32.32x32x8bf16.1k calls for input of i16 type
          auto cast = b.bitcast(val, i16_ty);
          vec = b.insert_element(vecTy, vec, cast, b.i32_val(elemId));
        } else {
          vec = b.insert_element(vecTy, vec, val, b.i32_val(elemId));
        }
      }
      if (type.getIntOrFloatBitWidth() == 8) {
        if (4 == kBase)
          // This is for int8 on pre- MI300 GPUs
          results.push_back(b.bitcast(vec, i32_ty));
        if (8 == kBase)
          results.push_back(b.bitcast(vec, i64_ty));
      } else {
        results.push_back(vec);
      }
    }
    return results;
  }

  /// Converts dot operand structure to value table and converts types
  /// appropriate for mfma instructions
  virtual SmallVector<ValueTable>
  getValuesFromDotOperandLayoutStruct(Value value, int batch, int n0, int n1,
                                      int kWidth, int kBase, Type type,
                                      bool allowXF32) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto elems = unpackLLElements(loc, value, rewriter);
    llvm::outs() << "elems.size(): " << elems.size() << " kWidth: " << kWidth
                 << " kBase: " << kBase << " batch: " << batch << " n0: " << n0
                 << " n1: " << n1 << "\n";
    int kpack = kWidth / kBase;
    SmallVector<ValueTable> dotOpVals(kpack);
    for (int b = 0; b < batch; ++b) {
      for (int i = 0; i < n0; i++) {
        for (int j = 0; j < n1; j++) {
          Type elemTy = typeConverter->convertType(type);
          Type ty = vec_ty(elemTy, kWidth);
          Value rawElems = tb.undef(ty);
          for (int k = 0; k < kWidth; ++k) {
            auto idx = kWidth * n1 * n0 * b + kWidth * n1 * i + kWidth * j + k;
            llvm::outs() << "idx: " << idx << "\n";
            rawElems = tb.insert_element(
                ty, rawElems,
                elems[kWidth * n1 * n0 * b + kWidth * n1 * i + kWidth * j + k],
                tb.i32_val(k));
          }

          Value convertedElems;
          if (type.isF32() && !allowXF32) {
            for (int k = 0; k < kpack; ++k)
              dotOpVals[k][{b, i, j}] =
                  tb.extract_element(type, rawElems, tb.i32_val(k));
          } else {
            SmallVector<Value> vals;
            if (type.isF32() && allowXF32) {
              vals = extractOperands(rawElems, kWidth, kBase, f32_ty);
            } else if (type.getIntOrFloatBitWidth() == 8) {
              vals = extractOperands(rawElems, kWidth, kBase, i8_ty);
            } else if (type.isBF16()) {
              vals = extractOperands(rawElems, kWidth, kBase, bf16_ty);
            } else {
              assert(type.isF16() && "Unsupported data type");
              vals = extractOperands(rawElems, kWidth, kBase, f16_ty);
            }
            for (int k = 0; k < kpack; ++k) {
              dotOpVals[k][{b, i, j}] = vals[k];
            }
          }
        }
      }
    }
    return dotOpVals;
  }
};

struct ScaledDotOpMFMAConversionHelper : DotOpMFMAConversionHelper {

  ScaledDotOpMFMAConversionHelper(AMDMfmaEncodingAttr mfmaLayout,
                                  ConversionPatternRewriter &rewriter,
                                  const LLVMTypeConverter *typeConverter,
                                  Location loc)
      : DotOpMFMAConversionHelper(mfmaLayout, rewriter, typeConverter, loc) {}

  Value generateScaledMFMAOp(MfmaInsn &mfmaInsn, Value valA, Value valB,
                             Value valC, Value valScaleA,
                             Value valScaleB) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resType = valC.getType();
    Value zeroFlag = b.i32_val(0);
    OperationState loweredOp(loc, mfmaInsn.getInsnName());
    int32_t cbsz = getMfmaF8F6F4MatrixFormat(mfmaInsn.getElementTypeA());
    int32_t blgp = getMfmaF8F6F4MatrixFormat(mfmaInsn.getElementTypeB());
    assert((cbsz != -1) && (blgp != -1));
    llvm::outs() << "ScaledDotOpMFMAConversionHelper::generateScaledMFMAOp"
                 << " cbsz: " << cbsz << " blgp: " << blgp << "\n";
    loweredOp.addTypes(resType);
    loweredOp.addOperands({valA, valB, valC, b.i32_val(cbsz), b.i32_val(blgp),
                           zeroFlag, valScaleA, zeroFlag, valScaleB});
    return rewriter.create(loweredOp)->getResult(0);
  }

  LogicalResult convertScaledDot(DotScaledOp op,
                                 DotScaledOpAdaptor adaptor) const {
    llvm::outs()
        << op.getLoc()
        << " ScaledDotOpMFMAConversionHelper::convertScaledDot start\n";
    // Check if this dot has come with priority set by setprio.
    auto setPrioOp = dyn_cast_or_null<ROCDL::SetPrioOp>(op->getPrevNode());

    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();
    auto mfmaVersion = mfmaLayout.getVersionMajor();
    assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
           (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));

    Value a = op.getLhs();
    Value b = op.getRhs();
    Value aScale = op.getLhsScale();
    Value bScale = op.getRhsScale();
    Value d = op.getD();
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());
    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();
    ScaleDotElemType aElemType = op.getLhsType();
    ScaleDotElemType bElemType = op.getRhsType();

    auto supportsTypes = [](ScaleDotElemType elemType) {
      return elemType == ScaleDotElemType::E2M1;
    };

    if (!supportsTypes(aElemType) || !supportsTypes(bElemType)) {
      llvm::report_fatal_error("NYI: mxfp6, mxfp8\n");
    }

    constexpr bool allowXF32 = false;
    auto ctx = op.getContext();
    auto maybeMfmaInsn = MfmaInsn::selectMfma(
        mDim, nDim, scaleDotElemTypeToMLIRType(ctx, aElemType),
        scaleDotElemTypeToMLIRType(ctx, bElemType), mfmaVersion, allowXF32);
    if (failed(maybeMfmaInsn))
      llvm::report_fatal_error("No match found in MFMA database\n");

    StringRef mfmaInsnName = maybeMfmaInsn->getInsnName();
    unsigned kBase = maybeMfmaInsn->getKBase();
    if (aElemType == ScaleDotElemType::E2M1) {
      kBase /= 2;
    }

    llvm::outs() << "ScaledDotOpMFMAConversionHelper::convertScaledDot "
                    "Selected inst name: "
                 << mfmaInsnName << "\n";

    auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
    int kWidth = aEncoding.getKWidth();
    auto rank = aTensorTy.getShape().size();
    const auto kDimOperandSize = aTensorTy.getShape()[rank - 1];
    const auto kDimInstrSize = mfmaLayout.getInstrShapeForOperand(kWidth, 0)[1];

    auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), kWidth, 0);
    auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), kWidth, 1);
    assert(repA[2] == repB[1]);

    auto aScaleTensorTy = cast<RankedTensorType>(aScale.getType());
    auto bScaleTensorTy = cast<RankedTensorType>(bScale.getType());

    int aScaleKWidth = 1;
    int bScaleKWidth = 1;
    // auto repAScale =
    //     mfmaLayout.getRepForOperand(aScaleTensorTy.getShape(), aScaleKWidth,
    //     0);
    auto repAScale = repA;
    auto bScaleShape = bScaleTensorTy.getShape();
    // auto repBScale = mfmaLayout.getRepForOperand(bScaleShape, bScaleKWidth,
    // 0);
    auto repBScale = repB;

    auto aScaleShape = aScaleTensorTy.getShape();
    auto aScaleInstrShape = mfmaLayout.getInstrShapeForOperand(aScaleKWidth, 0);
    auto bScaleInstrShape = mfmaLayout.getInstrShapeForOperand(bScaleKWidth, 0);
    llvm::outs() << op.getLoc()
                 << " ScaledDotOpMFMAConversionHelper::convertScaledDot"
                 << " aScaleKWidth: " << aScaleKWidth << "\n"
                 << " bScaleKWidth: " << bScaleKWidth << "\n"
                 << " aScaleInstrShape[0]: " << aScaleInstrShape[0]
                 << " aScaleInstrShape[1]: " << aScaleInstrShape[1] << "\n"
                 << " bScaleInstrShape[0]: " << bScaleInstrShape[0]
                 << " bScaleInstrShape[1]: " << bScaleInstrShape[1] << "\n"
                 << " warpsPerCTA[0]: " << warpsPerCTA[0]
                 << " warpsPerCTA[1]: " << warpsPerCTA[1] << "\n"
                 << " aScaleShape[0]: " << aScaleShape[0]
                 << " aScaleShape[1]: " << aScaleShape[1] << "\n"
                 << " bScaleShape[0]: " << bScaleShape[0]
                 << " bScaleShape[1]: " << bScaleShape[1] << "\n"
                 << " repA[0]: " << repA[0] << " repA[1]: " << repA[1]
                 << " repA[2]: " << repA[2] << "\n"
                 << " repB[0]: " << repB[0] << " repB[1]: " << repB[1]
                 << " repB[2]: " << repB[2] << "\n"
                 << " repAScale[0]: " << repAScale[0]
                 << " repAScale[1]: " << repAScale[1]
                 << " repAScale[2]: " << repAScale[2] << "\n"
                 << " repBScale[0]: " << repBScale[0]
                 << " repBScale[1]: " << repBScale[1]
                 << " repBScale[2]: " << repBScale[2] << "\n";

    // assert(repAScale[2] == repBScale[1]);

    Value loadedA = adaptor.getLhs();
    Value loadedB = adaptor.getRhs();
    Value loadedAScale = adaptor.getLhsScale();
    Value loadedBScale = adaptor.getRhsScale();
    Value loadedC = adaptor.getC();

    auto workIDX = rewriter.create<ROCDL::ThreadIdXOp>(loc, i32_ty);
    auto workIDY = rewriter.create<ROCDL::ThreadIdYOp>(loc, i32_ty);
    auto workIDZ = rewriter.create<ROCDL::ThreadIdZOp>(loc, i32_ty);

    // auto printElems = [&](const char *name, Value v) {
    //   mlir::triton::AMD::TargetInfo targetInfo("gfx950");
    //   auto elems = unpackLLElements(loc, v, rewriter);
    //   std::stringstream ss;
    //   ss << name << "(" << elems.size() << ")(tidx: %d, tidy: %d, tidz: %d):
    //   "; if (elems.size() >= 5) {
    //     ss << "(%d, %d, %d, ..., %d, %d)";
    //     auto size = elems.size();
    //     targetInfo.printf(rewriter, ss.str(),
    //                       {workIDX, workIDY, workIDZ, elems[0], elems[1],
    //                        elems[2], elems[size - 2], elems[size - 1]});
    //   } else if (elems.size() == 2) {
    //     ss << "(%d, %d)";
    //     targetInfo.printf(rewriter, ss.str(),
    //                       {workIDX, workIDY, workIDZ, elems[0], elems[1]});
    //   } else if (elems.size() == 1) {
    //     ss << "(%d)";
    //     targetInfo.printf(rewriter, ss.str(),
    //                       {workIDX, workIDY, workIDZ, elems[0]});
    //   } else if (elems.size() == 0) {
    //     ss << "empty";
    //     targetInfo.printf(rewriter, ss.str(), {workIDX, workIDY, workIDZ});
    //   }
    // };

    // printElems("loadedA", loadedA);
    // printElems("loadedAScale", loadedAScale);
    // printElems("loadedB", loadedB);
    // printElems("loadedBScale", loadedBScale);

    assert(repAScale[2] == repBScale[1]);

    auto numRepM = repA[1];
    auto numRepN = repB[2];
    auto numRepK = repA[2];
    auto numRepB = repA[0];
    assert(repA[0] == repB[0]);

    auto operandA = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepB, numRepM, numRepK, kWidth, kBase,
        aTensorTy.getElementType(), allowXF32);
    auto operandB = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepB, numRepN, numRepK, kWidth, kBase,
        bTensorTy.getElementType(), allowXF32);

    auto numRepScaleM = repAScale[1];
    auto numRepScaleN = repBScale[2];
    auto numRepScaleK = repAScale[2];
    auto numRepScaleB = repAScale[0];
    assert(repAScale[0] == repBScale[0]);

    constexpr int kBaseScale = 1;
    auto operandAScale = getValuesFromDotOperandLayoutStruct(
        loadedAScale, numRepScaleB, numRepScaleM, numRepScaleK, aScaleKWidth,
        kBaseScale, aScaleTensorTy.getElementType(), allowXF32);
    auto operandBScale = getValuesFromDotOperandLayoutStruct(
        loadedBScale, numRepScaleB, numRepScaleN, numRepScaleK, bScaleKWidth,
        kBaseScale, bScaleTensorTy.getElementType(), allowXF32);

    llvm::outs() << " operandA.size(): " << operandA.size() << "\n"
                 << " operandB.size(): " << operandB.size() << "\n"
                 << " operandAScale.size(): " << operandAScale.size() << "\n"
                 << " operandBScale.size(): " << operandBScale.size() << "\n"
                 << " loadedAScale.shape(): " << loadedAScale.getType() << "\n"
                 << " loadedBScale.shape(): " << loadedBScale.getType() << "\n";

    auto printOperand = [](std::string n, SmallVector<ValueTable> &v) {
      llvm::outs() << n << ": ";
      for (size_t i = 0; i < v.size(); i++) {
        llvm::outs() << "(" << i << ", " << v[i].size() << ")";
      }
      llvm::outs() << "\n";
    };
    printOperand("operandA", operandA);
    printOperand("operandB", operandB);
    printOperand("operandAScale", operandAScale);
    printOperand("operandBScale", operandBScale);

    auto dstElemTy = dTensorTy.getElementType();
    auto fc = unpackLLElements(loc, loadedC, rewriter);

    unsigned warpSize = triton::gpu::getWarpSize(mfmaLayout);
    // compute number of output elements that each thread holds for one MFMA
    // instruction. subBlocks
    const int subBlocks =
        getNumSubmatrices(aTensorTy.getElementType(), mDim, nDim);
    auto elemsPerVec = mDim * nDim * subBlocks / warpSize;

    Value firstMfma;
    auto setFirstMfma = [&](Value mfma) {
      if (!firstMfma)
        firstMfma = mfma;
    };

    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int b = 0; b < numRepB; ++b) {
      for (int m = 0; m < numRepM; ++m) {
        for (int n = 0; n < numRepN; ++n) {
          Value acc = tb.undef(vecTy);
          for (unsigned v = 0; v < elemsPerVec; ++v) {
            acc = tb.insert_element(
                vecTy, acc,
                fc[b * numRepM * numRepN * elemsPerVec +
                   m * numRepN * elemsPerVec + n * elemsPerVec + v],
                tb.i32_val(v));
          }
          acc = zeroAuxiliarBlocks(subBlocks, acc);
          for (int k = 0; k < numRepK; k++) {
            for (int kPack = 0; kPack < kWidth / kBase; ++kPack) {
              acc = mfmaLayout.getIsTransposed()
                        ? generateScaledMFMAOp(maybeMfmaInsn.value(),
                                               operandB[kPack][{b, n, k}],
                                               operandA[kPack][{b, m, k}], acc,
                                               operandBScale[kPack][{b, n, k}],
                                               operandAScale[kPack][{b, m, k}])
                        : generateScaledMFMAOp(maybeMfmaInsn.value(),
                                               operandA[kPack][{b, m, k}],
                                               operandB[kPack][{b, n, k}], acc,
                                               operandAScale[kPack][{b, m, k}],
                                               operandBScale[kPack][{b, n, k}]);
              setFirstMfma(acc);
            }
          }
          acc = reduceSubBlocks(subBlocks, acc);
          for (unsigned v = 0; v < elemsPerVec; ++v) {
            Value accElem = tb.extract_element(dstElemTy, acc, tb.i32_val(v));
            // Dot operand layout minimal tile is kDimInstrSize elements across
            // K dimension. If dot operand K dimension is smaller, layout
            // assigns tensor elements to multiple different hardware locations.
            // In this case mfma instruction adds elements in accumulator
            // multiple times.
            //
            // Let say A=[1,2]; B=[3,4], C = A*B = 1*3+2*4 = 11
            // Consider instruction K size is 4,
            // in this case operands will be duplicated:
            // A' = [1,2,1,2] B' = [3,4,3,4]
            // C' = (1*3+2*4) + (1*3+2*4) = 22
            //
            // Following code adjusts accumulator values in such cases.
            // If accumulator is integer, shift accumulator right by
            // log2(duplicationRate). If accumulator is float, multiply accum
            // with 1/duplicationRate constant.
            if (kDimInstrSize > kDimOperandSize) {
              assert(kDimInstrSize % kDimOperandSize == 0);
              int duplicationRate = kDimInstrSize / kDimOperandSize;
              assert(llvm::isPowerOf2_32(duplicationRate));
              if (dstElemTy.isInteger()) {
                auto shiftSize = llvm::Log2_32(duplicationRate);
                assert(!accElem.getType().isUnsignedInteger() &&
                       "MFMA uses signed accumulator");
                accElem = tb.ashr(accElem, tb.i32_val(shiftSize));
              } else {
                auto multiplierAttr =
                    rewriter.getFloatAttr(dstElemTy, 1.0 / duplicationRate);
                auto multiplierVal = rewriter.create<LLVM::ConstantOp>(
                    loc, dstElemTy, multiplierAttr);
                accElem = tb.fmul(accElem, multiplierVal);
              }
            }
            auto linearIdx = b * numRepM * numRepN * elemsPerVec +
                             m * numRepN * elemsPerVec + n * elemsPerVec + v;
            fc[linearIdx] = accElem;
          }
        }
      }
    }

    // Originally, setprio (high) is set to the high-level dot op. After dot is
    // being lowered to the series of mfma operations, it should be moved next
    // to the first mfma leaving the first mfma staying at the low priority. In
    // this way, incoming warp can be effectively waiting on the first mfma
    // instruction (low priority) while the other warp is executing mfma with
    // high priority. Otherwise, incoming warp can break the cluster.
    if (setPrioOp && firstMfma)
      setPrioOp->moveAfter(firstMfma.getDefiningOp());

    // replace with new packed result
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), dstElemTy));
    Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

    Type elemtTy = elemTyA;
    const size_t mmaCount =
        numRepB * numRepM * numRepN * numRepK * kWidth / kBase;
    setNumGeneratedMMAs(op, mmaCount, maybeMfmaInsn->getMDim(),
                        maybeMfmaInsn->getNDim(), maybeMfmaInsn->getKDim(),
                        elemtTy);

    rewriter.replaceOp(op, res);

    return success();
  }

  /// Extract vector from rawElems based on kWidth and kBase
  /// rawElems is a vector of kWidth elements. We need to prepare vector(s) of
  /// kBase elements for each mfma instruction
  SmallVector<Value> extractOperands(Value rawElems, int kWidth, int kBase,
                                     Type type) const {
    llvm::outs() << "ScaledDotOpMFMAConversionHelper::extractOperands start\n";
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    int kpack = kWidth / kBase;
    SmallVector<Value> results;
    auto vecTy = vec_ty(type, kBase);
    if (type.isBF16())
      vecTy = vec_ty(i16_ty, kBase);
    for (int k = 0; k < kpack; ++k) {
      Value vec = tb.undef(vecTy);
      for (int elemId = 0; elemId < kBase; ++elemId) {
        auto val =
            tb.extract_element(type, rawElems, tb.i32_val(elemId + k * kBase));
        if (type.isBF16()) {
          // rocdl.mfma.f32.32x32x8bf16.1k calls for input of i16 type
          auto cast = tb.bitcast(val, i16_ty);
          vec = tb.insert_element(vecTy, vec, cast, tb.i32_val(elemId));
        } else {
          vec = tb.insert_element(vecTy, vec, val, tb.i32_val(elemId));
        }
      }
      if (type.getIntOrFloatBitWidth() == 8) {
        if (1 == kBase) {
          llvm::outs()
              << "ScaledDotOpMFMAConversionHelper::extractOperands kBase==1\n";
          results.push_back(tb.zext(i32_ty, tb.bitcast(vec, i8_ty)));
        }
        if (4 == kBase)
          // This is for int8 on pre- MI300 GPUs
          results.push_back(tb.bitcast(vec, i32_ty));
        if (8 == kBase)
          results.push_back(tb.bitcast(vec, i64_ty));
        if (16 == kBase) {
          llvm::outs()
              << "ScaledDotOpMFMAConversionHelper::extractOperands kBase==16\n";
          results.push_back(tb.bitcast(vec, vec_ty(i32_ty, 4)));
        }
      } else {
        results.push_back(vec);
      }
    }
    return results;
  }
};

} // namespace

namespace mlir::triton::AMD {
LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return cast<RankedTensorType>(tensor.getType());
  };

  assert(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
         isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()) &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support $c with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  DotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter, loc);

  return helper.convertDot(op, adaptor);
}

LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return cast<RankedTensorType>(tensor.getType());
  };

  assert(isa<DotOperandEncodingAttr>(rankedTType(op.getLhs()).getEncoding()) &&
         isa<DotOperandEncodingAttr>(rankedTType(op.getRhs()).getEncoding()) &&
         "Both $lhs and $rhs should be DotOperand layout.");

  assert(isa<LinearEncodingAttr>(rankedTType(op.getLhsScale()).getEncoding()) &&
         isa<LinearEncodingAttr>(rankedTType(op.getRhsScale()).getEncoding()) &&
         "Both $lhs_scale and $rhs_scale should be linear layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support $c with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  ScaledDotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter,
                                         loc);

  return helper.convertScaledDot(op, adaptor);
}
} // namespace mlir::triton::AMD
