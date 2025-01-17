#include "TritonAMDGPUTransforms/MfmaGroup.h"

namespace mlir {

static MfmaTypeId chooseAppropriateMfmaId(mlir::Type dataTypeA,
                                          mlir::Type dataTypeB,
                                          bool allowXF32) {
  if (dataTypeA.isF32() && dataTypeB.isF32()) {
    if (allowXF32)
      return MfmaTypeId::Xf32TyId;
    else
      return MfmaTypeId::Fp32TyId;
  }
  if (dataTypeA.isF16() && dataTypeB.isF16()) {
    return MfmaTypeId::Fp16TyId;
  }
  if (dataTypeA.isBF16() && dataTypeB.isBF16()) {
    return MfmaTypeId::Bf16TyId;
  }
  if (dataTypeA.isInteger(8) && dataTypeB.isInteger(8)) {
    return MfmaTypeId::I8TyId;
  }
  if (llvm::isa<Float8E4M3FNUZType>(dataTypeA) &&
      llvm::isa<Float8E4M3FNUZType>(dataTypeB)) {
    return MfmaTypeId::Fp8Fp8TyId;
  }
  if (llvm::isa<Float8E4M3FNUZType>(dataTypeA) &&
      llvm::isa<Float8E5M2FNUZType>(dataTypeB)) {
    return MfmaTypeId::Fp8Bf8TyId;
  }
  if (llvm::isa<Float8E5M2FNUZType>(dataTypeA) &&
      llvm::isa<Float8E4M3FNUZType>(dataTypeB)) {
    return MfmaTypeId::Bf8Fp8TyId;
  }
  if (llvm::isa<Float8E5M2FNUZType>(dataTypeA) &&
      llvm::isa<Float8E5M2FNUZType>(dataTypeB)) {
    return MfmaTypeId::Bf8Bf8TyId;
  }
  if (llvm::isa<Float8E5M2Type>(dataTypeA) &&
      llvm::isa<Float8E5M2Type>(dataTypeB)) {
    return MfmaTypeId::Fp16TyId;
  }
  // fp8/bf8 * fp6/bf6
  if (dataTypeA.isFloat8E4M3FNUZ() && dataTypeB.isFloat6E3M2FN()) {
    return MfmaTypeId::Fp8Fp6TyId;
  }
  if (dataTypeA.isFloat8E4M3FNUZ() && dataTypeB.isFloat6E2M3FN()) {
    return MfmaTypeId::Fp8Bf6TyId;
  }
  if (dataTypeA.isFloat8E5M2FNUZ() && dataTypeB.isFloat6E3M2FN()) {
    return MfmaTypeId::Bf8Fp6TyId;
  }
  if (dataTypeA.isFloat8E5M2FNUZ() && dataTypeB.isFloat6E2M3FN()) {
    return MfmaTypeId::Bf8Bf6TyId;
  }
  // fp8/bf8 * fp4
  if (dataTypeA.isFloat8E4M3FNUZ() && dataTypeB.isFloat4E2M1FN()) {
    return MfmaTypeId::Fp8Fp4TyId;
  }
  if (dataTypeA.isFloat8E5M2FNUZ() && dataTypeB.isFloat4E2M1FN()) {
    return MfmaTypeId::Bf8Fp4TyId;
  }
  // fp6 * bf6
  if (dataTypeA.isFloat6E3M2FN() && dataTypeB.isFloat6E3M2FN()) {
    return MfmaTypeId::Fp6Fp6TyId;
  }
  if (dataTypeA.isFloat6E3M2FN() && dataTypeB.isFloat6E2M3FN()) {
    return MfmaTypeId::Fp6Bf6TyId;
  }
  if (dataTypeA.isFloat6E2M3FN() && dataTypeB.isFloat6E3M2FN()) {
    return MfmaTypeId::Bf6Fp6TyId;
  }
  if (dataTypeA.isFloat6E2M3FN() && dataTypeB.isFloat6E2M3FN()) {
    return MfmaTypeId::Bf6Bf6TyId;
  }
  // fp6/bf6 * fp4
  if (dataTypeA.isFloat6E3M2FN() && dataTypeB.isFloat4E2M1FN()) {
    return MfmaTypeId::Fp6Fp4TyId;
  }
  if (dataTypeA.isFloat6E2M3FN() && dataTypeB.isFloat4E2M1FN()) {
    return MfmaTypeId::Bf6Fp4TyId;
  }
  // fp4
  if (dataTypeA.isFloat4E2M1FN() && dataTypeB.isFloat4E2M1FN()) {
    return MfmaTypeId::Fp4Fp4TyId;
  }
  llvm_unreachable("Unsupported input argument type.");
}

using MfmaInsnGroupMap = llvm::DenseMap<MfmaInsnGroupSelectKey, MfmaInsnAttr,
                                        MfmaInsnGroupSelectKeyInfo>;

auto getMfmaInsnGroupAttrMap = []() -> const MfmaInsnGroupMap & {
  static MfmaInsnGroupMap MfmaInsnMap{
      // xf32
      // mfma.xf32.16x16x8xf32
      {{16, 16, MfmaTypeId::Xf32TyId, 3},
       {16, 16, 8, 2, ROCDL::mfma_f32_16x16x8_xf32::getOperationName()}},
      // mfma.xf32.32x32x4.xf32
      {{32, 32, MfmaTypeId::Xf32TyId, 3},
       {32, 32, 4, 2, ROCDL::mfma_f32_32x32x4_xf32::getOperationName()}},
      // f32
      // mfma_f32_32x32x2f32
      {{32, 32, MfmaTypeId::Fp32TyId, 1},
       {32, 32, 2, 1, ROCDL::mfma_f32_32x32x2f32::getOperationName()}},
      {{32, 32, MfmaTypeId::Fp32TyId, 2},
       {32, 32, 2, 1, ROCDL::mfma_f32_32x32x2f32::getOperationName()}},
      {{32, 32, MfmaTypeId::Fp32TyId, 3},
       {32, 32, 2, 1, ROCDL::mfma_f32_32x32x2f32::getOperationName()}},
      // mfma_f32_16x16x4f32
      {{16, 16, MfmaTypeId::Fp32TyId, 1},
       {16, 16, 4, 1, ROCDL::mfma_f32_16x16x4f32::getOperationName()}},
      {{16, 16, MfmaTypeId::Fp32TyId, 2},
       {16, 16, 4, 1, ROCDL::mfma_f32_16x16x4f32::getOperationName()}},
      {{16, 16, MfmaTypeId::Fp32TyId, 3},
       {16, 16, 4, 1, ROCDL::mfma_f32_16x16x4f32::getOperationName()}},
      // mfma_f32_4x4x1f32
      {{4, 4, MfmaTypeId::Fp32TyId, 1},
       {4, 4, 16, 1, ROCDL::mfma_f32_4x4x1f32::getOperationName()}},
      {{4, 4, MfmaTypeId::Fp32TyId, 2},
       {4, 4, 16, 1, ROCDL::mfma_f32_4x4x1f32::getOperationName()}},
      {{4, 64, MfmaTypeId::Fp32TyId, 1},
       {4, 64, 1, 1, ROCDL::mfma_f32_4x4x1f32::getOperationName()}},
      {{4, 64, MfmaTypeId::Fp32TyId, 2},
       {4, 64, 1, 1, ROCDL::mfma_f32_4x4x1f32::getOperationName()}},
      {{64, 4, MfmaTypeId::Fp32TyId, 1},
       {64, 4, 1, 1, ROCDL::mfma_f32_4x4x1f32::getOperationName()}},
      {{64, 4, MfmaTypeId::Fp32TyId, 2},
       {64, 4, 1, 1, ROCDL::mfma_f32_4x4x1f32::getOperationName()}},
      // mfma_f32_4x4x1_16B_f32
      {{4, 4, MfmaTypeId::Fp32TyId, 3},
       {4, 4, 16, 1, ROCDL::mfma_f32_4x4x1f32::getOperationName()}},
      {{4, 64, MfmaTypeId::Fp32TyId, 3},
       {4, 64, 1, 1, ROCDL::mfma_f32_4x4x1f32::getOperationName()}},
      {{64, 4, MfmaTypeId::Fp32TyId, 3},
       {64, 4, 1, 1, ROCDL::mfma_f32_4x4x1f32::getOperationName()}},
      // f16
      // mfma_f32_32x32x8f16
      {{32, 32, MfmaTypeId::Fp16TyId, 1},
       {32, 32, 8, 4, ROCDL::mfma_f32_32x32x8f16::getOperationName()}},
      {{32, 32, MfmaTypeId::Fp16TyId, 2},
       {32, 32, 8, 4, ROCDL::mfma_f32_32x32x8f16::getOperationName()}},
      {{32, 32, MfmaTypeId::Fp16TyId, 3},
       {32, 32, 8, 4, ROCDL::mfma_f32_32x32x8f16::getOperationName()}},
      // mfma_f32_16x16x16xf16
      {{16, 16, MfmaTypeId::Fp16TyId, 1},
       {16, 16, 16, 4, ROCDL::mfma_f32_16x16x16f16::getOperationName()}},
      {{16, 16, MfmaTypeId::Fp16TyId, 2},
       {16, 16, 16, 4, ROCDL::mfma_f32_16x16x16f16::getOperationName()}},
      {{16, 16, MfmaTypeId::Fp16TyId, 3},
       {16, 16, 16, 4, ROCDL::mfma_f32_16x16x16f16::getOperationName()}},
      // mfma_f32_4x4x4f16
      {{4, 4, MfmaTypeId::Fp16TyId, 1},
       {4, 4, 64, 4, ROCDL::mfma_f32_4x4x4f16::getOperationName()}},
      {{4, 4, MfmaTypeId::Fp16TyId, 2},
       {4, 4, 64, 4, ROCDL::mfma_f32_4x4x4f16::getOperationName()}},
      {{4, 4, MfmaTypeId::Fp16TyId, 3},
       {4, 4, 64, 4, ROCDL::mfma_f32_4x4x4f16::getOperationName()}},
      {{4, 64, MfmaTypeId::Fp16TyId, 1},
       {4, 64, 4, 4, ROCDL::mfma_f32_4x4x4f16::getOperationName()}},
      {{4, 64, MfmaTypeId::Fp16TyId, 2},
       {4, 64, 4, 4, ROCDL::mfma_f32_4x4x4f16::getOperationName()}},
      {{4, 64, MfmaTypeId::Fp16TyId, 3},
       {4, 64, 4, 4, ROCDL::mfma_f32_4x4x4f16::getOperationName()}},
      {{64, 4, MfmaTypeId::Fp16TyId, 1},
       {64, 4, 4, 4, ROCDL::mfma_f32_4x4x4f16::getOperationName()}},
      {{64, 4, MfmaTypeId::Fp16TyId, 2},
       {64, 4, 4, 4, ROCDL::mfma_f32_4x4x4f16::getOperationName()}},
      {{64, 4, MfmaTypeId::Fp16TyId, 3},
       {64, 4, 4, 4, ROCDL::mfma_f32_4x4x4f16::getOperationName()}},
      // bf16
      // mfma_f32_32x32x4_bf16
      {{32, 32, MfmaTypeId::Bf16TyId, 1},
       {32, 32, 4, 2, ROCDL::mfma_f32_32x32x4bf16::getOperationName()}},
      // mfma_f32_32x32x8_bf16_1K
      {{32, 32, MfmaTypeId::Bf16TyId, 2},
       {32, 32, 8, 4, ROCDL::mfma_f32_32x32x8bf16_1k::getOperationName()}},
      {{32, 32, MfmaTypeId::Bf16TyId, 3},
       {32, 32, 8, 4, ROCDL::mfma_f32_32x32x8bf16_1k::getOperationName()}},
      // mfma_f32_16x16x8_bf16
      {{16, 16, MfmaTypeId::Bf16TyId, 1},
       {16, 16, 8, 2, ROCDL::mfma_f32_16x16x8bf16::getOperationName()}},
      // mfma_f32_16x16x16_bf16_1K
      {{16, 16, MfmaTypeId::Bf16TyId, 2},
       {16, 16, 16, 4, ROCDL::mfma_f32_16x16x16bf16_1k::getOperationName()}},
      {{16, 16, MfmaTypeId::Bf16TyId, 3},
       {16, 16, 16, 4, ROCDL::mfma_f32_16x16x16bf16_1k::getOperationName()}},
      // mfma_f32_4x4x2_bf16
      {{4, 4, MfmaTypeId::Bf16TyId, 1},
       {4, 4, 32, 2, ROCDL::mfma_f32_4x4x2bf16::getOperationName()}},
      {{4, 64, MfmaTypeId::Bf16TyId, 1},
       {4, 64, 2, 2, ROCDL::mfma_f32_4x4x2bf16::getOperationName()}},
      {{64, 4, MfmaTypeId::Bf16TyId, 1},
       {64, 4, 2, 2, ROCDL::mfma_f32_4x4x2bf16::getOperationName()}},
      // mfma_f32_4x4x4_bf16_1K
      {{4, 4, MfmaTypeId::Bf16TyId, 2},
       {4, 4, 64, 4, ROCDL::mfma_f32_4x4x4bf16_1k::getOperationName()}},
      {{4, 4, MfmaTypeId::Bf16TyId, 3},
       {4, 4, 64, 4, ROCDL::mfma_f32_4x4x4bf16_1k::getOperationName()}},
      {{4, 64, MfmaTypeId::Bf16TyId, 2},
       {4, 64, 4, 4, ROCDL::mfma_f32_4x4x4bf16_1k::getOperationName()}},
      {{4, 64, MfmaTypeId::Bf16TyId, 3},
       {4, 64, 4, 4, ROCDL::mfma_f32_4x4x4bf16_1k::getOperationName()}},
      {{64, 4, MfmaTypeId::Bf16TyId, 2},
       {64, 4, 4, 4, ROCDL::mfma_f32_4x4x4bf16_1k::getOperationName()}},
      {{64, 4, MfmaTypeId::Bf16TyId, 3},
       {64, 4, 4, 4, ROCDL::mfma_f32_4x4x4bf16_1k::getOperationName()}},
      // int8
      // mfma_i32_32x32x8i8
      {{32, 32, MfmaTypeId::I8TyId, 1},
       {32, 32, 8, 4, ROCDL::mfma_i32_32x32x8i8::getOperationName()}},
      {{32, 32, MfmaTypeId::I8TyId, 2},
       {32, 32, 8, 4, ROCDL::mfma_i32_32x32x8i8::getOperationName()}},
      // mfma_i32_32x32x16i8
      {{32, 32, MfmaTypeId::I8TyId, 3},
       {32, 32, 16, 8, ROCDL::mfma_i32_32x32x16_i8::getOperationName()}},
      // mfma_i32_16x16x16i8
      {{16, 16, MfmaTypeId::I8TyId, 1},
       {16, 16, 16, 4, ROCDL::mfma_i32_16x16x16i8::getOperationName()}},
      {{16, 16, MfmaTypeId::I8TyId, 2},
       {16, 16, 16, 4, ROCDL::mfma_i32_16x16x16i8::getOperationName()}},
      // mfma_i32_16x16x32i8
      {{16, 16, MfmaTypeId::I8TyId, 3},
       {16, 16, 32, 8, ROCDL::mfma_i32_16x16x32_i8::getOperationName()}},
      // mfma_i32_4x4x4i8
      {{4, 4, MfmaTypeId::I8TyId, 1},
       {4, 4, 64, 4, ROCDL::mfma_i32_4x4x4i8::getOperationName()}},
      {{4, 4, MfmaTypeId::I8TyId, 2},
       {4, 4, 64, 4, ROCDL::mfma_i32_4x4x4i8::getOperationName()}},
      {{4, 4, MfmaTypeId::I8TyId, 3},
       {4, 4, 64, 4, ROCDL::mfma_i32_4x4x4i8::getOperationName()}},
      {{4, 64, MfmaTypeId::I8TyId, 1},
       {4, 64, 4, 4, ROCDL::mfma_i32_4x4x4i8::getOperationName()}},
      {{4, 64, MfmaTypeId::I8TyId, 2},
       {4, 64, 4, 4, ROCDL::mfma_i32_4x4x4i8::getOperationName()}},
      {{4, 64, MfmaTypeId::I8TyId, 3},
       {4, 64, 4, 4, ROCDL::mfma_i32_4x4x4i8::getOperationName()}},
      {{64, 4, MfmaTypeId::I8TyId, 1},
       {64, 4, 4, 4, ROCDL::mfma_i32_4x4x4i8::getOperationName()}},
      {{64, 4, MfmaTypeId::I8TyId, 2},
       {64, 4, 4, 4, ROCDL::mfma_i32_4x4x4i8::getOperationName()}},
      {{64, 4, MfmaTypeId::I8TyId, 3},
       {64, 4, 4, 4, ROCDL::mfma_i32_4x4x4i8::getOperationName()}},
      // fp8 * bf8
      // mfma_f32_32x32x16_FP8_FP8
      {{32, 32, MfmaTypeId::Fp8Fp8TyId, 3},
       {32, 32, 16, 8, ROCDL::mfma_f32_32x32x16_fp8_fp8::getOperationName()}},
      // mfma_f32_16x16x32_FP8_FP8
      {{16, 16, MfmaTypeId::Fp8Fp8TyId, 3},
       {16, 16, 32, 8, ROCDL::mfma_f32_16x16x32_fp8_fp8::getOperationName()}},
      // mfma_f32_32x32x16_FP8_BF8
      {{32, 32, MfmaTypeId::Fp8Bf8TyId, 3},
       {32, 32, 16, 8, ROCDL::mfma_f32_32x32x16_fp8_bf8::getOperationName()}},
      // mfma_f32_16x16x32_FP8_BF8
      {{16, 16, MfmaTypeId::Fp8Bf8TyId, 3},
       {16, 16, 32, 8, ROCDL::mfma_f32_16x16x32_fp8_bf8::getOperationName()}},
      // mfma_f32_32x32x16_BF8_FP8
      {{32, 32, MfmaTypeId::Bf8Fp8TyId, 3},
       {32, 32, 16, 8, ROCDL::mfma_f32_32x32x16_bf8_fp8::getOperationName()}},
      // mfma_f32_16x16x32_BF8_FP8
      {{16, 16, MfmaTypeId::Bf8Fp8TyId, 3},
       {16, 16, 32, 8, ROCDL::mfma_f32_16x16x32_bf8_fp8::getOperationName()}},
      // mfma_f32_32x32x16_BF8_BF8
      {{32, 32, MfmaTypeId::Bf8Bf8TyId, 3},
       {32, 32, 16, 8, ROCDL::mfma_f32_32x32x16_bf8_bf8::getOperationName()}},
      // mfma_f32_16x16x32_BF8_BF8
      {{16, 16, MfmaTypeId::Bf8Bf8TyId, 3},
       {16, 16, 32, 8, ROCDL::mfma_f32_16x16x32_bf8_bf8::getOperationName()}},
      // fp8/bf8 * fp6/bf6
      // mfma_f32_16x16x128_f8f6f4
      {{16, 16, MfmaTypeId::Fp8Fp6TyId, 3},
       {16, 16, 128, 8, ROCDL::mfma_f32_16x16x128_f8f6f4::getOperationName()}},
      {{16, 16, MfmaTypeId::Fp8Bf6TyId, 3},
       {16, 16, 128, 8, ROCDL::mfma_f32_16x16x128_f8f6f4::getOperationName()}},
      {{16, 16, MfmaTypeId::Bf8Fp6TyId, 3},
       {16, 16, 128, 8, ROCDL::mfma_f32_16x16x128_f8f6f4::getOperationName()}},
      {{16, 16, MfmaTypeId::Bf8Bf6TyId, 3},
       {16, 16, 128, 8, ROCDL::mfma_f32_16x16x128_f8f6f4::getOperationName()}},
      // mfma_scale_f32_16x16x128_f8f6f4
      {{16, 16, MfmaTypeId::Fp8Fp6TyId, 3},
       {16, 16, 128, 8,
        ROCDL::mfma_scale_f32_16x16x128_f8f6f4::getOperationName()}},
      {{16, 16, MfmaTypeId::Fp8Bf6TyId, 3},
       {16, 16, 128, 8,
        ROCDL::mfma_scale_f32_16x16x128_f8f6f4::getOperationName()}},
      {{16, 16, MfmaTypeId::Bf8Fp6TyId, 3},
       {16, 16, 128, 8,
        ROCDL::mfma_scale_f32_16x16x128_f8f6f4::getOperationName()}},
      {{16, 16, MfmaTypeId::Bf8Bf6TyId, 3},
       {16, 16, 128, 8,
        ROCDL::mfma_scale_f32_16x16x128_f8f6f4::getOperationName()}},
      // mfma_f32_32x32x64_f8f6f4
      {{32, 32, MfmaTypeId::Fp8Fp6TyId, 3},
       {32, 32, 64, 8, ROCDL::mfma_f32_32x32x64_f8f6f4::getOperationName()}},
      // mfma_scale_f32_32x32x64_f8f6f4

  };
  return MfmaInsnMap;
};

std::pair<mlir::Type, mlir::Type> TypesFromMfmaId(mlir::MLIRContext *ctx,
                                                  MfmaTypeId id) {
  auto f8e5m2 = Float8E5M2Type::get(ctx);
  auto f8e4m3fnuz = Float8E4M3FNUZType::get(ctx);
  auto f8e5m2fnuz = Float8E5M2FNUZType::get(ctx);
  auto f16 = Float16Type::get(ctx);
  auto bf16 = BFloat16Type::get(ctx);
  auto f32 = Float32Type::get(ctx);
  auto i8 = IntegerType::get(ctx, 8, IntegerType::Signed);
  auto f6e3m2fn = Float6E3M2FNType::get(ctx);
  auto f6e2m3fn = Float6E2M3FNType::get(ctx);
  auto f4e2m1fn = Float4E2M1FNType::get(ctx);
  switch (id) {
  case MfmaTypeId::Xf32TyId:
  case MfmaTypeId::Fp32TyId:
    return {f32, f32};
  case MfmaTypeId::Fp16TyId:
    return {f16, f16};
  case MfmaTypeId::Bf16TyId:
    return {bf16, bf16};
  case MfmaTypeId::I8TyId:
    return {i8, i8};
  case MfmaTypeId::Fp8Fp8TyId:
    return {f8e4m3fnuz, f8e4m3fnuz};
  case MfmaTypeId::Fp8Bf8TyId:
    return {f8e4m3fnuz, f8e5m2fnuz};
  case MfmaTypeId::Bf8Fp8TyId:
    return {f8e5m2fnuz, f8e4m3fnuz};
  case MfmaTypeId::Bf8Bf8TyId:
    return {f8e5m2fnuz, f8e5m2fnuz};
  case MfmaTypeId::Fp8Fp6TyId:
    return {f8e4m3fnuz, f6e3m2fn};
  case MfmaTypeId::Fp8Bf6TyId:
    return {f8e4m3fnuz, f6e2m3fn};
  case MfmaTypeId::Bf8Fp6TyId:
    return {f8e5m2fnuz, f6e3m2fn};
  case MfmaTypeId::Bf8Bf6TyId:
    return {f8e5m2fnuz, f6e2m3fn};
  case MfmaTypeId::Fp8Fp4TyId:
    return {f8e4m3fnuz, f4e2m1fn};
  case MfmaTypeId::Bf8Fp4TyId:
    return {f8e5m2fnuz, f4e2m1fn};
  case MfmaTypeId::Fp6Fp6TyId:
    return {f6e3m2fn, f6e3m2fn};
  case MfmaTypeId::Fp6Bf6TyId:
    return {f6e3m2fn, f6e2m3fn};
  case MfmaTypeId::Bf6Fp6TyId:
    return {f6e2m3fn, f6e3m2fn};
  case MfmaTypeId::Bf6Bf6TyId:
    return {f6e2m3fn, f6e2m3fn};
  case MfmaTypeId::Fp6Fp4TyId:
    return {f6e3m2fn, f4e2m1fn};
  case MfmaTypeId::Bf6Fp4TyId:
    return {f6e2m3fn, f4e2m1fn};
  case MfmaTypeId::Fp4Fp4TyId:
    return {f4e2m1fn, f4e2m1fn};
  default:
    llvm_unreachable("unsupported MfmaTypeId!");
  }
}

FailureOr<MfmaInsn> MfmaInsn::selectMfma(unsigned mDim, unsigned nDim,
                                         Type elementTypeA, Type elementTypeB,
                                         int mfmaVersion, bool allowXF32) {
  auto mfmaInsnAttrMap = getMfmaInsnGroupAttrMap();
  MfmaTypeId mfmaId =
      chooseAppropriateMfmaId(elementTypeA, elementTypeB, allowXF32);
  MfmaInsnGroupSelectKey key = {mDim, nDim, mfmaId, mfmaVersion};
  auto it = mfmaInsnAttrMap.find(key);
  if (it == mfmaInsnAttrMap.end())
    return failure();
  auto [instrElementTypeA, instrElementTypeB] =
      TypesFromMfmaId(elementTypeA.getContext(), mfmaId);
  return MfmaInsn(instrElementTypeA, instrElementTypeB, it->second);
}

MfmaInsn::MfmaInsn(Type elementTypeA, Type elementTypeB,
                   const MfmaInsnAttr &attr)
    : elementTypeA(elementTypeA), elementTypeB(elementTypeB), attr(attr) {}

unsigned MfmaInsn::getKDim() { return attr.k; }
unsigned MfmaInsn::getMDim() { return attr.m; }
unsigned MfmaInsn::getNDim() { return attr.n; }
StringRef MfmaInsn::getInsnName() { return attr.insn; }
unsigned MfmaInsn::getKBase() { return attr.kBase; }
Type MfmaInsn::getElementTypeA() { return elementTypeA; }
Type MfmaInsn::getElementTypeB() { return elementTypeB; }
} // namespace mlir
