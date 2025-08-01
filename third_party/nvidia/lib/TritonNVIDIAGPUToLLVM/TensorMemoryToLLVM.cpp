#include "Dialect/NVGPU/IR/Dialect.h"
#include "DotOpToLLVM/MMAHelpers.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// The maximum number of tensor memory registers that can be accessed
// by a single message regardless of shape or repetitions
static constexpr int largestTmemLoadStore = 128;
// The maximum number of thread registers that can be populated by
// multiple messages
static constexpr int maxRegisters = 256;

namespace {

struct TMemAccessAtom {
  int opBitWidth;
  int colsPerThread;
  int rowsPerThread;
  int rowStored;
  const char *opShape;
  bool usesSecondHalfOffset;
};

constexpr TMemAccessAtom TMemAccess32x32b{.opBitWidth = 32,
                                          .colsPerThread = 1,
                                          .rowsPerThread = 1,
                                          .rowStored = 32,
                                          .opShape = "32x32b",
                                          .usesSecondHalfOffset = false};

constexpr TMemAccessAtom TMemAccess16x32bx2{.opBitWidth = 32,
                                            .colsPerThread = 1,
                                            .rowsPerThread = 1,
                                            .rowStored = 32,
                                            .opShape = "16x32bx2",
                                            .usesSecondHalfOffset = true};

constexpr TMemAccessAtom TMemAccess16x256b{.opBitWidth = 256,
                                           .colsPerThread = 2,
                                           .rowsPerThread = 2,
                                           .rowStored = 16,
                                           .opShape = "16x256b",
                                           .usesSecondHalfOffset = false};

struct TMemMessageTraits {
  TMemAccessAtom atom;
  bool usesSecondHalfOffset;
  int numThreadsPerWarp;
  int maxNumRepeats;
  int maxCols;
  int numRows;
  int numCols;
  int numRepeats;
  int numRegs;

  bool operator<(const TMemMessageTraits &other) const {
    return numRegs < other.numRegs;
  }

  LLVM_DUMP_METHOD void dump() const {
    llvm::dbgs() << "TMemMessageTraits:\n";
    llvm::dbgs() << "  atom.opBitWidth: " << atom.opBitWidth << "\n";
    llvm::dbgs() << "  atom.colsPerThread: " << atom.colsPerThread << "\n";
    llvm::dbgs() << "  atom.rowsPerThread: " << atom.rowsPerThread << "\n";
    llvm::dbgs() << "  atom.opShape: " << atom.opShape << "\n";
    llvm::dbgs() << "  atom.usesSecondHalfOffset: " << atom.usesSecondHalfOffset
                 << "\n";
    llvm::dbgs() << "  usesSecondHalfOffset: " << usesSecondHalfOffset << "\n";
    llvm::dbgs() << "  numThreadsPerWarp: " << numThreadsPerWarp << "\n";
    llvm::dbgs() << "  maxNumRepeats: " << maxNumRepeats << "\n";
    llvm::dbgs() << "  maxCols: " << maxCols << "\n";
    llvm::dbgs() << "  numRows: " << numRows << "\n";
    llvm::dbgs() << "  numCols: " << numCols << "\n";
    llvm::dbgs() << "  numRepeats: " << numRepeats << "\n";
    llvm::dbgs() << "  numRegs: " << numRegs << "\n";
  }
};

struct TMemRuntimeInfo {
  static constexpr int numRowsPerWarp = 32;
  int numWarps;
  int numWarpGroups;
  int numElementsPer32B;
  int numElements;
  int numCols;
  int blockM;
  int blockN;
  bool unpackedb16;
  bool useStridedMessage;
  int numBlocks;
  bool blocksInterleaved;
  int numColsPerBlock;
  int colsPerWarpGroup;
  bool splitWarpgroupsAlongM;
  TMemAccessAtom layoutAtom;

  LLVM_DUMP_METHOD void dump() const {
    llvm::dbgs() << "TMemRuntimeInfo:\n";
    llvm::dbgs() << "  numWarps: " << numWarps << "\n";
    llvm::dbgs() << "  numWarpGroups: " << numWarpGroups << "\n";
    llvm::dbgs() << "  numElementsPer32B: " << numElementsPer32B << "\n";
    llvm::dbgs() << "  numElements: " << numElements << "\n";
    llvm::dbgs() << "  numCols: " << numCols << "\n";
    llvm::dbgs() << "  blockM: " << blockM << "\n";
    llvm::dbgs() << "  blockN: " << blockN << "\n";
    llvm::dbgs() << "  unpackedb16: " << unpackedb16 << "\n";
    llvm::dbgs() << "  useStridedMessage: " << useStridedMessage << "\n";
    llvm::dbgs() << "  numBlocks: " << numBlocks << "\n";
    llvm::dbgs() << "  blocksInterleaved: " << blocksInterleaved << "\n";
    llvm::dbgs() << "  numColsPerBlock: " << numColsPerBlock << "\n";
    llvm::dbgs() << "  colsPerWarpGroup: " << colsPerWarpGroup << "\n";
    llvm::dbgs() << "  splitWarpgroupsAlongM: " << splitWarpgroupsAlongM
                 << "\n";
    llvm::dbgs() << "  message shape: " << layoutAtom.opShape << "\n";
  }
};

TMemMessageTraits getTMemMessageFromAtom(const TMemAccessAtom &atom,
                                         int narrowingFactor) {
  TMemMessageTraits m;
  m.atom = atom;
  m.usesSecondHalfOffset = atom.usesSecondHalfOffset;
  m.numThreadsPerWarp = 32;
  m.maxNumRepeats =
      largestTmemLoadStore / (atom.colsPerThread * atom.rowsPerThread);
  m.maxCols = (atom.opBitWidth / 32) * m.maxNumRepeats;
  m.numRows = m.numThreadsPerWarp / atom.rowsPerThread;
  m.numCols = m.maxCols / narrowingFactor;
  m.numRepeats = m.numCols / (atom.opBitWidth / 32);
  m.numRegs = atom.colsPerThread * atom.rowsPerThread * m.numRepeats;
  return m;
}

// Narrow the TMEM message by reducing the number of registers per TMEM
// instruction such that:
// - No instruction uses more than half the available registers at a time.
// - If the total number of registers required by the workload is more than half
//   of the available registers, don't use the largest TMEM message.
int getTMemMessageNarrowingFactor(const TMemAccessAtom &atom,
                                  int workloadThreadRegs, int maxnreg) {
  const int allowedRegUsage = maxnreg / 2;
  int narrowingFactor = 1;
  while (getTMemMessageFromAtom(atom, narrowingFactor).numRegs >
             allowedRegUsage ||
         workloadThreadRegs > allowedRegUsage) {
    workloadThreadRegs /= 2;
    narrowingFactor *= 2;
  }
  return narrowingFactor;
}

int getEffectiveRegs(bool unpackedb16, bool useStridedMessage, int numRegs) {
  // The effective register count is less when using unpacked or strided
  // messages
  if (unpackedb16) {
    numRegs /= 2;
  }
  if (useStridedMessage) {
    numRegs /= 2;
  }
  return std::max(1, numRegs);
}

// If the workload runtime requires fewer registers than the default message
// width, use the widest possible message that matches the workload
TMemMessageTraits constrainMessageFromWorkload(TMemMessageTraits m,
                                               const TMemRuntimeInfo &info,
                                               int numRegs) {
  m.numRegs =
      getEffectiveRegs(info.unpackedb16, info.useStridedMessage, numRegs);
  m.numRegs = std::min(largestTmemLoadStore, m.numRegs);
  // Invert the above formulas to calculate the effective runtime message width
  m.numCols = (m.numRegs * (m.atom.opBitWidth / 32)) /
              (m.atom.colsPerThread * m.atom.rowsPerThread);
  // Half as many registers are needed for 16-bit packed elements,
  // so twice as many columns are accessed per message.
  if (info.unpackedb16)
    m.numCols *= info.numElementsPer32B;
  m.numRepeats = m.numCols / (m.atom.opBitWidth / 32);
  return m;
}

SmallVector<Value> packToI32(const SmallVector<Value> &values, Location loc,
                             ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> packedValues;
  Type elType = values[0].getType();
  int numElementsPer32B = 32 / elType.getIntOrFloatBitWidth();
  if (numElementsPer32B == 1)
    return values;
  Value packed = b.undef(vec_ty(elType, numElementsPer32B));
  for (int i = 0; i < values.size(); i++) {
    Value val = values[i];
    packed = b.insert_element(packed.getType(), packed, val,
                              b.i32_val(i % numElementsPer32B));
    if (i % numElementsPer32B == numElementsPer32B - 1 ||
        i == values.size() - 1) {
      packed = b.bitcast(packed, i32_ty);
      packedValues.push_back(packed);
      packed = b.undef(vec_ty(elType, numElementsPer32B));
    }
  }
  return packedValues;
}

static bool is16x256Layout(RankedTensorType tensorType, Attribute memEncoding,
                           int numWarps) {
  auto tmemLayout =
      dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(memEncoding);
  if (!tmemLayout)
    return false;
  int blockM = tmemLayout.getBlockM();
  int blockN = tmemLayout.getBlockN();
  std::optional<LinearLayout> ll0 =
      getTmemLoadStoreLayout16x256(blockM, blockN, tensorType, numWarps);
  auto ll1 = toLinearLayout(tensorType);
  return ll0.has_value() && ll0.value() == ll1;
}

TMemRuntimeInfo getTMemRuntimeInfo(Operation *op, RankedTensorType tensorType,
                                   MemDescType memType) {
  TMemRuntimeInfo info;
  static_assert(TMemRuntimeInfo::numRowsPerWarp == 32,
                "A single warp must access exactly 32 rows of tmem");
  assert(
      nvidia_gpu::isDistributedLayoutTMemCompatible(op, tensorType, memType) &&
      "unsupported distributed layout for tensor memory");

  info.numWarps = triton::gpu::lookupNumWarps(op);
  assert(info.numWarps % 4 == 0 && "Unexpected number of warps");
  info.numWarpGroups = info.numWarps / 4;
  info.numElementsPer32B = 32 / tensorType.getElementTypeBitWidth();
  auto shapePerCTA = mlir::triton::gpu::getShapePerCTA(tensorType);
  info.numElements = product(shapePerCTA);

  triton::nvidia_gpu::TMemAllocation tmemAlloc =
      triton::nvidia_gpu::getTmemAllocSizes(memType);
  info.numCols = tmemAlloc.numCols;

  info.blockM = 0;
  info.blockN = 0;
  info.unpackedb16 = false;
  if (auto attr = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
          memType.getEncoding())) {
    info.blockM = attr.getBlockM();
    info.blockN = attr.getBlockN();
    assert((!attr.getUnpacked() || info.numElementsPer32B <= 2) &&
           "unsupported unpacked layout");
    info.unpackedb16 = attr.getUnpacked() && (info.numElementsPer32B == 2);
  } else {
    assert(isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
               memType.getEncoding()) &&
           "Expecting a tensor memory encoding attribute");
    info.blockM = 128;
    info.blockN = 32;
  }

  info.splitWarpgroupsAlongM =
      nvidia_gpu::isDistributedLayoutSplitMTmemLoadStore(tensorType, memType,
                                                         info.numWarps);
  info.numBlocks = ceil<int>(info.numElements, info.blockM * info.blockN);
  info.blocksInterleaved = (info.numBlocks > 1 && info.blockM == 64);
  info.numColsPerBlock = info.numCols / info.numBlocks;
  info.useStridedMessage = false;
  if (info.blocksInterleaved) {
    info.numColsPerBlock *= 2;
  }
  if (info.splitWarpgroupsAlongM) {
    info.colsPerWarpGroup = info.numColsPerBlock;
    info.useStridedMessage = true;
    assert(info.blockM == 128);
  } else {
    int numWarpGroupsPerBlock = ceil<int>(info.numWarpGroups, info.numBlocks);
    info.colsPerWarpGroup = info.numColsPerBlock / numWarpGroupsPerBlock;
    // If more than one warp group processes the same block,
    // then fewer columns must be processed per message per warp group
    info.numColsPerBlock /= numWarpGroupsPerBlock;
  }
  if (is16x256Layout(tensorType, memType.getEncoding(), info.numWarps)) {
    assert(info.useStridedMessage == false);
    info.layoutAtom = TMemAccess16x256b;
  } else {
    info.useStridedMessage |= (info.blockM == 64);
    if (info.useStridedMessage) {
      info.layoutAtom = TMemAccess16x32bx2;
    } else {
      info.layoutAtom = TMemAccess32x32b;
    }
  }
  return info;
}

void calculateAddressAndEmitTmemMessage(
    Location loc, Value baseAddress, const TMemRuntimeInfo &info,
    const TMemMessageTraits &message, ConversionPatternRewriter &rewriter,
    const std::function<void(Value, int, std::optional<int>, bool, int, bool)>
        &createMemoryOp) {

  TritonLLVMOpBuilder b(loc, rewriter);
  Value warpId = rewriter.create<nvgpu::WarpIdOp>(loc);
  // Note: optimizing this when we know `info.numWarpGroups` is 1 can result in
  // performance regressions.
  Value warpIdInGroup = b.urem(warpId, b.i32_val(4));
  Value warpGroupId = b.udiv(warpId, b.i32_val(4));

  // When split along M, blockM=128 and num_warps=8, and a strided message is
  // selected such that all 8 warps read a 16 rows of a block at a time.
  int blocksPerWarpTile = info.splitWarpgroupsAlongM ? 1 : info.numWarpGroups;
  for (int block = 0; block < info.numBlocks; block += blocksPerWarpTile) {
    Value address = b.ptrtoint(i32_ty, baseAddress);
    Value blockId = b.i32_val(block);
    Value startColumnId = b.i32_val(0);
    Value blockRowId =
        b.mul(warpIdInGroup, b.i32_val(TMemRuntimeInfo::numRowsPerWarp));

    if (info.splitWarpgroupsAlongM) {
      // When split along M warp 0 loads the 16 top rows, warp 4 loads the 16
      // bottom rows.
      blockRowId = b.add(blockRowId, b.mul(warpGroupId, b.i32_val(16)));
    } else {
      int numWarpGroupsPerBlock = ceil<int>(info.numWarpGroups, info.numBlocks);
      Value warpGroupIdInBlock =
          b.urem(warpGroupId, b.i32_val(numWarpGroupsPerBlock));
      blockId =
          b.add(blockId, b.udiv(warpGroupId, b.i32_val(numWarpGroupsPerBlock)));
      startColumnId =
          b.mul(warpGroupIdInBlock, b.i32_val(info.colsPerWarpGroup));
    }

    if (info.blocksInterleaved) {
      Value blockIdIsOdd = b.urem(blockId, b.i32_val(2));
      Value blockIdPrevEven = b.sub(blockId, blockIdIsOdd);
      blockRowId = b.add(blockRowId, b.mul(blockIdIsOdd, b.i32_val(16)));
      startColumnId =
          b.add(startColumnId,
                b.mul(blockIdPrevEven, b.i32_val(info.numColsPerBlock / 2)));
    } else {
      startColumnId =
          b.add(startColumnId, b.mul(blockId, b.i32_val(info.numColsPerBlock)));
    }

    // A strided message accesses twice as many columns per message,
    // thus half as many messages are required
    int numColumns = info.useStridedMessage ? info.numColsPerBlock / 2
                                            : info.numColsPerBlock;
    // For messages that span only 16 rows (e.g. 16x256b), multiple messages
    // are required to cover the entire set of rows per warp.
    int numRowPerWarp =
        (info.layoutAtom.rowStored == 16 && info.blockM == 64) ? 16 : 32;

    for (int rowStart = 0; rowStart < numRowPerWarp;
         rowStart += message.numRows) {
      for (int colStart = 0; colStart < std::max(1, numColumns);
           colStart += message.numCols) {

        Value rowOffset = b.add(blockRowId, b.i32_val(rowStart));
        Value warpGroupAddress =
            b.add(address, b.shl(rowOffset, b.i32_val(16)));
        warpGroupAddress = b.add(warpGroupAddress, startColumnId);

        std::optional<int> secondHalfColOffset;
        if (info.useStridedMessage) {
          // Offset to half way through the set of columns for this warpgroup.
          secondHalfColOffset = numColumns;
        }
        createMemoryOp(warpGroupAddress, colStart, secondHalfColOffset,
                       info.unpackedb16, message.numRegs,
                       info.useStridedMessage);
      }
    }
  }
}

void createTensorMemoryStore(Location loc, Value address, int colOffset,
                             SmallVector<Value> &srcs,
                             std::optional<int> secondHalfOffset, Value pred,
                             bool unpacked, const TMemAccessAtom &atom,
                             ConversionPatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  std::string packedStr = unpacked ? ".unpack::16b" : "";
  unsigned numRepeats = srcs.size() / (atom.rowsPerThread * atom.colsPerThread);
  std::string opcode = "@$0 tcgen05.st.sync.aligned." +
                       std::string(atom.opShape) + ".x" +
                       std::to_string(numRepeats) + packedStr;
  opcode += ".b32 [$1 + " + std::to_string(colOffset) + "], ";
  if (secondHalfOffset)
    opcode += std::to_string(*secondHalfOffset) + ", {";
  else
    opcode += "{";

  SmallVector<PTXInstr::Operand *> operands;
  operands.push_back(ptxBuilder.newOperand(pred, "b"));
  operands.push_back(ptxBuilder.newOperand(address, "r"));
  for (int i = 0; i < srcs.size(); i++) {
    opcode += "$" + std::to_string(i + 2);
    auto *resultOp = ptxBuilder.newOperand(srcs[i], "r");
    operands.push_back(resultOp);
    if (i < srcs.size() - 1)
      opcode += ", ";
  }
  opcode += "};";

  auto &st = *ptxBuilder.create<PTXInstr>(opcode);
  st(operands, /*onlyAttachMLIRArgs=*/true);
  Type voidTy = void_ty(rewriter.getContext());
  ptxBuilder.launch(rewriter, loc, voidTy);
}

TMemMessageTraits selectTMemMessage(const TMemRuntimeInfo &info, int maxnreg) {
  auto atom = info.layoutAtom;

  int totalRegsNeeded =
      getEffectiveRegs(info.unpackedb16, info.useStridedMessage,
                       info.numCols / info.numWarpGroups);
  int narrowingFactor =
      getTMemMessageNarrowingFactor(atom, totalRegsNeeded, maxnreg);
  auto narrowedMessage = getTMemMessageFromAtom(atom, narrowingFactor);
  narrowedMessage = constrainMessageFromWorkload(narrowedMessage, info,
                                                 narrowedMessage.numRegs);

  auto maxWidthMessage = getTMemMessageFromAtom(atom, /*narrowingFactor=*/1);
  int numRegs = (info.layoutAtom.rowStored == 16) ? info.colsPerWarpGroup / 2
                                                  : info.colsPerWarpGroup;
  maxWidthMessage =
      constrainMessageFromWorkload(maxWidthMessage, info, numRegs);
  return std::min(narrowedMessage, maxWidthMessage);
}

// Get the maximum number of registers per thread based on the context. This is
// by default 256, but it can be overridden by `ttg.maxnreg` set on the module
// or a contextual register limit set by the compiler on partitions.
static int getContextualMaxNReg(Operation *op) {
  // Check the immediate parent op to see if it places a register constraint.
  auto getFromParent = [](Operation *op) -> std::optional<int> {
    Operation *parent = op->getParentOp();
    if (auto mod = dyn_cast<ModuleOp>(parent)) {
      if (auto attr = mod->getAttrOfType<IntegerAttr>(AttrMaxRegistersName))
        return attr.getInt();
      return {};
    }

    if (auto partitions = dyn_cast<WarpSpecializePartitionsOp>(parent)) {
      // Check if the partition has reduced registers.
      unsigned idx = op->getParentRegion()->getRegionNumber();
      if (auto actRegisters = partitions.getParentOp().getActualRegisters())
        return (*actRegisters)[1 + idx];
      return {};
    }

    if (auto wsOp = dyn_cast<WarpSpecializeOp>(op->getParentOp())) {
      // Check the register usage of the default warpgroup.
      if (auto actRegisters = wsOp.getActualRegisters())
        return actRegisters->front();
      return {};
    }

    return {};
  };

  // PTXAS validates the register usage of `tcgen05.ld` and `tcgen05.st`
  // instructions based on the static number of registers set on the module, not
  // the dynamic allocation. This just means the register limit used for the
  // purpose of subtiling TMEM messages cannot be higher than the module's.
  auto mod = op->getParentOfType<ModuleOp>();
  int maxnreg = maxRegisters;

  for (; op != mod; op = op->getParentOp()) {
    if (std::optional<int> limit = getFromParent(op)) {
      maxnreg = std::min(maxnreg, *limit);
      break;
    }
  }

  if (auto maxnregAttr = mod->getAttrOfType<IntegerAttr>(AttrMaxRegistersName))
    maxnreg = std::min<int>(maxnreg, maxnregAttr.getInt());

  return maxnreg;
}

static void lowerStoreToTensorMemory(Location loc, Operation *op,
                                     TypedValue<RankedTensorType> src,
                                     TypedValue<MemDescType> dest, Value llSrc,
                                     Value pred, Value tmemBase,
                                     ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> srcValues = unpackLLElements(loc, llSrc, rewriter);
  srcValues = packToI32(srcValues, loc, rewriter);
  auto info = getTMemRuntimeInfo(op, src.getType(), dest.getType());
  const TMemMessageTraits message =
      selectTMemMessage(info, getContextualMaxNReg(op));
  int regIdx = 0;
  calculateAddressAndEmitTmemMessage(
      loc, tmemBase, info, message, rewriter,
      [&](Value startAddress, int colOffset,
          std::optional<int> secondHalfColOffset, bool unpackedb16,
          int regsPerMsg, bool useStridedMessage) {
        SmallVector<Value> srcValuesSlice(srcValues.begin() + regIdx,
                                          srcValues.begin() + regIdx +
                                              regsPerMsg);
        regIdx += regsPerMsg;
        createTensorMemoryStore(loc, startAddress, colOffset, srcValuesSlice,
                                secondHalfColOffset, pred, unpackedb16,
                                message.atom, rewriter);
      });
  rewriter.create<NVVM::Tcgen05WaitOp>(loc, NVVM::Tcgen05WaitKind::STORE);

  // Emit a barrier to ensure all threads have finished writing to tensor memory
  // before any use of the tensor memory.
  b.barrier();
}

struct TensorMemoryAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMAllocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value base = rewriter.create<nvgpu::TensorMemoryBaseAddress>(loc);
    Value baseInt = b.ptrtoint(i32_ty, base);
    int colOffset = cast<IntegerAttr>(op->getAttr("tensor_memory_col_offset"))
                        .getValue()
                        .getZExtValue();
    int rowOffset = cast<IntegerAttr>(op->getAttr("tensor_memory_row_offset"))
                        .getValue()
                        .getZExtValue();
    Value allocAddress = b.add(baseInt, b.i32_val(colOffset | rowOffset << 16));
    // Cast to address space 3 as the shared memory object uses 3.
    // TODO: clean this up and use either a int or ptr address space 6
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    Value ptr = b.inttoptr(ptrTy, allocAddress);
    SmallVector<unsigned> order(op.getType().getRank());
    std::iota(order.begin(), order.end(), 0);
    std::reverse(order.begin(), order.end());
    auto shape = op.getType().getShape();

    if (op.getSrc()) {
      lowerStoreToTensorMemory(loc, op, op.getSrc(), op.getResult(),
                               adaptor.getSrc(), b.i1_val(true), ptr, rewriter);
    }

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

Value createTensorMemoryLoad(Location loc, triton::nvidia_gpu::TMEMLoadOp op,
                             Value address, int colOffset,
                             std::optional<int> secondHalfOffset, bool unpacked,
                             int numRegPerMessage, const TMemAccessAtom &atom,
                             ConversionPatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  // If the memory is unpacked we need to pack on the fly when loading.
  std::string packedStr = unpacked ? ".pack::16b" : "";
  unsigned numRepeats =
      numRegPerMessage / (atom.rowsPerThread * atom.colsPerThread);
  std::string opcode = "tcgen05.ld.sync.aligned." + std::string(atom.opShape) +
                       ".x" + std::to_string(numRepeats) + packedStr + ".b32 {";

  SmallVector<PTXInstr::Operand *> operands;
  for (int i = 0; i < numRegPerMessage; i++) {
    opcode += "$" + std::to_string(i);
    auto *resultOp = ptxBuilder.newOperand("=r");
    operands.push_back(resultOp);
    if (i < numRegPerMessage - 1)
      opcode += ", ";
  }
  opcode += "}, [$" + std::to_string(numRegPerMessage) + " + " +
            std::to_string(colOffset) + "]";
  if (secondHalfOffset)
    opcode += ", " + std::to_string(*secondHalfOffset);
  opcode += ";";
  operands.push_back(ptxBuilder.newOperand(address, "r"));
  auto &ld = *ptxBuilder.create<PTXInstr>(opcode);
  ld(operands, /*onlyAttachMLIRArgs=*/true);

  // LLVM inline_asm with 1 result cannot return a struct.
  Type retTy;
  if (numRegPerMessage == 1) {
    retTy = i32_ty;
  } else {
    SmallVector<Type> elemTypes(numRegPerMessage, i32_ty);
    MLIRContext *ctx = op.getContext();
    retTy = struct_ty(elemTypes);
  }
  Value ret = ptxBuilder.launch(rewriter, loc, retTy);
  return ret;
}

static SmallVector<Value> unpackResults(Value packedValues, Type elemTy,
                                        int numCols, Location loc,
                                        ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> resultVals;
  int numElementsPer32B = 32 / elemTy.getIntOrFloatBitWidth();
  Type packedType = elemTy;
  if (numElementsPer32B > 1)
    packedType = vec_ty(elemTy, numElementsPer32B);

  auto unpackElement = [&](Value result) {
    result = b.bitcast(result, packedType);
    if (numElementsPer32B > 1) {
      for (int j = 0; j < numElementsPer32B; j++) {
        Value elem = b.extract_element(elemTy, result, b.i32_val(j));
        resultVals.push_back(elem);
      }
    } else {
      resultVals.push_back(result);
    }
  };

  if (isa<LLVM::LLVMStructType>(packedValues.getType())) {
    for (int i = 0; i < numCols; i++) {
      Value result = b.extract_val(i32_ty, packedValues, i);
      unpackElement(result);
    }
  } else {
    unpackElement(packedValues);
  }
  return resultVals;
}

struct TensorMemoryLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMLoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto llvmElemTy =
        getTypeConverter()->convertType(op.getSrc().getType().getElementType());
    auto tmemBase = adaptor.getSrc();

    auto info = getTMemRuntimeInfo(op, cast<RankedTensorType>(op.getType()),
                                   cast<MemDescType>(op.getSrc().getType()));
    const TMemMessageTraits message =
        selectTMemMessage(info, getContextualMaxNReg(op));
    SmallVector<Value> resultVals;
    calculateAddressAndEmitTmemMessage(
        loc, tmemBase, info, message, rewriter,
        [&](Value startAddress, int colOffset,
            std::optional<int> secondHalfColOffset, bool unpackedb16,
            int regsPerMessage, bool useStridedMessage) {
          Value packedValues = createTensorMemoryLoad(
              loc, op, startAddress, colOffset, secondHalfColOffset,
              unpackedb16, regsPerMessage, message.atom, rewriter);
          auto results =
              unpackResults(packedValues, op.getType().getElementType(),
                            regsPerMessage, loc, rewriter);
          resultVals.append(results.begin(), results.end());
        });
    Type structTy = getTypeConverter()->convertType(op.getType());
    Value resultStruct =
        packLLElements(loc, getTypeConverter(), resultVals, rewriter, structTy);
    // Wait insertion could be moved to the TTGIR level if needed.
    rewriter.create<NVVM::Tcgen05WaitOp>(loc, NVVM::Tcgen05WaitKind::LOAD);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct TensorMemoryStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMStoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto llvmElemTy =
        getTypeConverter()->convertType(op.getDst().getType().getElementType());
    auto tmemBase = adaptor.getDst();
    Value pred = adaptor.getPred();
    lowerStoreToTensorMemory(loc, op, op.getSrc(), op.getDst(),
                             adaptor.getSrc(), pred, tmemBase, rewriter);

    rewriter.eraseOp(op);
    return success();
  }
};

static Value
createBlockedScalesSMEMDescriptor(ConversionPatternRewriter &rewriter,
                                  Location loc, Value baseSrc) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  static_assert(sizeof(NVIDIA::SMEMDescriptor) == 8,
                "Descriptor size should be 64 bits.");
  NVIDIA::SMEMDescriptor desc;
  desc.descriptor = 0;
  desc.swizzlingMode = 0;                    // No swizzling for now
  desc.leadDimensionBaseOffset = 16 >> 4;    // 16 bytes
  desc.strideDimensionBaseOffset = 128 >> 4; // 8 x 16 bytes
  // See matrix-descriptor-encode(x) function in the ptx doc.
  // matrix-descriptor-encode(addr) = (addr & 0x3FFFF) >> 4
  auto smemAddr = b.ptrtoint(i64_ty, baseSrc);
  return b.add(b.int_val(64, desc.descriptor),
               b.lshr(b.shl(smemAddr, b.int_val(64, 46)), b.int_val(64, 50)));
}

static void createCommit(ConversionPatternRewriter &rewriter, Location loc,
                         Value barrier, Value pred) {
  PTXBuilder ptxBuilder;
  auto *barrierOperand = ptxBuilder.newAddrOperand(barrier, "r");
  std::string opcode = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64";
  auto &barrierOp = *ptxBuilder.create<PTXInstr>(opcode);
  barrierOp(barrierOperand).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void createTcgen05Cp(ConversionPatternRewriter &rewriter, Location loc,
                            Value tmem_address, Value src_desc, Value pred) {
  PTXBuilder ptxBuilder;
  auto dst = ptxBuilder.newAddrOperand(tmem_address, "r");
  auto src = ptxBuilder.newOperand(src_desc, "l");
  std::string opcode = "tcgen05.cp.cta_group::1.warpx4.32x128b";
  auto &op = *ptxBuilder.create<PTXInstr>(opcode);
  op({dst, src}).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

struct TensorMemoryCopyOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMCopyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    assert(isa<triton::gpu::SharedMemorySpaceAttr>(srcTy.getMemorySpace()));

    Value baseSrc =
        LLVM::getSharedMemoryObjectFromStruct(
            loc, adaptor.getSrc(),
            typeConverter->convertType(srcTy.getElementType()), rewriter)
            .getBase();

    Value baseDst = adaptor.getDst();
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    auto llvmElementTy = typeConverter->convertType(srcTy.getElementType());

    // The following codegen assumes that we use tcgen05.cp only with
    // the warpx4.32x128b mode, to load blocked scales from MXFP.
    // We will expand the support as we find more use cases for the instruction.

    auto ll = toLinearLayout(srcTy);
    // flattenOuts flattens into fortran order, so need to transpose first to
    // get C-order
    auto ctx = op.getContext();
    auto outDimNames = standardOutDimNames(ctx, srcTy.getRank());
    std::reverse(outDimNames.begin(), outDimNames.end());
    ll = ll.transposeOuts(outDimNames).flattenOuts();
    auto invLayout = ll.flattenOuts().invert();
    auto kDim = *ll.getOutDimNames().begin();

    Value smemDesc = createBlockedScalesSMEMDescriptor(rewriter, loc, baseSrc);
    Value pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);

    auto createCopy = [&](int repMorN, int repK) {
      for (int i = 0; i < repMorN; ++i) {
        for (int j = 0; j < repK; ++j) {
          // Multiple copies of 32x128b blocks are laid out along M/N first then
          // K
          auto colOffset = b.int_val(32, (j * repMorN + i) * 4);
          auto tmemAddr = b.add(b.ptrtoint(i32_ty, baseDst), colOffset);
          auto blockSize = (32 * 128) / llvmElementTy.getIntOrFloatBitWidth();
          auto linearIdx = (i * repK + j) * blockSize;
          auto smemOffset = applyLinearLayout(loc, rewriter, invLayout,
                                              {{kDim, b.i32_val(linearIdx)}})[0]
                                .second;
          auto smemAddr = b.gep(elemPtrTy, llvmElementTy, baseSrc, smemOffset);
          smemDesc = createBlockedScalesSMEMDescriptor(rewriter, loc, smemAddr);
          createTcgen05Cp(rewriter, loc, tmemAddr, smemDesc, pred);
        }
      }
    };

    // Break up src axes into rep_m x rep_k x 32x128b, where rep_m = BLOCK_M /
    // 128 and rep_k = BLOCK_K / 128 32x128b blockes are contiguously laid out
    // in SMEM. rep_m * rep_k copies of such blocks are consumed by one
    // dot_scaled op for given BLOCK_M / BLOCK_K. Some axes of the scale shape
    // can be flattened into one, to reduce the rank of the load. Since rep_m
    // blocks are not contiguous in SMEM, we need to identify the original rep_m
    // axis from the given input shape.

    // The SMEM shapes are expected to be one of the followings. As long as
    // rep_m and rep_k can be identified correctly, other patterns are allowed.
    // * (rep_m x 32, 16B), meant only for TMEMCopy unit tests
    // * (rep_m, rep_k * 32 x 4 x 4B), 2D scale load with cp.async
    // * (rep_m, rep_k, 32, 16B), 4D scale load with TMA
    // * (1, rep_m, rep_k, 2, 256B), 5D scale load with TMA
    // * (rep_m, rep_k, 32, 4, 4B), 5D scale load with cp.async
    auto elemBits = srcTy.getElementType().getIntOrFloatBitWidth();
    int prodInner = 1;
    int repMorN = 1;
    int repK = 1;

    for (int i = srcTy.getRank() - 1; i >= 0; --i) {
      prodInner *= srcTy.getDimSize(i);
      if (prodInner * elemBits >= 32 * 128) {
        if (i == 0) {
          repMorN = prodInner * elemBits / (32 * 128);
          repK = 1;
        } else if (i == 1) {
          repMorN = srcTy.getDimSize(0);
          repK = prodInner * elemBits / (32 * 128);
        } else {
          if (srcTy.getDimSize(0) == 1 &&
              srcTy.getDimSize(srcTy.getRank() - 1) == 256) {
            repMorN = srcTy.getDimSize(1);
            repK = srcTy.getDimSize(2);
          } else {
            repMorN = srcTy.getDimSize(0);
            repK = srcTy.getDimSize(1);
          }
        }
        break;
      }
    }

    createCopy(repMorN, repK);

    if (op.getBarrier()) {
      auto barrier = LLVM::getSharedMemoryObjectFromStruct(
          op.getLoc(), adaptor.getBarrier(), i64_ty, rewriter);
      createCommit(rewriter, loc, barrier.getBase(), pred);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct MemDescIndexOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::MemDescIndexOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::MemDescIndexOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());

    if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
            srcTy.getEncoding())) {
      return failure();
    }

    // newBase = base + offset
    auto tmemBase = adaptor.getSrc();
    auto idx = op.getIndex();
    triton::nvidia_gpu::TMemAllocation tmemAlloc =
        triton::nvidia_gpu::getTmemAllocSizes(cast<MemDescType>(dstTy));
    int numColOffset = tmemAlloc.numCols;
    Value newBase = b.ptrtoint(rewriter.getI32Type(), tmemBase);
    newBase = rewriter.create<LLVM::AddOp>(
        loc, newBase,
        rewriter.create<LLVM::MulOp>(loc, idx, b.i32_val(numColOffset)));
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    rewriter.replaceOp(op, b.inttoptr(elemPtrTy, newBase));
    return success();
  }
};

class MemDescReinterpretOpConversion
    : public ConvertOpToLLVMPattern<MemDescReinterpretOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MemDescReinterpretOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
            op.getSrc().getType().getEncoding())) {
      return failure();
    }
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

struct TMEMSubSliceOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMSubSliceOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::TMEMSubSliceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMSubSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());

    auto encoding = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
        srcTy.getEncoding());
    auto shapePerCTA = getShapePerCTA(srcTy);
    int blockN = encoding.getBlockN();
    int blockM = encoding.getBlockM();
    int offsetCol = 0;
    int offsetRow = 0;
    assert(llvm::is_contained({64, 128}, blockM) && "checked by the verifier");
    offsetCol = op.getN();

    if (blockM == 64) {
      // The layout interleaves blocks along the N dimension with the rows, such
      // that the odd numbered blocks are in lanes [16, 32), below the previous
      // even-numbered block.
      int blockOffset = op.getN() / blockN;
      if (blockOffset % 2) {
        // Offset into rows [16, 32).
        offsetRow = 16;
        // Normalize column offset to the even block.
        offsetCol -= blockN;
      }
      offsetCol -= blockN * (blockOffset / 2);
    }

    if (!encoding.getUnpacked()) {
      // Adjust the column offset based on the element size.
      int numElementsPer32B = 32 / srcTy.getElementTypeBitWidth();
      if (offsetCol % numElementsPer32B != 0) {
        return failure();
      }
      offsetCol /= numElementsPer32B;
    }

    Value tmemBase = adaptor.getSrc();
    Value offsetVal = b.i32_val(offsetCol | offsetRow << 16);
    Value newBase = b.add(b.ptrtoint(i32_ty, tmemBase), offsetVal);
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    rewriter.replaceOp(op, b.inttoptr(elemPtrTy, newBase));
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateTensorMemoryOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<TensorMemoryAllocOpConversion, TensorMemoryLoadOpConversion,
               TensorMemoryStoreOpConversion, TensorMemoryCopyOpConversion,
               TMEMSubSliceOpConversion>(typeConverter, benefit);
  return;
}

void mlir::triton::NVIDIA::populateTensorMemorySubviewOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MemDescIndexOpConversion>(typeConverter, benefit);
  patterns.add<MemDescReinterpretOpConversion>(typeConverter, benefit);
  return;
}
