#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>
#include <numeric>

namespace mlir {
class DominanceInfo;
class PostDominanceInfo;

namespace triton {
class ModuleAxisInfoAnalysis;
class LoadOp;
class StoreOp;
class FuncOp;
namespace gpu {
class SwizzledSharedEncodingAttr;
}
} // namespace triton

// Return a tuple of two or three entries representing the shape of the
// instruction used to perform a matrix multiplication operation.
// Version = 1: <m, n>
// Version = 2: <1, m, n>
// Version = 3: <m, n, k>
SmallVector<unsigned, 3> mmaVersionToInstrShape(int version,
                                                const ArrayRef<int64_t> &shape,
                                                Type type, int numWarps);

// Return true if the Load uses block pointer.
bool isLoadFromTensorPtr(triton::LoadOp op);

// Gets the order of a tensor from its contiguity. Places the dimensions with
// the largest contiguity as the inner most dimension. If the contiguity is
// all ones, returns the order {dim - 1, dim - 2, ..., 0}
SmallVector<unsigned, 4>
getOrderFromContiguity(const SmallVector<int64_t> &contiguity);

// Return the operand used to access the memory in the operation
Value getMemAccessPtr(Operation *op);

// Return bitwidth of tensor element
unsigned getElementBitWidth(RankedTensorType type);

// Calculate the optimal number of elements per thread for a given operation
// along an axis with greatest continuity.
unsigned
getNumElementsPerThread(Operation *op, SmallVector<unsigned> order,
                        triton::ModuleAxisInfoAnalysis &axisInfoAnalysis);

// Returns whether the op is a "view op", i.e. doesn't move any data
bool isView(Operation *op);

// Returns whether the op is a "noop op", i.e. has one input and one output
// and lowers to llvm as the identity function (returns the input)
bool isNoop(Operation *op);

/* Dump Triton IR in graphviz dot format.
 *
 * You can override `onValue` and `onOperation` in a subclass to mark
 * specific Values and Operations. The below subclass
 * GraphLayoutMarker is an example.
 *
 * Default NodeInfo for Value nodes:
 *   {{"shape": "box"},
 *    {"style", "filled"},
 *    {"fillcolor", "white"},
 *    {"label", shapeStr}}
 *
 * Default NodeInfo for Operation nodes:
 *   {{"shape": "ellipse"},
 *    {"style", "filled"},
 *    {"fillcolor", "white"},
 *    {"label", operationName}}
 *
 * If the key "label" is not set by `onValue` or `onOperation`, default labels
 * will be generated. For Value node, the default label is the shape string and
 * for Operation node, it is the operation name.
 *
 * Reference:
 *   https://graphviz.org/doc/info/shapes.html
 *   https://graphviz.org/doc/info/colors.html
 *
 * Usage:
 *   C++:   GraphDumper().dumpToFile(func, "func.dot");
 *   Shell: dot -Tjpg func.dot -o func.jpg
 */
class GraphDumper {
public:
  using NodeInfo = std::map<std::string, std::string>;

  // Override this function to mark specific Values
  virtual NodeInfo onValue(Value value) const;
  // Override this function to mark specific Operations
  virtual NodeInfo onOperation(Operation *op) const;

  std::string dump(triton::FuncOp func) const;
  void dumpToFile(triton::FuncOp func, const std::string &filename) const;

protected:
  std::string getShapeStr(const Type &type) const;

  std::string getUniqueId(Value value) const;
  std::string getUniqueId(Operation *op) const;

  std::string emitNode(const std::string &id, const NodeInfo style) const;
  std::string emitEdge(const std::string &srcId,
                       const std::string &destId) const;

  std::string emitValueNode(Value value) const;
  std::string emitOperationNode(Operation *op) const;
};

/* A subclass of GraphDumper that marks different layout kinds in different
 * colors.*/
class GraphLayoutMarker : public GraphDumper {
public:
  NodeInfo onValue(Value value) const override;

protected:
  std::string getColor(const Type &type) const;
};

// Infers the encoding of the result of op given the source encoding.
Attribute inferDstEncoding(Operation *op, Attribute encoding);

// Infers the encoding of the source of op given the result encoding.
Attribute inferSrcEncoding(Operation *op, Attribute encoding);

bool isExpensiveLoadOrStore(Operation *op);

bool canFoldIntoConversion(Operation *op, Attribute targetEncoding);

// Replace ForOp with a new ForOp with extra operands. The YieldOp is not
// updated and needs to be updated separately for the loop to be correct.
scf::ForOp replaceForOpWithNewSignature(
    OpBuilder &rewriter, scf::ForOp loop, ValueRange newIterOperands,
    SmallVectorImpl<std::tuple<Value, Value>> &replacements);
scf::ForOp replaceForOpWithNewSignature(OpBuilder &rewriter, scf::ForOp loop,
                                        ValueRange newIterOperands);
[[nodiscard]] scf::ForOp addIterArgsToLoop(OpBuilder &rewriter, scf::ForOp loop,
                                           ValueRange newIterOperands);

// Replace WhileOp with a new WhileOp with extra operands. The YieldOp is not
// updated and needs to be updated separately for the loop to be correct.
scf::WhileOp replaceWhileOpWithNewSignature(
    OpBuilder &rewriter, scf::WhileOp loop, ValueRange newIterOperands,
    TypeRange newResultTypes,
    SmallVectorImpl<std::tuple<Value, Value>> &replacements);
scf::WhileOp replaceWhileOpWithNewSignature(OpBuilder &rewriter,
                                            scf::WhileOp loop,
                                            ValueRange newIterOperands,
                                            TypeRange newResultTypes);

// Replace IfOp with a new IfOp with extra results operands. The YieldOp is not
// updated and needs to be updated separately for the bodies to be correct.
scf::IfOp replaceIfOpWithNewSignature(
    OpBuilder &rewriter, scf::IfOp loop, TypeRange newResultTypes,
    SmallVectorImpl<std::tuple<Value, Value>> &replacements);
scf::IfOp replaceIfOpWithNewSignature(OpBuilder &rewriter, scf::IfOp ifOp,
                                      TypeRange newResultTypes);

// Append the given |newOperands| to the |forOp|'s yield op.
void appendToForOpYield(scf::ForOp forOp, ArrayRef<Value> newOperands);

Operation *cloneWithInferType(mlir::OpBuilder &rewriter, Operation *op,
                              IRMapping &mapping);

// Get backward slice of tensor values starting from the root node along with
// encoding propagation.
LogicalResult getConvertBackwardSlice(
    OpOperand &root, SetVector<Value> &slice, Attribute rootEncoding,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation = nullptr,
    std::function<Value(OpOperand &, Attribute)> getExistingConversion =
        nullptr);

// Populate pattern to remove dead cycles in ForOp.
void populateForOpDeadArgumentElimination(RewritePatternSet &patterns);

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(OpBuilder &b, Location loc, Value linear,
                               ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order);

SmallVector<Value> delinearize(OpBuilder &b, Location loc, unsigned linear,
                               ArrayRef<unsigned> shape);

SmallVector<Value> delinearize(OpBuilder &b, Location loc, Value linear,
                               ArrayRef<unsigned> shape);
Value linearize(OpBuilder &b, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape, ArrayRef<unsigned> order);

Value linearize(OpBuilder &b, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape);

// Return true if the op is a pure elementwise_inline_asm op with a single
// operand and single result.
bool isPureUnaryInlineAsm(Operation *op);

// read the compute capability from the module attributes
int getNVIDIAComputeCapability(Operation *module);

// Read the amd target from the module attributes
std::optional<StringRef> getAMDArch(Operation *module);

std::optional<mlir::triton::gpu::SwizzledSharedEncodingAttr>
getSharedEncIfAllUsersAreDotEnc(Value val, bool &incompatible);

// Convert \param op operands and results to layout \param encoding.
void convertOpEncoding(Attribute encoding, Operation *op);

// Returns the original memory allocation for a memdesc value
triton::gpu::LocalAllocOp findShmemAlloc(Value operand);

// Returns MMAs inside a for loop that are multi-buffered for pipeline analysis
SmallVector<Operation *>
getMMAsWithMultiBufferredOperands(scf::ForOp forOp,
                                  SmallVector<Operation *> &mmaOps);

// Given a list of ops, find the naerest common dominator of all ops or return
// null if one could not be found. The ops are allowed to be in different
// regions. The result op is not necessarily one of the ops in the list.
Operation *findNearestCommonDominator(ArrayRef<Operation *> ops,
                                      DominanceInfo &domInfo);
// Given a list of ops, find the naerest common postdominator of all ops or
// return null if one could not be found. The ops are allowed to be in different
// regions. The result op is not necessarily one of the ops in the list.
Operation *findNearestCommonPostDominator(ArrayRef<Operation *> ops,
                                          PostDominanceInfo &postDomInfo);

/// Visit the operands of `op` and the operands of any nested ops defined
/// outside of `op`.
void visitNestedOperands(Operation *op,
                         function_ref<void(OpOperand &)> visitor);
/// Visit the operands of `op` and the operands of any nested ops defined
/// outside of `op`.
void visitNestedOperands(Operation *op, function_ref<void(Value)> visitor);
/// Get the operands of `op` and the operands of any nested ops defined outside
/// of `op`.
SetVector<Value> getNestedOperands(Operation *op);

// Erase the given loop carried values from the loop, where `loop` is replaced
// with a new loop.
void eraseLoopCarriedValues(scf::ForOp &loop, llvm::BitVector indices);
} // namespace mlir

namespace mlir::triton {
/// Replace all uses of `oldUse` with `val` and propagate the type if needed.
/// This is useful when we need to change a memory descriptor from immutable to
/// mutable.
/// The callback is invoked for each pair of an old and a cloned memdesc op
/// as the type is propagated.
void replaceUsesAndPropagateType(
    OpBuilder &builder, Operation *oldUse, Value val,
    std::function<void(Operation *, Operation *)> callback = nullptr);

/// Replace all uses of `old` with a local load from `alloc` unless the use is a
/// `ttg.local_alloc` with a matching shared encoding, in which case the shared
/// memory is forwarded directly into the use. Returns the `ttg.local_load` if
/// it created one.
triton::gpu::LocalLoadOp
replaceUsesWithLocalLoad(OpBuilder &builder, OpResult old,
                         TypedValue<triton::gpu::MemDescType> alloc,
                         TypedValue<triton::gpu::AsyncTokenType> token = {});

// Return true if the value comes from a load or a block argument.
// This will skip convert layouts and memdesc views.
// This is a helper useful to know if value is likely to come from shared memory
// after converting loads into async loads.
bool comesFromLoadOrBlockArg(Value v);

// For structured control flow ops, returns the values associated with the
// `resultIdx`th result.
SmallVector<Value> getTiedArgs(Operation *op, int resultIdx);

} // namespace mlir::triton

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
