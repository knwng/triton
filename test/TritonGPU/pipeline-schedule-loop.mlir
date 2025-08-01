// RUN: triton-opt %s -allow-unregistered-dialect -split-input-file -tritongpu-schedule-loops -canonicalize | FileCheck %s

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#A = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 16}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 32]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_dep
tt.func @one_dep(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> tensor<128x32xf16, #A> {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (tensor<128x32xf16, #A>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %a = tt.load %a_ptr_init {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res = arith.addf %acc, %a : tensor<128x32xf16, #A>
    scf.yield %res : tensor<128x32xf16, #A>
  }
  // CHECK: tt.scheduled_max_stage
  tt.return %loop#0 : tensor<128x32xf16, #A>
}

// CHECK-LABEL: @parallel_deps
tt.func @parallel_deps(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>,
                       %b_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%acc_a = %init, %acc_b = %init) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %a = tt.load %a_ptr_init {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %b = tt.load %a_ptr_init {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res_a = arith.addf %acc_a, %a : tensor<128x32xf16, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res_b = arith.addf %acc_b, %b : tensor<128x32xf16, #A>
    scf.yield %res_a, %res_b : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
  }
  tt.return %loop#0, %loop#1 : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
}

// CHECK-LABEL: @parallel_deps_uneven1
tt.func @parallel_deps_uneven1(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>,
                       %b_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%acc_a = %init, %acc_b = %init) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %a = tt.load %a_ptr_init {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: tt.load {{.*}} {loop.cluster = 1 : i32, loop.stage = 1 : i32}
    %b = tt.load %a_ptr_init {tt.latency = 1 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res_a = arith.addf %acc_a, %a : tensor<128x32xf16, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res_b = arith.addf %acc_b, %b : tensor<128x32xf16, #A>
    scf.yield %res_a, %res_b : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
  }
  tt.return %loop#0, %loop#1 : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
}

// CHECK-LABEL: @parallel_deps_uneven2
tt.func @parallel_deps_uneven2(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>,
                       %b_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%acc_a = %init, %acc_b = %init) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 1 : i32, loop.stage = 1 : i32}
    %a = tt.load %a_ptr_init {tt.latency = 1 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %b = tt.load %a_ptr_init {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res_a = arith.addf %acc_a, %a : tensor<128x32xf16, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res_b = arith.addf %acc_b, %b : tensor<128x32xf16, #A>
    scf.yield %res_a, %res_b : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
  }
  tt.return %loop#0, %loop#1 : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
}

// CHECK-LABEL: @direct_deps
tt.func @direct_deps(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> tensor<128x32xf16, #A> {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #A>
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init, %a_ptr = %a_ptr_init) -> (tensor<128x32xf16, #A>, tensor<128x32x!tt.ptr<f16>, #A>) {
    // CHECK: tt.addptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %a_ptr_next = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #A>, tensor<128x32xi32, #A>
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %a = tt.load %a_ptr_next {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res = arith.addf %acc, %a : tensor<128x32xf16, #A>
    scf.yield %res, %a_ptr_next : tensor<128x32xf16, #A>, tensor<128x32x!tt.ptr<f16>, #A>
  }
  tt.return %loop#0 : tensor<128x32xf16, #A>
}

// CHECK-LABEL: @dist1_deps
tt.func @dist1_deps(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> tensor<128x32xf16, #A> {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #A>
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init, %a_ptr = %a_ptr_init) -> (tensor<128x32xf16, #A>, tensor<128x32x!tt.ptr<f16>, #A>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %a = tt.load %a_ptr {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res = arith.addf %acc, %a : tensor<128x32xf16, #A>
    // CHECK: tt.addptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    %a_ptr_next = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #A>, tensor<128x32xi32, #A>
    scf.yield %res, %a_ptr_next : tensor<128x32xf16, #A>, tensor<128x32x!tt.ptr<f16>, #A>
  }
  tt.return %loop#0 : tensor<128x32xf16, #A>
}

// CHECK-LABEL: @prologue_if
tt.func @prologue_if(%lb : index, %ub : index, %step : index, %cnd : i1,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> tensor<128x32xf16, #A> {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #A>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (tensor<128x32xf16, #A>) {
    // CHECK: scf.if
    // CHECK: {loop.cluster = 0 : i32, loop.stage = 0 : i32}
    %a_ptr = scf.if %cnd -> tensor<128x32x!tt.ptr<f16>, #A> {
      %a_ptr_ret = tt.addptr %a_ptr_init, %a_off : tensor<128x32x!tt.ptr<f16>, #A>, tensor<128x32xi32, #A>
      scf.yield %a_ptr_ret : tensor<128x32x!tt.ptr<f16>, #A>
    } else {
      scf.yield %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>
    }
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %a = tt.load %a_ptr {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %res = arith.addf %acc, %a : tensor<128x32xf16, #A>
    scf.yield %res : tensor<128x32xf16, #A>
  }
  tt.return %loop#0 : tensor<128x32xf16, #A>
}

// CHECK-LABEL: @independent_epilogue_if
tt.func @independent_epilogue_if(%lb : index, %ub : index, %step : index, %cnd : i1,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> tensor<128x32xf16, #A> {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #A>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (tensor<128x32xf16, #A>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %a = tt.load %a_ptr_init {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res = arith.addf %acc, %a : tensor<128x32xf16, #A>
    // CHECK: scf.if
    // CHECK: {loop.cluster = 4 : i32, loop.stage = 2 : i32}
    scf.if %cnd {
      tt.store %a_ptr_init, %init : tensor<128x32x!tt.ptr<f16>, #A>
    }
    scf.yield %res : tensor<128x32xf16, #A>
  }
  tt.return %loop#0 : tensor<128x32xf16, #A>
}

// CHECK-LABEL: @independent_last_stage
tt.func @independent_last_stage(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init, %acc2 = %init) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %a = tt.load %a_ptr_init {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res = arith.addf %acc, %a : tensor<128x32xf16, #A>
    // CHECK: arith.addf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %res2 = arith.addf %acc2, %init : tensor<128x32xf16, #A>
    scf.yield %res, %res2 : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
  }
  tt.return %loop#0, %loop#1 : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
}

// CHECK-LABEL: @basic_pipeline
tt.func @basic_pipeline(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL>,
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL>) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %a_ = tt.load %a_ptr {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #AL>
    // CHECK: ttg.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %b_ = tt.load %b_ptr {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #BL>
    // CHECK: ttg.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    // CHECK: tt.dot {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    // CHECK: tt.addptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    // CHECK: tt.addptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @unpipelined_load
tt.func @unpipelined_load(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL>,
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL>) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %a_ = tt.load %a_ptr {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #AL>
    // CHECK: ttg.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // load below should be in the same stage as tt.dot (not pipelined)
    // CHECK: tt.load {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    // CHECK: ttg.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    // CHECK: tt.dot {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    // CHECK: tt.addptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    // addptr below should be scheduled to the last stage
    // CHECK: tt.addptr {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @epilogue_if
tt.func @epilogue_if(%lb : index, %ub : index, %step : index, %cnd : i1,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL>,
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL>,
                  %c_ptr_store : tensor<128x128x!tt.ptr<f32>, #C>) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %a_ = tt.load %a_ptr {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #AL>
    // CHECK: ttg.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %b_ = tt.load %b_ptr {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #BL>
    // CHECK: ttg.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    // CHECK: tt.dot {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    // CHECK: scf.if
    // CHECK: {loop.cluster = 4 : i32, loop.stage = 2 : i32}
    scf.if %cnd {
      tt.store %c_ptr_store, %c : tensor<128x128x!tt.ptr<f32>, #C>
    }
    // CHECK: tt.addptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    // CHECK: tt.addptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @intermediate_use
tt.func @intermediate_use(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL>,
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL>) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>
  %c2 = arith.constant dense<2.00> : tensor<32x128xf16, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %a_ = tt.load %a_ptr {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #AL>
    // CHECK: ttg.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %b_ = tt.load %b_ptr {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #BL>
    // CHECK: arith.mulf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %b_2 = arith.mulf %b_ , %c2 : tensor<32x128xf16, #BL>
    // CHECK: ttg.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %b = ttg.convert_layout %b_2 : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    // CHECK: tt.dot {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    // CHECK: tt.addptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    // CHECK: tt.addptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @indirect_load
tt.func @indirect_load(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL>,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL>,
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL>) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:4 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32}
    %a_off = tt.load %a_ind_ptr {tt.latency = 1 : i32} : tensor<128x32x!tt.ptr<i32>, #AL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    // CHECK: tt.addptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    // addptr below scheduled by scheduleDependencies to the same stage as tt.load that is using it
    // CHECK: tt.addptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    %a_ = tt.load %next_a_ptr {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #AL>
    // CHECK: ttg.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    %b_ = tt.load %next_b_ptr {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #BL>
    // CHECK: ttg.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    // CHECK: tt.dot {{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#3: tensor<128x128xf32, #C>
}

// Verify that we don't schedule/pipeline loops with gpu.barrier
// CHECK-LABEL: @gpu_barrier
tt.func @gpu_barrier(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> tensor<128x32xf16, #A> {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (tensor<128x32xf16, #A>) {
    // CHECK-NOT: loop.cluster
    %a = tt.load %a_ptr_init {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    %res = arith.addf %acc, %a : tensor<128x32xf16, #A>
    gpu.barrier
    scf.yield %res : tensor<128x32xf16, #A>
  }
  tt.return %loop#0 : tensor<128x32xf16, #A>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma
tt.func @tc_gen5_mma(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B: tensor<128x128xf16, #blocked1>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %A = tt.load %A_ptr {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[], %true, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    // CHECK: ttng.tmem_load {{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
    %c, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked1>
    // CHECK: "use"{{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
    "use"(%c) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_if_user
tt.func @tc_gen5_mma_if_user(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B: tensor<128x128xf16, #blocked1>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>,
                  %cnd: i1) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %A = tt.load %A_ptr {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[], %true, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    scf.if %cnd {
      %c, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked1>
      "use"(%c) : (tensor<128x128xf32, #blocked1>) -> ()
    }
    // CHECK: scf.if
    // CHECK: tmem_load
    // CHECK: "use"{{.*}}
    // CHECK-NOT: loop.cluster
    // CHECK: } {loop.cluster = 4 : i32, loop.stage = 3 : i32}
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_scaled
tt.func @tc_gen5_mma_scaled(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B: tensor<128x128xf16, #blocked1>,
                  %A_sc_sh: !ttg.memdesc<1x2x32x4x4xi8, #shared1, #ttg.shared_memory>,
                  %B_sc_sh: !ttg.memdesc<1x2x32x4x4xi8, #shared1, #ttg.shared_memory>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %A = tt.load %A_ptr {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %acc_tm[], %A_sc_sh, %B_sc_sh, %true, %true lhs = e5m2 rhs = e5m2 {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #ttg.shared_memory>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #ttg.shared_memory>
    // CHECK: ttng.tmem_load {{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
    %c, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked1>
    // CHECK: "use"{{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
    "use"(%c) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @select_after_mma
  tt.func public @select_after_mma(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = "cnd"() : () -> i1
    %1, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %1[%acc_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      %4 = tt.load %arg0 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = ttg.local_alloc %4 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = tt.load %arg1 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %7 = ttg.local_alloc %6 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %mma_tok = ttng.tc_gen5_mma %5, %7, %1[%tok], %true, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: arith.xori {{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
      %8 = arith.xori %0, %true : i1
      // CHECK: ttng.tmem_store {{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
      %store_tok = ttng.tmem_store %cst_0, %1[%mma_tok], %8 : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %store_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32}
    %2, %res_tok = ttng.tmem_load %1[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %3 = arith.truncf %2 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %3 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @select_before_mma
  tt.func public @select_before_mma(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = "cnd"() : () -> i1
    %1, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %1[%acc_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      // CHECK: arith.xori {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %8 = arith.xori %0, %true : i1
      // CHECK: ttng.tmem_store {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %store_tok = ttng.tmem_store %cst_0, %1[%tok], %8 : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %4 = tt.load %arg0 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = ttg.local_alloc %4 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = tt.load %arg1 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %7 = ttg.local_alloc %6 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32}
      %mma_tok = ttng.tc_gen5_mma %5, %7, %1[%store_tok], %true, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %mma_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32}
    %2, %res_tok = ttng.tmem_load %1[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %3 = arith.truncf %2 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %3 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @two_dots
  tt.func public @two_dots(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: tensor<128x128x!tt.ptr<f32>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg4: i32) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0, %acc_tok0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %1, %acc_tok1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %last_tok:2 = scf.for %arg5 = %c0_i32 to %arg4 step %c1_i32 iter_args(%tok0 = %acc_tok0, %tok1 = %acc_tok1) -> (!ttg.async.token, !ttg.async.token) : i32 {
      // CHECK: tt.load {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32}
      %2 = tt.load %arg0 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
      %3 = ttg.local_alloc %2 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: tt.load {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32}
      %4 = tt.load %arg1 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
      %5 = ttg.local_alloc %4 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
      %6 = tt.load %arg2 : tensor<128x128x!tt.ptr<f32>, #blocked>
      // CHECK: ttng.tmem_store {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
      %store_tok0 = ttng.tmem_store %6, %0[%tok0], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32}
      %mma_tok0 = ttng.tc_gen5_mma %3, %5, %0[%store_tok0], %true, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_load {{.*}} {loop.cluster = 1 : i32, loop.stage = 3 : i32}
      %7, %load_tok0 = ttng.tmem_load %0[%mma_tok0] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: ttng.tmem_store {{.*}} {loop.cluster = 1 : i32, loop.stage = 3 : i32}
      %store_tok1 = ttng.tmem_store %7, %1[%tok1], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 1 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32}
      %mma_tok1 = ttng.tc_gen5_mma %3, %5, %1[%store_tok1], %true, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_load {{.*}} {loop.cluster = 0 : i32, loop.stage = 4 : i32}
      %8, %load_tok1 = ttng.tmem_load %1[%mma_tok1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: tt.store {{.*}} {loop.cluster = 0 : i32, loop.stage = 4 : i32}
      tt.store %arg3, %8 : tensor<128x128x!tt.ptr<f32>, #blocked>
      scf.yield %load_tok0, %load_tok1 : !ttg.async.token, !ttg.async.token
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma
tt.func @tc_gen5_mma(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B: tensor<128x128xf16, #blocked>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %A = tt.load %A_ptr {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
    // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[], %true, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    // CHECK: ttng.tmem_load {{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
    %c, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    // CHECK: "use"{{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
    "use"(%c) : (tensor<128x128xf32, #blocked>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_if_user
tt.func @tc_gen5_mma_if_user(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B: tensor<128x128xf16, #blocked>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>,
                  %cnd: i1) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %A = tt.load %A_ptr {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
    // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[], %true, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    scf.if %cnd {
      %c, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      "use"(%c) : (tensor<128x128xf32, #blocked>) -> ()
    }
    // CHECK: scf.if
    // CHECK: tmem_load
    // CHECK: "use"{{.*}}
    // CHECK-NOT: loop.cluster
    // CHECK: } {loop.cluster = 4 : i32, loop.stage = 3 : i32}
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_scaled
tt.func @tc_gen5_mma_scaled(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B: tensor<128x128xf16, #blocked>,
                  %A_sc_sh: !ttg.memdesc<1x2x32x4x4xi8, #shared1, #ttg.shared_memory>,
                  %B_sc_sh: !ttg.memdesc<1x2x32x4x4xi8, #shared1, #ttg.shared_memory>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %A = tt.load %A_ptr {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
    // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %acc_tm[], %A_sc_sh, %B_sc_sh, %true, %true lhs = e5m2 rhs = e5m2 {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #ttg.shared_memory>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #ttg.shared_memory>
    // CHECK: ttng.tmem_load {{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
    %c, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    // CHECK: "use"{{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
    "use"(%c) : (tensor<128x128xf32, #blocked>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @select_after_mma
  tt.func public @select_after_mma(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = "cnd"() : () -> i1
    %1, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %1[%acc_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      %4 = tt.load %arg0 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = ttg.local_alloc %4 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = tt.load %arg1 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %7 = ttg.local_alloc %6 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %mma_tok = ttng.tc_gen5_mma %5, %7, %1[%tok], %true, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: arith.xori {{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
      %8 = arith.xori %0, %true : i1
      // CHECK: ttng.tmem_store {{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
      %store_tok = ttng.tmem_store %cst_0, %1[%mma_tok], %8 : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %store_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32}
    %2, %res_tok = ttng.tmem_load %1[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %3 = arith.truncf %2 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %3 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @select_before_mma
  tt.func public @select_before_mma(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = "cnd"() : () -> i1
    %1, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %1[%acc_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      // CHECK: arith.xori {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %8 = arith.xori %0, %true : i1
      // CHECK: ttng.tmem_store {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %store_tok = ttng.tmem_store %cst_0, %1[%tok], %8 : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %4 = tt.load %arg0 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = ttg.local_alloc %4 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = tt.load %arg1 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %7 = ttg.local_alloc %6 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32}
      %mma_tok = ttng.tc_gen5_mma %5, %7, %1[%store_tok], %true, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %mma_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32}
    %2, %res_tok = ttng.tmem_load %1[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %3 = arith.truncf %2 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %3 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @two_dots
  tt.func public @two_dots(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: tensor<128x128x!tt.ptr<f32>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg4: i32) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0, %acc_tok0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %1, %acc_tok1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %last_tok:2 = scf.for %arg5 = %c0_i32 to %arg4 step %c1_i32 iter_args(%tok0 = %acc_tok0, %tok1 = %acc_tok1) -> (!ttg.async.token, !ttg.async.token) : i32 {
      // CHECK: tt.load {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32}
      %2 = tt.load %arg0 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
      %3 = ttg.local_alloc %2 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: tt.load {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32}
      %4 = tt.load %arg1 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
      %5 = ttg.local_alloc %4 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
      %6 = tt.load %arg2 : tensor<128x128x!tt.ptr<f32>, #blocked>
      // CHECK: ttng.tmem_store {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
      %store_tok0 = ttng.tmem_store %6, %0[%tok0], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32}
      %mma_tok0 = ttng.tc_gen5_mma %3, %5, %0[%store_tok0], %true, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_load {{.*}} {loop.cluster = 1 : i32, loop.stage = 3 : i32}
      %7, %load_tok0 = ttng.tmem_load %0[%mma_tok0] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: ttng.tmem_store {{.*}} {loop.cluster = 1 : i32, loop.stage = 3 : i32}
      %store_tok1 = ttng.tmem_store %7, %1[%tok1], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 1 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32}
      %mma_tok1 = ttng.tc_gen5_mma %3, %5, %1[%store_tok1], %true, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_load {{.*}} {loop.cluster = 0 : i32, loop.stage = 4 : i32}
      %8, %load_tok1 = ttng.tmem_load %1[%mma_tok1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: tt.store {{.*}} {loop.cluster = 0 : i32, loop.stage = 4 : i32}
      tt.store %arg3, %8 : tensor<128x128x!tt.ptr<f32>, #blocked>
      scf.yield %load_tok0, %load_tok1 : !ttg.async.token, !ttg.async.token
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @changed_acc_before_mma
  tt.func public @changed_acc_before_mma(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %0[%acc_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
      %3 = tt.load %arg0 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %4 = ttg.local_alloc %3 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
      %5 = tt.load %arg1 {tt.latency = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      // CHECK: ttg.local_alloc {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %6 = ttg.local_alloc %5 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttng.tmem_load {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %7, %load_tok = ttng.tmem_load %0[%tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      // CHECK: arith.mulf {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %8 = arith.mulf %7, %cst_0 : tensor<128x128xf32, #blocked1>
      // CHECK: ttng.tmem_store {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %store_tok = ttng.tmem_store %8, %0[%load_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %mma_tok = ttng.tc_gen5_mma %4, %6, %0[%store_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %mma_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32}
    %1, %res_tok = ttng.tmem_load %0[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %2 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @backwards_prop_existing
tt.func public @backwards_prop_existing(%arg0: i32, %arg1: tensor<128x4x!tt.ptr<i8>, #blocked>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  scf.for %arg2 = %c0_i32 to %arg0 step %c1_i32  : i32 {
    %0 = tt.load %arg1 {loop.cluster = 2 : i32, loop.stage = 3 : i32} : tensor<128x4x!tt.ptr<i8>, #blocked>
    %1 = ttg.local_alloc %0 : (tensor<128x4xi8, #blocked>) -> !ttg.memdesc<128x4xi8, #shared, #smem>
    // CHECK: ttg.local_load %{{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32}
    %2 = ttg.local_load %1 : !ttg.memdesc<128x4xi8, #shared, #smem> -> tensor<128x4xi8, #linear>
    %result = ttng.tmem_alloc %2 {loop.cluster = 2 : i32, loop.stage = 3 : i32} : (tensor<128x4xi8, #linear>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
    "use"(%result) {loop.cluster = 2 : i32, loop.stage = 3 : i32} : (!ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>) -> ()
  } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize}
  tt.return
}

}
