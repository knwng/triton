# Backends and Modes

Proton automatically selects the profiling backend that matches the active
Triton runtime when `backend=None`.

## Backends

| Backend | Platform | Notes |
| --- | --- | --- |
| `cupti` | NVIDIA GPUs | Default NVIDIA backend. Supports regular profiling, `pcsampling`, and `periodic_flushing`. |
| `rocprofiler` | AMD GPUs | Preferred AMD backend when rocprofiler-sdk is available. Supports regular profiling, `pcsampling`, and `periodic_flushing`. |
| `roctracer` | AMD GPUs | **Deprecated** AMD fallback backend. Supports regular profiling and `periodic_flushing`. |
| `instrumentation` | NVIDIA and AMD GPUs | Intra-kernel instrumentation backend for scope-level cycle metrics inside kernels. |

Examples:

```python
proton.start("nvidia_profile", backend="cupti")
proton.start("amd_profile", backend="rocprofiler")
proton.start("instrumented_profile", backend="instrumentation")
```

On AMD GPUs, `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are not supported
by Proton. Use `ROCR_VISIBLE_DEVICES` instead so profiler device IDs can be
mapped correctly.

## Instruction Sampling

Instruction sampling is available through CUPTI on NVIDIA GPUs and
rocprofiler-sdk on AMD GPUs:

```python
proton.start(
    "profile_name",
    context="shadow",
    backend="cupti",
    mode="pcsampling",
)
```

Proton currently uses the CUPTI backend's default sampling period; the sampling
interval is not configurable through this mode.

On AMD, source-line attribution requires both code-object address translation
support in the build's rocprofiler-sdk package and usable DWARF line
information in the sampled code object. Without either, Proton reports the
samples at kernel level.

AMD PC sampling uses `PROTON_PC_SAMPLING_INTERVAL` as an optional positive
integer sampling interval. The default is `131072`, and rocprofiler-sdk clamps
the requested value to the supported range reported by the GPU. Smaller values
can produce more samples and overhead; larger values reduce both.

To retain individual AMD samples instead of only the aggregated Hatchet
metrics, pass a raw JSON Lines output path in the mode string:

```python
proton.start(
    "profile_name",
    backend="rocprofiler",
    mode="pcsampling:raw_output=/tmp/profile.pc_sampling.jsonl",
)
```

The optional file contains one `sample` record per hardware sample and one
`pc_info` record per unique dispatch and sampled PC. Sample records preserve
the logical workgroup and wave, physical shader-engine/WGP/SIMD/wave-slot,
timestamp, PC, issue state, stochastic stall reason, arbiter state, and memory
counters when rocprofiler-sdk marks those fields as available. `pc_info`
records map the dispatch and PC to the disassembled instruction, kernel, and
source location. If rocprofiler-sdk reports buffer loss, a `dropped_samples`
record preserves the number of missing samples. This mode can generate large
files and adds callback-thread I/O overhead.

To generate a conventional Proton Chrome trace augmented with every raw AMD PC
sample, use trace data and enable the `chrome_trace` PC-sampling option:

```python
proton.start(
    "profile_name",
    data="trace",
    backend="rocprofiler",
    mode="pcsampling:format=chrome_trace",
)
```

This writes both `profile_name.pc_sampling.jsonl` and
`profile_name.chrome_trace`. The Chrome trace retains Proton's CPU scopes,
kernel intervals, and launch flows, then adds one instant event per hardware
sample on dispatch lanes, with a separate thread lane for each logical
`(workgroup, wave_in_group)` pair. This distinction is important because the
same wave index is reused by every workgroup. By default the trace also adds
sampled-state counters in 20 microsecond bins. Set `counter_bin_us=0` to omit
counters or use another non-negative width, for example:

```python
mode="pcsampling:format=chrome_trace:counter_bin_us=10"
```

The integrated Chrome-trace path is raw-only: Proton does not build aggregated
`PCSamplingMetric` nodes while collecting it. This avoids retaining the same
samples in both the raw and aggregated forms. Legacy `mode="pcsampling"`
continues to aggregate into Hatchet for compatibility. An explicit raw-only
capture without Chrome-trace merging can use:

```python
mode="pcsampling:raw_output=/tmp/profile.pc_sampling.jsonl:aggregate=false"
```

When timestamp-offset metadata is available, PC sample instants are aligned to
the same clock domain as the conventional kernel events. PC samples remain
statistical observations rather than instruction-duration intervals.

By default, Proton prefers stochastic sampling and falls back to host-trap
sampling. Set `PROTON_ROCPROFILER_PC_SAMPLING_METHOD` to `stochastic` or
`host-trap` to require a specific method. Profiling fails to start if the
selected method is invalid or unavailable on the visible GPU.

Proton enables rocprofiler-sdk's PC-sampling feature during backend
configuration because the SDK locks configuration before profiling sessions are
started. If `ROCPROFILER_PC_SAMPLING_BETA_ENABLED` is already set, Proton
preserves the user-provided value.

Instruction sampling can add significant end-to-end overhead because Proton
transfers and processes sample data on the CPU. Viewer filters such as
`-i <regex>`, `-d <depth>`, and `-t <threshold>` are useful for narrowing the
output.

## Periodic Flushing

`periodic_flushing` splits long profiling sessions into phases and writes
completed phases while the session is still running. It is supported by
`cupti`, `rocprofiler`, and `roctracer`.

See [periodic profiling](periodic-profiling.md) for phase advancement, output
file naming, in-memory phase APIs, and tuning guidance.

## Instrumentation Backend

The instrumentation backend collects fine-grained intra-kernel measurements.
By default it records cycle metrics for each profiled unit.

```python
import triton.profiler as proton
import triton.profiler.mode as pmode

proton.start(
    "profile_name",
    backend="instrumentation",
    mode=pmode.Default(),
)
```

The string form is also accepted:

```python
proton.start(
    "profile_name",
    backend="instrumentation",
    mode="default:buffer_type=global:buffer_size=16384",
)
```

Instrumentation mode options include:

| Option | Values |
| --- | --- |
| `metric_type` | `cycle` |
| `sampling_strategy` | `none`, `selective` |
| `sampling_options` | Comma-separated unit IDs, such as `0,1,2,3` |
| `granularity` | `cta`, `warp`, `warp_2`, `warp_4`, `warp_8`, `warp_group`, `warp_group_2`, `warp_group_4`, `warp_group_8` |
| `buffer_strategy` | `circular`, `flush` |
| `buffer_type` | `shared`, `global` |
| `buffer_size` | Integer byte count; `0` selects the backend default. |
| `optimizations` | Comma-separated `time_shift`, `sched_stores`, `sched_barriers`, `clock32` |

Mode object example:

```python
mode = pmode.Default(
    sampling_strategy="selective",
    sampling_options="0,1,2,3",
    buffer_type="global",
    optimizations="clock32,time_shift",
)
proton.start("profile_name", backend="instrumentation", mode=mode)
```

See [intra-kernel profiling](intra-kernel.md) for end-to-end examples.
