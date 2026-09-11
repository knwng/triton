from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RAW_SCHEMA = {"type": "schema", "schema": "proton-amd-pc-sampling", "version": 1}


@dataclass(frozen=True)
class PCSamplingTraceConfig:
    raw_path: Path
    trace_path: Path
    counter_bin_us: float


def _parse_bool_option(name: str, value: str) -> bool:
    lower = value.lower()
    if lower in ("true", "1"):
        return True
    if lower in ("false", "0"):
        return False
    raise ValueError(f"pcsampling {name} must be true or false")


def prepare_pc_sampling_trace(
    name: str,
    data: str | None,
    backend: str,
    mode: str,
) -> tuple[str, PCSamplingTraceConfig | None]:
    """Translate the Python-only Chrome-trace format into native raw capture.

    The native profiler writes the per-sample JSONL stream. Python merges that
    stream into TraceData's conventional Chrome trace after native finalize.
    """
    options = mode.split(":") if mode else []
    if not options or options[0].lower() != "pcsampling":
        return mode, None

    output_format: str | None = None
    counter_bin_us = 20.0
    counter_bin_seen = False
    raw_output: str | None = None
    aggregate: bool | None = None
    native_options = [options[0]]

    for option in options[1:]:
        key, separator, value = option.partition("=")
        lower_key = key.lower()
        if lower_key == "format":
            if not separator or output_format is not None:
                raise ValueError("pcsampling format must be specified once as key=value")
            output_format = value.lower()
            if output_format != "chrome_trace":
                raise ValueError("pcsampling format currently supports only chrome_trace")
            continue
        if lower_key == "chrome_trace":
            raise ValueError("use pcsampling format=chrome_trace instead of chrome_trace=true")
        if lower_key == "counter_bin_us":
            if not separator or counter_bin_seen:
                raise ValueError("pcsampling counter_bin_us must be specified once as key=value")
            counter_bin_seen = True
            try:
                counter_bin_us = float(value)
            except ValueError as error:
                raise ValueError("pcsampling counter_bin_us must be a number") from error
            if not math.isfinite(counter_bin_us) or counter_bin_us < 0:
                raise ValueError("pcsampling counter_bin_us must be finite and non-negative")
            continue
        if lower_key == "raw_output" and separator:
            if raw_output is not None:
                raise ValueError("duplicate pcsampling raw_output option")
            if not value:
                raise ValueError("pcsampling raw_output cannot be empty")
            raw_output = value
        elif lower_key == "aggregate" and separator:
            if aggregate is not None:
                raise ValueError("duplicate pcsampling aggregate option")
            aggregate = _parse_bool_option("aggregate", value)
        native_options.append(option)

    if output_format is None:
        if counter_bin_seen:
            raise ValueError("pcsampling counter_bin_us requires format=chrome_trace")
        return ":".join(native_options), None
    if backend.lower() != "rocprofiler":
        raise ValueError("pcsampling format=chrome_trace is currently supported only by the AMD rocprofiler backend")
    if data is None or data.lower() != "trace":
        raise ValueError("pcsampling format=chrome_trace requires data='trace'")
    if not name or name == "-":
        raise ValueError("pcsampling format=chrome_trace requires a file output name")
    if aggregate is True:
        raise ValueError("pcsampling format=chrome_trace is raw-only and cannot be combined with aggregate=true")

    if raw_output is None:
        raw_output = f"{name}.pc_sampling.jsonl"
        native_options.append(f"raw_output={raw_output}")
    if aggregate is None:
        native_options.append("aggregate=false")

    trace_path = Path(f"{name}.chrome_trace")
    raw_path = Path(raw_output)
    if os.path.abspath(raw_path) == os.path.abspath(trace_path):
        raise ValueError("pcsampling raw_output must differ from the Chrome trace output")

    return ":".join(native_options), PCSamplingTraceConfig(
        raw_path=raw_path,
        trace_path=trace_path,
        counter_bin_us=counter_bin_us,
    )


def _records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise TypeError(f"expected a JSON object at {path}:{line_number}")
            yield line_number, record


def _pc_key(record: dict[str, Any]) -> tuple[int, int, int]:
    try:
        return (
            int(record["dispatch_id"]),
            int(record["code_object_id"]),
            int(record["pc_offset"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"record has an invalid PC identity: {record}") from error


def _sample_state(sample: dict[str, Any]) -> str:
    if sample.get("wave_issued") is True:
        # reason_not_issued is undefined for an issued wave and can contain
        # stale non-zero hardware bits.
        return "issued"
    if sample.get("wave_issued") is False:
        reason = sample.get("reason_not_issued")
        if isinstance(reason, str) and reason != "none":
            return reason
        return "not_issued_unknown"
    return "unknown"


def _instruction_mnemonic(instruction: Any) -> str:
    if not isinstance(instruction, str) or not instruction:
        return "unknown_pc"
    return instruction.split(None, 1)[0]


def _compact_nonzero_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result = {key: item for key, item in value.items() if item not in (0, False, None)}
    return result or None


def _metadata_event(name: str, pid: int, tid: int, args: dict[str, Any]) -> dict[str, Any]:
    return {"ph": "M", "pid": pid, "tid": tid, "name": name, "args": args}


def _logical_wave_key(sample: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """Identify a logical wave by workgroup coordinates and wave index."""
    wave = sample.get("wave_in_group")
    if not isinstance(wave, int):
        return None
    workgroup = sample.get("workgroup_id")
    if (isinstance(workgroup, list) and len(workgroup) == 3
            and all(isinstance(component, int) for component in workgroup)):
        return workgroup[0], workgroup[1], workgroup[2], wave
    return -1, -1, -1, wave


def _logical_wave_name(key: tuple[int, int, int, int]) -> str:
    x, y, z, wave = key
    if (x, y, z) == (-1, -1, -1):
        return f"wave {wave} (unknown workgroup)"
    return f"workgroup ({x}, {y}, {z}) / wave {wave}"


def _sample_event(
    sample: dict[str, Any],
    info: dict[str, Any],
    state: str,
    timestamp_us: float,
    lane: int,
) -> dict[str, Any]:
    dispatch_id, code_object_id, pc_offset = _pc_key(sample)
    wave = int(sample["wave_in_group"])
    instruction = info.get("instruction")
    source_file = info.get("source_file")
    source_line = info.get("source_line")
    hardware = sample.get("hardware")

    args: dict[str, Any] = {
        "asm": instruction,
        "pc": f"code_object={code_object_id} offset=0x{pc_offset:x}",
        "sampling_method": sample.get("method"),
        "workgroup": sample.get("workgroup_id"),
        "wave_in_group": wave,
        "wave_count_on_cu": sample.get("wave_count"),
        "wave_issued": sample.get("wave_issued"),
        "instruction_type": sample.get("instruction_type"),
    }
    if sample.get("wave_issued") is False:
        args["reason_not_issued"] = state
    for field in (
            "correlation_id_internal",
            "correlation_id_external",
            "sampling_lock_error",
            "dual_issue_valu",
    ):
        if sample.get(field) is not None:
            args[field] = sample[field]
    if source_file is not None:
        args["source"] = f"{source_file}:{source_line}" if source_line is not None else source_file
    if isinstance(hardware, dict):
        args["hardware"] = hardware
        chiplet = hardware.get("chiplet")
        chiplet_prefix = f"chiplet{chiplet}/" if chiplet is not None else ""
        args["physical_location"] = (
            f"{chiplet_prefix}SE{hardware.get('shader_engine_id')}/SA{hardware.get('shader_array_id')}"
            f"/WGP{hardware.get('cu_or_wgp_id')}/SIMD{hardware.get('simd_id')}"
            f"/slot{hardware.get('wave_slot')}")
    exec_mask = sample.get("exec_mask")
    if isinstance(exec_mask, int):
        args["exec_mask"] = f"0x{exec_mask:x}"
    memory_counters = _compact_nonzero_mapping(sample.get("memory_counters"))
    if memory_counters:
        args["nonzero_memory_counters"] = memory_counters
    arb_state = _compact_nonzero_mapping(sample.get("arb_state"))
    if arb_state:
        args["nonzero_arb_state"] = arb_state

    return {
        "name": f"{state}: {_instruction_mnemonic(instruction)}",
        "cat": f"pcsampling,{state}",
        "ph": "i",
        "s": "t",
        "pid": dispatch_id,
        "tid": lane,
        "ts": timestamp_us,
        "args": args,
    }


class _StreamingTraceWriter:

    def __init__(self, output: Any, top_level: dict[str, Any], existing_events: list[Any]):
        self.output = output
        self.first_field = True
        self.first_event = True
        self.output.write("{")
        for key, value in top_level.items():
            if key in ("traceEvents", "pcSamplingMetadata"):
                continue
            self._field_prefix()
            json.dump(key, self.output)
            self.output.write(":")
            json.dump(value, self.output, separators=(",", ":"), ensure_ascii=False)
        self._field_prefix()
        self.output.write('"traceEvents":[')
        for event in existing_events:
            self.event(event)

    def _field_prefix(self) -> None:
        if not self.first_field:
            self.output.write(",")
        self.first_field = False

    def event(self, event: dict[str, Any]) -> None:
        if not self.first_event:
            self.output.write(",")
        json.dump(event, self.output, separators=(",", ":"), ensure_ascii=False)
        self.first_event = False

    def finish(self, metadata: dict[str, Any]) -> None:
        self.output.write('],"pcSamplingMetadata":')
        json.dump(metadata, self.output, separators=(",", ":"), ensure_ascii=False)
        self.output.write("}\n")


def merge_pc_sampling_chrome_trace(config: PCSamplingTraceConfig) -> dict[str, Any]:
    """Merge all raw AMD PC samples into a conventional Proton Chrome trace."""
    if not config.raw_path.is_file():
        raise RuntimeError(f"PC-sampling raw output does not exist: {config.raw_path}")
    if not config.trace_path.is_file():
        raise RuntimeError(f"Proton Chrome trace does not exist: {config.trace_path}")

    with config.trace_path.open(encoding="utf-8") as trace_file:
        trace = json.load(trace_file)
    if not isinstance(trace, dict) or not isinstance(trace.get("traceEvents"), list):
        raise TypeError(f"invalid Proton Chrome trace: {config.trace_path}")

    schema_count = 0
    clock_offsets: set[int] = set()
    pc_info: dict[tuple[int, int, int], dict[str, Any]] = {}
    kernels_by_dispatch: dict[int, str] = {}
    logical_waves_by_dispatch: dict[int, set[tuple[int, int, int, int]]] = defaultdict(set)
    sample_count = 0
    dropped_sample_count = 0
    min_sample_timestamp: int | None = None
    max_sample_timestamp: int | None = None

    for line_number, record in _records(config.raw_path):
        record_type = record.get("type")
        if record_type == "schema":
            schema_count += 1
            if record != RAW_SCHEMA:
                raise RuntimeError(f"unsupported schema at {config.raw_path}:{line_number}: {record}")
        elif record_type == "clock_info":
            offset = record.get("timestamp_offset_ns")
            if not isinstance(offset, int) or record.get("timestamp_unit") != "ns":
                raise RuntimeError(f"invalid clock_info at {config.raw_path}:{line_number}: {record}")
            clock_offsets.add(offset)
        elif record_type == "pc_info":
            key = _pc_key(record)
            pc_info[key] = record
            kernel_name = record.get("kernel_name")
            if isinstance(kernel_name, str):
                kernels_by_dispatch.setdefault(key[0], kernel_name)
        elif record_type == "sample":
            sample_count += 1
            timestamp = record.get("timestamp")
            dispatch_id = record.get("dispatch_id")
            if isinstance(timestamp, int):
                min_sample_timestamp = (timestamp if min_sample_timestamp is None else min(
                    min_sample_timestamp, timestamp))
                max_sample_timestamp = (timestamp if max_sample_timestamp is None else max(
                    max_sample_timestamp, timestamp))
            logical_wave = _logical_wave_key(record)
            if isinstance(dispatch_id, int) and logical_wave is not None:
                logical_waves_by_dispatch[dispatch_id].add(logical_wave)
        elif record_type == "dropped_samples":
            count = record.get("count")
            if not isinstance(count, int) or count < 0:
                raise RuntimeError(f"invalid dropped_samples at {config.raw_path}:{line_number}: {record}")
            dropped_sample_count += count
        else:
            raise RuntimeError(f"unknown record type at {config.raw_path}:{line_number}: {record_type!r}")

    if schema_count != 1:
        raise RuntimeError(f"expected exactly one raw schema record, found {schema_count}")
    if len(clock_offsets) > 1:
        raise RuntimeError("PC-sampling raw output contains inconsistent timestamp offsets")

    trace_base_ns = trace.get("baseTimeNanoseconds")
    timestamp_offset_ns = next(iter(clock_offsets)) if clock_offsets else None
    if isinstance(trace_base_ns, int) and timestamp_offset_ns is not None:
        timestamp_origin_ns = trace_base_ns - timestamp_offset_ns
        clock_alignment = "rocprofiler_timestamp_offset"
    elif min_sample_timestamp is not None:
        timestamp_origin_ns = min_sample_timestamp
        clock_alignment = "first_pc_sample"
    else:
        timestamp_origin_ns = 0
        clock_alignment = "unavailable"

    existing_events = trace["traceEvents"]
    temporary_path = config.trace_path.with_name(f".{config.trace_path.name}.tmp.{os.getpid()}")
    counter_bins: dict[tuple[int, int, int], Counter[str]] = defaultdict(Counter)
    reason_counts: Counter[str] = Counter()
    samples_by_wave: Counter[tuple[int, int, int, int, int]] = Counter()
    written_samples = 0
    skipped_missing_fields = 0

    wave_lanes: dict[int, dict[tuple[int, int, int, int], int]] = {
        dispatch_id: {logical_wave: lane
                      for lane, logical_wave in enumerate(sorted(logical_waves))}
        for dispatch_id, logical_waves in logical_waves_by_dispatch.items()
    }
    wave_names_by_lane = {(dispatch_id, lane): _logical_wave_name(logical_wave)
                          for dispatch_id, lanes in wave_lanes.items()
                          for logical_wave, lane in lanes.items()}

    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            writer = _StreamingTraceWriter(output_file, trace, existing_events)
            all_dispatches = sorted(wave_lanes)
            for sort_index, dispatch_id in enumerate(all_dispatches, 1):
                kernel_name = kernels_by_dispatch.get(dispatch_id, "<unknown>")
                writer.event(
                    _metadata_event(
                        "process_name",
                        dispatch_id,
                        0,
                        {"name": f"PC sampling dispatch {dispatch_id}: {kernel_name}"},
                    ))
                writer.event(_metadata_event("process_sort_index", dispatch_id, 0, {"sort_index": sort_index}))
                for logical_wave, lane in wave_lanes[dispatch_id].items():
                    writer.event(
                        _metadata_event(
                            "thread_name",
                            dispatch_id,
                            lane,
                            {"name": _logical_wave_name(logical_wave)},
                        ))
                    writer.event(_metadata_event("thread_sort_index", dispatch_id, lane, {"sort_index": lane}))

            for _, sample in _records(config.raw_path):
                if sample.get("type") != "sample":
                    continue
                try:
                    key = _pc_key(sample)
                except RuntimeError:
                    skipped_missing_fields += 1
                    continue
                timestamp = sample.get("timestamp")
                logical_wave = _logical_wave_key(sample)
                if not isinstance(timestamp, int) or logical_wave is None:
                    skipped_missing_fields += 1
                    continue
                dispatch_id = key[0]
                lane = wave_lanes.get(dispatch_id, {}).get(logical_wave)
                if lane is None:
                    skipped_missing_fields += 1
                    continue
                info = pc_info.get(key, {})
                state = _sample_state(sample)
                timestamp_us = (timestamp - timestamp_origin_ns) / 1000.0
                writer.event(_sample_event(sample, info, state, timestamp_us, lane))
                written_samples += 1
                reason_counts[state] += 1
                samples_by_wave[(dispatch_id, *logical_wave)] += 1
                if config.counter_bin_us > 0:
                    bin_index = math.floor(timestamp_us / config.counter_bin_us)
                    counter_bins[(dispatch_id, lane, bin_index)][state] += 1

            counter_events = 0
            if config.counter_bin_us > 0:
                for (dispatch_id, lane, bin_index), counts in sorted(counter_bins.items()):
                    writer.event({
                        "name": (f"{wave_names_by_lane[(dispatch_id, lane)]} sampled states"
                                 f" / {config.counter_bin_us:g} us"),
                        "cat":
                        "pcsampling_counter",
                        "ph":
                        "C",
                        "pid":
                        dispatch_id,
                        "tid":
                        lane,
                        "ts":
                        bin_index * config.counter_bin_us,
                        "args": {state: counts.get(state, 0)
                                 for state in sorted(reason_counts)},
                    })
                    counter_events += 1

            metadata = {
                "raw_source":
                str(config.raw_path),
                "schema":
                RAW_SCHEMA["schema"],
                "schema_version":
                RAW_SCHEMA["version"],
                "clock_alignment":
                clock_alignment,
                "timestamp_offset_ns":
                timestamp_offset_ns,
                "min_sample_timestamp":
                min_sample_timestamp,
                "max_sample_timestamp":
                max_sample_timestamp,
                "raw_samples":
                sample_count,
                "dropped_samples":
                dropped_sample_count,
                "sample_events":
                written_samples,
                "skipped_samples":
                skipped_missing_fields,
                "counter_events":
                counter_events,
                "sample_states":
                dict(reason_counts.most_common()),
                "warning": ("PC samples are statistical instant events, not continuous instruction intervals. "
                            "Do not interpret spacing between samples as instruction latency. "
                            "reason_not_issued is interpreted only when wave_issued is false."),
            }
            writer.finish(metadata)
        os.replace(temporary_path, config.trace_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "raw_output": str(config.raw_path),
        "chrome_trace": str(config.trace_path),
        "sample_events": written_samples,
        "counter_events": counter_events,
        "sample_states": dict(reason_counts.most_common()),
        "samples_by_wave": {
            f"dispatch {dispatch_id} / {_logical_wave_name((x, y, z, wave))}": count
            for (dispatch_id, x, y, z, wave), count in sorted(samples_by_wave.items())
        },
        "skipped_missing_fields": skipped_missing_fields,
        "clock_alignment": clock_alignment,
    }
