#include "Profiler/RocprofSDK/RocprofSDKPCSampling.h"

#if PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING

#include "Context/Context.h"
#include "Driver/GPU/RocprofApi.h"
#include "Utility/Env.h"

#include "rocprofiler-sdk/agent.h"
#include "rocprofiler-sdk/pc_sampling.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>

namespace proton {

namespace {

constexpr size_t PCSamplingBufferSize = 1024 * 1024;
struct AgentPCSamplingConfig {
  rocprofiler_agent_id_t agentId;
  std::vector<rocprofiler_pc_sampling_configuration_t> configs;
};

rocprofiler_status_t
pcSamplingConfigCallback(const rocprofiler_pc_sampling_configuration_t *configs,
                         size_t numConfigs, void *userData) {
  auto *out =
      static_cast<std::vector<rocprofiler_pc_sampling_configuration_t> *>(
          userData);
  for (size_t i = 0; i < numConfigs; ++i)
    out->push_back(configs[i]);
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t agentQueryCallback(rocprofiler_agent_version_t version,
                                        const void **agents, size_t count,
                                        void *userData) {
  if (version != ROCPROFILER_AGENT_INFO_VERSION_0)
    return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

  auto *out = static_cast<std::vector<AgentPCSamplingConfig> *>(userData);
  auto agentList = reinterpret_cast<const rocprofiler_agent_t *const *>(agents);
  for (size_t i = 0; i < count; ++i) {
    if (agentList[i]->type != ROCPROFILER_AGENT_TYPE_GPU)
      continue;

    AgentPCSamplingConfig entry{agentList[i]->id, {}};
    auto status = rocprofiler::queryPCSamplingAgentConfigurations<false>(
        agentList[i]->id, pcSamplingConfigCallback, &entry.configs);
    if (status == ROCPROFILER_STATUS_SUCCESS && !entry.configs.empty())
      out->push_back(std::move(entry));
  }
  return ROCPROFILER_STATUS_SUCCESS;
}

const rocprofiler_pc_sampling_configuration_t *pickPCSamplingConfig(
    const std::vector<rocprofiler_pc_sampling_configuration_t> &configs,
    std::optional<rocprofiler_pc_sampling_method_t> requestedMethod) {
  if (requestedMethod) {
    auto requested =
        std::find_if(configs.begin(), configs.end(), [&](const auto &cfg) {
          return cfg.method == *requestedMethod;
        });
    return requested == configs.end() ? nullptr : &*requested;
  }

  auto stochastic =
      std::find_if(configs.begin(), configs.end(), [](const auto &cfg) {
        return cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
      });
  if (stochastic != configs.end())
    return &*stochastic;

  auto hostTrap =
      std::find_if(configs.begin(), configs.end(), [](const auto &cfg) {
        return cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
      });
  if (hostTrap != configs.end())
    return &*hostTrap;
  return nullptr;
}

bool parsePCSamplingMethod(
    const std::string &value,
    std::optional<rocprofiler_pc_sampling_method_t> &method) {
  if (value.empty()) {
    method = std::nullopt;
    return true;
  }
  if (value == "stochastic") {
    method = ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
    return true;
  }
  if (value == "host-trap") {
    method = ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
    return true;
  }
  return false;
}

PCSamplingMetric::PCSamplingMetricKind mapNotIssuedReasonToStallMetric(
    rocprofiler_pc_sampling_instruction_not_issued_reason_t reason) {
  constexpr std::array ReasonToMetric = {
      PCSamplingMetric::StalledMisc,
      PCSamplingMetric::StalledNoInstruction,
      PCSamplingMetric::StalledALUDependency,
      PCSamplingMetric::StalledWaitcnt,
      PCSamplingMetric::StalledInternalInstruction,
      PCSamplingMetric::StalledBarrier,
      PCSamplingMetric::StalledNotSelected,
      PCSamplingMetric::StalledArbiterWinExStall,
      PCSamplingMetric::StalledOtherWait,
      PCSamplingMetric::StalledSleeping,
  };

  auto index = static_cast<int64_t>(reason);
  if (index < 0 || static_cast<size_t>(index) >= ReasonToMetric.size())
    return PCSamplingMetric::StalledMisc;
  return ReasonToMetric[index];
}

template <typename SampleT>
rocprofiler_pc_t getSamplePC(const SampleT *sample) {
  if (sample->size >= offsetof(SampleT, pc) + sizeof(sample->pc))
    return sample->pc;
  return rocprofiler_pc_t{ROCPROFILER_CODE_OBJECT_ID_NONE, 0};
}

template <typename SampleT>
bool hasSampleField(const SampleT *sample, size_t offset, size_t size) {
  return sample->size >= offset + size;
}

const char *notIssuedReasonName(
    rocprofiler_pc_sampling_instruction_not_issued_reason_t reason) {
  constexpr std::array Names = {
      "none",
      "no_instruction_available",
      "alu_dependency",
      "waitcnt",
      "internal_instruction",
      "barrier_wait",
      "arbiter_not_win",
      "arbiter_win_ex_stall",
      "other_wait",
      "sleep_wait",
  };
  auto index = static_cast<int64_t>(reason);
  if (index < 0 || static_cast<size_t>(index) >= Names.size())
    return "unknown";
  return Names[index];
}

const char *instructionTypeName(
    rocprofiler_pc_sampling_instruction_type_t instructionType) {
  constexpr std::array Names = {
      "none",         "valu",    "matrix",     "scalar",
      "tex",          "lds",     "lds_direct", "flat",
      "export",       "message", "barrier",    "branch_not_taken",
      "branch_taken", "jump",    "other",      "no_instruction",
      "dual_valu",
  };
  auto index = static_cast<int64_t>(instructionType);
  if (index < 0 || static_cast<size_t>(index) >= Names.size())
    return "unknown";
  return Names[index];
}

void writeJSONString(std::ostream &stream, std::string_view value) {
  constexpr char Hex[] = "0123456789abcdef";
  stream << '"';
  for (unsigned char character : value) {
    switch (character) {
    case '"':
      stream << "\\\"";
      break;
    case '\\':
      stream << "\\\\";
      break;
    case '\b':
      stream << "\\b";
      break;
    case '\f':
      stream << "\\f";
      break;
    case '\n':
      stream << "\\n";
      break;
    case '\r':
      stream << "\\r";
      break;
    case '\t':
      stream << "\\t";
      break;
    default:
      if (character < 0x20) {
        stream << "\\u00" << Hex[character >> 4] << Hex[character & 0xf];
      } else {
        stream << character;
      }
    }
  }
  stream << '"';
}

template <typename SampleT>
void appendCommonRawSample(std::ostream &stream, const SampleT *sample,
                           const char *method) {
  auto pc = getSamplePC(sample);
  stream << "{\"type\":\"sample\",\"method\":";
  writeJSONString(stream, method);
  stream << ",\"record_size\":" << sample->size
         << ",\"dispatch_id\":" << sample->dispatch_id
         << ",\"code_object_id\":" << pc.code_object_id
         << ",\"pc_offset\":" << pc.code_object_offset;

  if (hasSampleField(sample, offsetof(SampleT, correlation_id),
                     sizeof(sample->correlation_id))) {
    stream << ",\"correlation_id_internal\":" << sample->correlation_id.internal
           << ",\"correlation_id_external\":"
           << sample->correlation_id.external.value;
  } else {
    stream << ",\"correlation_id_internal\":null"
              ",\"correlation_id_external\":null";
  }

  if (hasSampleField(sample, offsetof(SampleT, timestamp),
                     sizeof(sample->timestamp)))
    stream << ",\"timestamp\":" << sample->timestamp;
  else
    stream << ",\"timestamp\":null";

  if (hasSampleField(sample, offsetof(SampleT, exec_mask),
                     sizeof(sample->exec_mask)))
    stream << ",\"exec_mask\":" << sample->exec_mask;
  else
    stream << ",\"exec_mask\":null";

  if (hasSampleField(sample, offsetof(SampleT, workgroup_id),
                     sizeof(sample->workgroup_id))) {
    stream << ",\"workgroup_id\":[" << sample->workgroup_id.x << ','
           << sample->workgroup_id.y << ',' << sample->workgroup_id.z << ']';
  } else {
    stream << ",\"workgroup_id\":null";
  }

  if (hasSampleField(sample, offsetof(SampleT, hw_id), sizeof(sample->hw_id))) {
    const auto &hw = sample->hw_id;
    stream << ",\"hardware\":{\"chiplet\":" << static_cast<uint64_t>(hw.chiplet)
           << ",\"wave_slot\":" << static_cast<uint64_t>(hw.wave_id)
           << ",\"simd_id\":" << static_cast<uint64_t>(hw.simd_id)
           << ",\"pipe_id\":" << static_cast<uint64_t>(hw.pipe_id)
           << ",\"cu_or_wgp_id\":" << static_cast<uint64_t>(hw.cu_or_wgp_id)
           << ",\"shader_array_id\":"
           << static_cast<uint64_t>(hw.shader_array_id)
           << ",\"shader_engine_id\":"
           << static_cast<uint64_t>(hw.shader_engine_id)
           << ",\"hardware_workgroup_id\":"
           << static_cast<uint64_t>(hw.workgroup_id)
           << ",\"vm_id\":" << static_cast<uint64_t>(hw.vm_id)
           << ",\"queue_id\":" << static_cast<uint64_t>(hw.queue_id)
           << ",\"microengine_id\":" << static_cast<uint64_t>(hw.microengine_id)
           << '}';
  } else {
    stream << ",\"hardware\":null";
  }
}

void appendStochasticRawSample(
    std::ostream &stream,
    const rocprofiler_pc_sampling_record_stochastic_v0_t *sample) {
  using Sample = rocprofiler_pc_sampling_record_stochastic_v0_t;
  appendCommonRawSample(stream, sample, "stochastic");

  if (hasSampleField(sample, offsetof(Sample, wave_in_group),
                     sizeof(sample->wave_in_group)))
    stream << ",\"wave_in_group\":"
           << static_cast<uint32_t>(sample->wave_in_group);
  else
    stream << ",\"wave_in_group\":null";

  // wave_issued and inst_type are bit fields immediately before hw_id and
  // cannot be used with offsetof/sizeof directly. A complete hw_id implies
  // that those preceding fields are present as well.
  if (hasSampleField(sample, offsetof(Sample, hw_id), sizeof(sample->hw_id))) {
    const auto instructionType =
        static_cast<rocprofiler_pc_sampling_instruction_type_t>(
            sample->inst_type);
    stream << ",\"wave_issued\":" << (sample->wave_issued ? "true" : "false")
           << ",\"instruction_type\":";
    writeJSONString(stream, instructionTypeName(instructionType));
    stream << ",\"instruction_type_id\":"
           << static_cast<uint32_t>(sample->inst_type);
  } else {
    stream << ",\"wave_issued\":null,\"instruction_type\":null"
              ",\"instruction_type_id\":null";
  }

  if (hasSampleField(sample, offsetof(Sample, wave_count),
                     sizeof(sample->wave_count)))
    stream << ",\"wave_count\":" << sample->wave_count;
  else
    stream << ",\"wave_count\":null";

  if (hasSampleField(sample, offsetof(Sample, snapshot),
                     sizeof(sample->snapshot))) {
    const auto reason =
        static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
            sample->snapshot.reason_not_issued);
    stream << ",\"reason_not_issued\":";
    writeJSONString(stream, notIssuedReasonName(reason));
    stream << ",\"reason_not_issued_id\":"
           << static_cast<uint32_t>(sample->snapshot.reason_not_issued)
           << ",\"sampling_lock_error\":"
           << (sample->snapshot.sampling_lock_error ? "true" : "false")
           << ",\"dual_issue_valu\":"
           << (sample->snapshot.dual_issue_valu ? "true" : "false")
           << ",\"arb_state\":{\"issue_valu\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_issue_valu)
           << ",\"issue_matrix\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_issue_matrix)
           << ",\"issue_lds\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_issue_lds)
           << ",\"issue_lds_direct\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_issue_lds_direct)
           << ",\"issue_scalar\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_issue_scalar)
           << ",\"issue_vmem_tex\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_issue_vmem_tex)
           << ",\"issue_flat\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_issue_flat)
           << ",\"issue_exp\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_issue_exp)
           << ",\"issue_misc\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_issue_misc)
           << ",\"issue_brmsg\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_issue_brmsg)
           << ",\"stall_valu\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_stall_valu)
           << ",\"stall_matrix\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_stall_matrix)
           << ",\"stall_lds\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_stall_lds)
           << ",\"stall_lds_direct\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_stall_lds_direct)
           << ",\"stall_scalar\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_stall_scalar)
           << ",\"stall_vmem_tex\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_stall_vmem_tex)
           << ",\"stall_flat\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_stall_flat)
           << ",\"stall_exp\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_stall_exp)
           << ",\"stall_misc\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_stall_misc)
           << ",\"stall_brmsg\":"
           << static_cast<uint32_t>(sample->snapshot.arb_state_stall_brmsg)
           << '}';
  } else {
    stream << ",\"reason_not_issued\":null,\"reason_not_issued_id\":null"
              ",\"sampling_lock_error\":null,\"dual_issue_valu\":null"
              ",\"arb_state\":null";
  }

  const bool hasMemoryCounters =
      hasSampleField(sample, offsetof(Sample, memory_counters),
                     sizeof(sample->memory_counters)) &&
      sample->flags.has_memory_counter;
  if (hasMemoryCounters) {
    const auto &counters = sample->memory_counters;
    stream << ",\"memory_counters\":{\"load_cnt\":"
           << static_cast<uint64_t>(counters.load_cnt)
           << ",\"store_cnt\":" << static_cast<uint64_t>(counters.store_cnt)
           << ",\"bvh_cnt\":" << static_cast<uint64_t>(counters.bvh_cnt)
           << ",\"sample_cnt\":" << static_cast<uint64_t>(counters.sample_cnt)
           << ",\"ds_cnt\":" << static_cast<uint64_t>(counters.ds_cnt)
           << ",\"km_cnt\":" << static_cast<uint64_t>(counters.km_cnt)
           << ",\"async_cnt\":" << static_cast<uint64_t>(counters.async_cnt)
           << ",\"tensor_cnt\":" << static_cast<uint64_t>(counters.tensor_cnt)
           << ",\"xnack_cnt\":" << static_cast<uint64_t>(counters.xnack_cnt)
           << '}';
  } else {
    stream << ",\"memory_counters\":null";
  }
  stream << "}\n";
}

void appendHostTrapRawSample(
    std::ostream &stream,
    const rocprofiler_pc_sampling_record_host_trap_v0_t *sample) {
  using Sample = rocprofiler_pc_sampling_record_host_trap_v0_t;
  appendCommonRawSample(stream, sample, "host-trap");
  // wave_in_group is a trailing bit field in the host-trap record.
  if (sample->size >= sizeof(Sample))
    stream << ",\"wave_in_group\":"
           << static_cast<uint32_t>(sample->wave_in_group);
  else
    stream << ",\"wave_in_group\":null";
  stream << ",\"wave_issued\":null,\"instruction_type\":null"
            ",\"instruction_type_id\":null"
            ",\"wave_count\":null,\"reason_not_issued\":null"
            ",\"reason_not_issued_id\":null,\"sampling_lock_error\":null"
            ",\"dual_issue_valu\":null,\"arb_state\":null"
            ",\"memory_counters\":null}\n";
}

template <bool CheckSuccess>
void stopContextIfStarted(rocprofiler_context_id_t context, bool &started) {
  if (!started)
    return;
  rocprofiler::stopContext<CheckSuccess>(context);
  started = false;
}

template <bool CheckSuccess>
void flushBuffers(const std::vector<rocprofiler_buffer_id_t> &buffers) {
  for (auto &buffer : buffers)
    rocprofiler::flushBuffer<CheckSuccess>(buffer);
}

std::optional<uint64_t> parsePCSamplingInterval(const std::string &value) {
  if (value.empty())
    return std::nullopt;

  uint64_t parsed = 0;
  auto begin = value.data();
  auto end = begin + value.size();
  auto [ptr, ec] = std::from_chars(begin, end, parsed);
  if (ec != std::errc{} || ptr != end || parsed == 0)
    return std::nullopt;
  return parsed;
}

const char *rocprofilerStatusName(rocprofiler_status_t status) {
  switch (status) {
  case ROCPROFILER_STATUS_SUCCESS:
    return "ROCPROFILER_STATUS_SUCCESS";
  case ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE:
    return "ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE";
  default:
    return "ROCPROFILER_STATUS_ERROR";
  }
}

} // namespace

RocprofSDKPCSampling::RocprofSDKPCSampling() = default;

RocprofSDKPCSampling::~RocprofSDKPCSampling() = default;

void RocprofSDKPCSampling::configure(rocprofiler_buffer_tracing_cb_t callback) {
  pcSamplingConfigurationFailureReason.clear();
  auto methodStr = getStrEnv("PROTON_ROCPROFILER_PC_SAMPLING_METHOD");
  std::optional<rocprofiler_pc_sampling_method_t> requestedMethod;
  if (!parsePCSamplingMethod(methodStr, requestedMethod)) {
    pcSamplingConfigurationFailureReason =
        "invalid PROTON_ROCPROFILER_PC_SAMPLING_METHOD='" + methodStr +
        "'; expected 'stochastic' or 'host-trap'";
    return;
  }
  auto intervalStr = getStrEnv("PROTON_PC_SAMPLING_INTERVAL");
  if (!intervalStr.empty()) {
    auto parsedInterval = parsePCSamplingInterval(intervalStr);
    if (parsedInterval) {
      pcSamplingInterval = *parsedInterval;
    } else {
      invalidPCSamplingInterval = intervalStr;
    }
  }
  rocprofiler::createContext<true>(&pcSamplingContext);

  std::vector<AgentPCSamplingConfig> agentsWithPCSampling;
  rocprofiler::queryAvailableAgents<true>(
      ROCPROFILER_AGENT_INFO_VERSION_0, agentQueryCallback,
      sizeof(rocprofiler_agent_t), &agentsWithPCSampling);

  if (agentsWithPCSampling.empty()) {
    pcSamplingConfigurationFailureReason =
        "rocprofiler-sdk did not report PC sampling configurations for any "
        "visible AMD GPU agent";
    return;
  }

  for (auto &agent : agentsWithPCSampling) {
    auto *picked = pickPCSamplingConfig(agent.configs, requestedMethod);
    if (!picked) {
      pcSamplingConfigurationFailureReason =
          "agent " + std::to_string(agent.agentId.handle);
      if (requestedMethod) {
        pcSamplingConfigurationFailureReason +=
            " does not support PROTON_ROCPROFILER_PC_SAMPLING_METHOD='" +
            methodStr + "'";
      } else {
        pcSamplingConfigurationFailureReason +=
            " has no supported PC sampling method";
      }
      return;
    }

    auto interval = pcSamplingInterval;
    if (interval < picked->min_interval)
      interval = picked->min_interval;
    if (interval > picked->max_interval)
      interval = picked->max_interval;

    rocprofiler_buffer_id_t pcSamplingBuffer{};
    size_t pcSamplingWatermark =
        PCSamplingBufferSize - (PCSamplingBufferSize / 4);
    rocprofiler::createBuffer<true>(pcSamplingContext, PCSamplingBufferSize,
                                    pcSamplingWatermark,
                                    ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                    callback, nullptr, &pcSamplingBuffer);

    auto cfgStatus = rocprofiler::configurePCSamplingService<false>(
        pcSamplingContext, agent.agentId, picked->method, picked->unit,
        interval, pcSamplingBuffer, 0);

    if (cfgStatus == ROCPROFILER_STATUS_SUCCESS) {
      rocprofiler_callback_thread_t pcSamplingThread{};
      rocprofiler::createCallbackThread<true>(&pcSamplingThread);
      rocprofiler::assignCallbackThread<true>(pcSamplingBuffer,
                                              pcSamplingThread);
      pcSamplingBuffers.push_back(pcSamplingBuffer);
    } else {
      if (cfgStatus == ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE) {
        pcSamplingConfigurationFailureReason =
            "rocprofiler-sdk PC sampling service is not available for agent ";
      } else {
        pcSamplingConfigurationFailureReason =
            "rocprofiler-sdk failed to configure PC sampling for agent ";
      }
      pcSamplingConfigurationFailureReason +=
          std::to_string(agent.agentId.handle) +
          " with status=" + rocprofilerStatusName(cfgStatus) + "(" +
          std::to_string(static_cast<int>(cfgStatus)) + ")";
      return;
    }
  }
  pcSamplingServiceConfigured = true;
}

void RocprofSDKPCSampling::warnIfInvalidInterval() {
  if (invalidPCSamplingInterval.empty() || intervalWarningEmitted)
    return;
  intervalWarningEmitted = true;
  std::cerr << "[PROTON] Ignoring invalid PROTON_PC_SAMPLING_INTERVAL='"
            << invalidPCSamplingInterval
            << "'; expected a positive integer. Using the default interval."
            << std::endl;
}

void RocprofSDKPCSampling::warnIfSourceLocationsUnavailable() {
  if (sourceLocationWarningEmitted ||
      PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE)
    return;
  sourceLocationWarningEmitted = true;
  std::cerr
      << "[PROTON] AMD PC sampling source-line attribution is unavailable "
         "with this rocprofiler-sdk build; samples will fall back to "
         "kernel-level attribution."
      << std::endl;
}

void RocprofSDKPCSampling::setRawOutputPath(const std::string &path) {
  std::lock_guard<std::mutex> lock(rawOutputMutex);
  // A second concurrently active session with the same profiler mode shares
  // this stream. A later, sequential session must reopen it so each capture
  // starts with a fresh schema record.
  if (pcSamplingStarted && path == rawOutputPath && rawOutput && *rawOutput)
    return;
  rawOutputEnabled.store(false, std::memory_order_release);
  rawOutput.reset();
  rawOutputPath.clear();
  rawOutputErrorEmitted = false;
  rawClockInfoWritten = false;
  if (path.empty())
    return;

  auto output =
      std::make_unique<std::ofstream>(path, std::ios::out | std::ios::trunc);
  if (!*output)
    throw std::runtime_error("[PROTON] Failed to open AMD PC-sampling raw "
                             "output file: " +
                             path);
  *output << "{\"type\":\"schema\",\"schema\":"
             "\"proton-amd-pc-sampling\",\"version\":1}\n";
  output->flush();
  if (!*output)
    throw std::runtime_error("[PROTON] Failed to initialize AMD PC-sampling "
                             "raw output file: " +
                             path);

  rawOutputPath = path;
  rawOutput = std::move(output);
  rawOutputEnabled.store(true, std::memory_order_release);
}

void RocprofSDKPCSampling::setAggregationEnabled(bool enabled) {
  aggregationEnabled.store(enabled, std::memory_order_release);
}

void RocprofSDKPCSampling::appendRawOutput(const std::string &records) {
  if (records.empty())
    return;
  std::lock_guard<std::mutex> lock(rawOutputMutex);
  if (!rawOutput)
    return;
  *rawOutput << records;
  if (!*rawOutput && !rawOutputErrorEmitted) {
    rawOutputErrorEmitted = true;
    std::cerr << "[PROTON] Failed to write AMD PC-sampling raw output file: "
              << rawOutputPath << std::endl;
    rawOutputEnabled.store(false, std::memory_order_release);
  }
}

void RocprofSDKPCSampling::appendRawClockInfo(int64_t timestampOffsetNs) {
  std::lock_guard<std::mutex> lock(rawOutputMutex);
  if (!rawOutput || rawClockInfoWritten)
    return;
  *rawOutput << "{\"type\":\"clock_info\",\"timestamp_unit\":\"ns\","
                "\"timestamp_offset_ns\":"
             << timestampOffsetNs << "}\n";
  rawClockInfoWritten = true;
  if (!*rawOutput && !rawOutputErrorEmitted) {
    rawOutputErrorEmitted = true;
    std::cerr << "[PROTON] Failed to write AMD PC-sampling clock metadata: "
              << rawOutputPath << std::endl;
    rawOutputEnabled.store(false, std::memory_order_release);
  }
}

void RocprofSDKPCSampling::flushRawOutput() {
  std::lock_guard<std::mutex> lock(rawOutputMutex);
  if (!rawOutput)
    return;
  rawOutput->flush();
  if (!*rawOutput && !rawOutputErrorEmitted) {
    rawOutputErrorEmitted = true;
    std::cerr << "[PROTON] Failed to flush AMD PC-sampling raw output file: "
              << rawOutputPath << std::endl;
    rawOutputEnabled.store(false, std::memory_order_release);
  }
}

void RocprofSDKPCSampling::recordKernelSymbol(
    const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
        &symbol) {
  if (symbol.kernel_id == 0 || symbol.kernel_name == nullptr)
    return;

  KernelSymbolInfo info;
  info.name = symbol.kernel_name;
  // AMDGPU ELF objects append ".kd" (kernel descriptor) to symbol names.
  // Strip it so user-visible kernel names match the source.
  const std::string suffix = ".kd";
  if (info.name.size() > suffix.size() &&
      info.name.compare(info.name.size() - suffix.size(), suffix.size(),
                        suffix) == 0)
    info.name.resize(info.name.size() - suffix.size());
  info.codeObjectId = symbol.code_object_id;
  metadataState.withLock([&](MetadataState &state) {
    state.kernelSymbols.insert_or_assign(symbol.kernel_id, std::move(info));
  });
}

void RocprofSDKPCSampling::start() {
  if (pcSamplingServiceConfigured && !pcSamplingStarted) {
    rocprofiler::startContext<true>(pcSamplingContext);
    pcSamplingStarted = true;
  }
}

void RocprofSDKPCSampling::stop() {
  stopContextIfStarted<true>(pcSamplingContext, pcSamplingStarted);
}

void RocprofSDKPCSampling::stopNoThrow() {
  stopContextIfStarted<false>(pcSamplingContext, pcSamplingStarted);
}

void RocprofSDKPCSampling::flushBuffers() {
  proton::flushBuffers<true>(pcSamplingBuffers);
}

void RocprofSDKPCSampling::flushBuffersNoThrow() {
  proton::flushBuffers<false>(pcSamplingBuffers);
}

void RocprofSDKPCSampling::recordResolvedTarget(
    uint64_t dispatchId, uint64_t kernelId, const DataToEntryMap &dataToEntry,
    bool needsKernelChild) {
  PCSamplingTarget target;
  target.dataToEntry = dataToEntry;
  target.needsKernelChild = needsKernelChild;
  metadataState.withLock([&](MetadataState &state) {
    auto symbol = state.kernelSymbols.find(kernelId);
    if (symbol != state.kernelSymbols.end()) {
      target.kernelName = symbol->second.name;
      target.codeObjectId = symbol->second.codeObjectId;
    }
    state.dispatchTargets.insert_or_assign(dispatchId, std::move(target));
  });
}

std::unique_ptr<PCSamplingMetric>
RocprofSDKPCSampling::makePCSamplingMetric(const PCSamplingAccum &accum) {
  auto metric = std::make_unique<PCSamplingMetric>();
  for (int i = 0; i < PCSamplingMetric::PCSamplingMetricKind::Count; ++i)
    metric->updateValue(
        i, MetricValueType(static_cast<uint64_t>(accum.values[i])));
  return metric;
}

void RocprofSDKPCSampling::recordSample(
    PCSamplingMetric::PCSamplingMetricKind stallKind, bool isStalled,
    uint64_t dispatchId, uint64_t codeObjectId, uint64_t pcOffset,
    bool retainRawPC) {
  const bool aggregate = aggregationEnabled.load(std::memory_order_acquire);
  if (!aggregate && !retainRawPC)
    return;
  samplingState.withLock([&](SamplingState &state) {
    if (codeObjectId != ROCPROFILER_CODE_OBJECT_ID_NONE)
      state.pendingCodeObjectIds.insert(codeObjectId);
    const PCSamplingKey key{dispatchId, codeObjectId, pcOffset};
    if (retainRawPC)
      state.rawPCs.insert(key);
    if (aggregate) {
      auto &accum = state.accum[key];
      accum.values[PCSamplingMetric::NumSamples]++;
      if (isStalled) {
        accum.values[PCSamplingMetric::NumStalledSamples]++;
        accum.values[stallKind]++;
      }
    }
  });
}

void RocprofSDKPCSampling::processBuffer(rocprofiler_record_header_t **headers,
                                         size_t numHeaders,
                                         uint64_t dropCount) {
  if (dropCount > 0) {
    std::cerr << "[PROTON] ROCProfiler-SDK dropped " << dropCount
              << " PC sampling records" << std::endl;
  }

  const bool writeRawOutput = rawOutputEnabled.load(std::memory_order_acquire);
  std::ostringstream rawRecords;
  if (writeRawOutput && dropCount > 0)
    rawRecords << "{\"type\":\"dropped_samples\",\"count\":" << dropCount
               << "}\n";
  for (size_t i = 0; i < numHeaders; ++i) {
    auto *header = headers[i];
    if (!header || header->category != ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
      continue;

    if (header->kind == ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE) {
      auto *sample =
          static_cast<rocprofiler_pc_sampling_record_stochastic_v0_t *>(
              header->payload);
      using StochasticSample = rocprofiler_pc_sampling_record_stochastic_v0_t;
      bool hasWaveIssueInfo = hasSampleField(
          sample, offsetof(StochasticSample, hw_id), sizeof(sample->hw_id));
      bool hasSnapshot =
          hasSampleField(sample, offsetof(StochasticSample, snapshot),
                         sizeof(sample->snapshot));
      bool isStalled = hasWaveIssueInfo && !sample->wave_issued;
      auto stallKind =
          isStalled && hasSnapshot
              ? mapNotIssuedReasonToStallMetric(
                    static_cast<
                        rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
                        sample->snapshot.reason_not_issued))
              : PCSamplingMetric::StalledSelected;
      auto pc = getSamplePC(sample);
      if (writeRawOutput)
        appendStochasticRawSample(rawRecords, sample);
      recordSample(stallKind, isStalled, sample->dispatch_id, pc.code_object_id,
                   pc.code_object_offset, writeRawOutput);
    } else if (header->kind ==
               ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE) {
      auto *sample =
          static_cast<rocprofiler_pc_sampling_record_host_trap_v0_t *>(
              header->payload);
      auto pc = getSamplePC(sample);
      if (writeRawOutput)
        appendHostTrapRawSample(rawRecords, sample);
      recordSample(PCSamplingMetric::NumSamples, false, sample->dispatch_id,
                   pc.code_object_id, pc.code_object_offset, writeRawOutput);
    }
  }
  if (writeRawOutput)
    appendRawOutput(rawRecords.str());
}

void RocprofSDKPCSampling::flushAccum(int64_t timestampOffsetNs) {
  std::lock_guard<std::mutex> flushLock(flushMutex);
  PCSamplingAccumMap snapshot;
  std::unordered_set<PCSamplingKey, PCSamplingKeyHash> rawPCSnapshot;
  std::unordered_set<uint64_t> snapshotCodeObjectIds;
  samplingState.withLock([&](SamplingState &state) {
    snapshot.swap(state.accum);
    rawPCSnapshot.swap(state.rawPCs);
    snapshotCodeObjectIds.swap(state.pendingCodeObjectIds);
    state.flushingCodeObjectIds.insert(snapshotCodeObjectIds.begin(),
                                       snapshotCodeObjectIds.end());
  });
  std::unordered_map<uint64_t, PCSamplingTarget> snapshotTargets;
  metadataState.withLock([&](MetadataState &state) {
    snapshotTargets.swap(state.dispatchTargets);
  });
  appendRawClockInfo(timestampOffsetNs);
  if (snapshot.empty() && rawPCSnapshot.empty()) {
    flushRawOutput();
    return;
  }

  std::unordered_map<uint64_t, PCSamplingAccum> unresolvedAccum;
  std::ostringstream rawPCInfo;
  const bool writeRawOutput = rawOutputEnabled.load(std::memory_order_acquire);
  const PCSamplingTarget unknownTarget;

  std::unordered_set<PCSamplingKey, PCSamplingKeyHash> sampledPCs =
      std::move(rawPCSnapshot);
  for (const auto &[key, _] : snapshot)
    sampledPCs.insert(key);

  for (const auto &key : sampledPCs) {
    const auto dispatchId = key.dispatchId;
    const auto codeObjectId = key.codeObjectId;
    const auto pcOffset = key.pcOffset;
    auto accumIt = snapshot.find(key);
    const PCSamplingAccum *accum =
        accumIt == snapshot.end() ? nullptr : &accumIt->second;
    auto found = snapshotTargets.find(dispatchId);
    PCInfo pcInfo;
    if (found != snapshotTargets.end() || writeRawOutput) {
      const auto &target =
          found == snapshotTargets.end() ? unknownTarget : found->second;
      metadataState.withLock([&](MetadataState &state) {
        pcInfo = resolvePCInfoLocked(state, codeObjectId, pcOffset, target);
      });
    }

    if (writeRawOutput) {
      rawPCInfo << "{\"type\":\"pc_info\",\"dispatch_id\":" << dispatchId
                << ",\"code_object_id\":" << codeObjectId
                << ",\"pc_offset\":" << pcOffset << ",\"kernel_name\":";
      if (found != snapshotTargets.end())
        writeJSONString(rawPCInfo, found->second.kernelName);
      else
        rawPCInfo << "null";
      rawPCInfo << ",\"instruction\":";
      if (pcInfo.instruction)
        writeJSONString(rawPCInfo, *pcInfo.instruction);
      else
        rawPCInfo << "null";
      rawPCInfo << ",\"source_file\":";
      if (pcInfo.sourceLocation)
        writeJSONString(rawPCInfo, pcInfo.sourceLocation->file);
      else
        rawPCInfo << "null";
      rawPCInfo << ",\"source_line\":";
      if (pcInfo.sourceLocation)
        rawPCInfo << pcInfo.sourceLocation->line;
      else
        rawPCInfo << "null";
      rawPCInfo << ",\"source_function\":";
      if (pcInfo.sourceLocation)
        writeJSONString(rawPCInfo, pcInfo.sourceLocation->function);
      else
        rawPCInfo << "null";
      rawPCInfo << "}\n";
    }

    if (found == snapshotTargets.end() || accum == nullptr)
      continue;
    const auto &target = found->second;

    if (!pcInfo.sourceLocation) {
      auto &unresolved = unresolvedAccum[dispatchId];
      for (int i = 0; i < PCSamplingMetric::PCSamplingMetricKind::Count; ++i)
        unresolved.values[i] += accum->values[i];
      continue;
    }

    for (auto &[data, entry] : target.dataToEntry) {
      auto pcEntry = entry;
      if (target.needsKernelChild)
        pcEntry =
            data->addOp(entry.phase, entry.id, {Context(target.kernelName)});
      pcEntry = data->addOp(
          pcEntry.phase, pcEntry.id,
          {Context(formatFileLineFunction(pcInfo.sourceLocation->file,
                                          pcInfo.sourceLocation->line,
                                          pcInfo.sourceLocation->function))});
      pcEntry.upsertMetric(makePCSamplingMetric(*accum));
    }
  }

  for (auto &[dispatchId, accum] : unresolvedAccum) {
    auto found = snapshotTargets.find(dispatchId);
    if (found == snapshotTargets.end())
      continue;
    const auto &target = found->second;

    for (auto &[data, entry] : target.dataToEntry) {
      auto pcEntry = entry;
      if (target.needsKernelChild)
        pcEntry =
            data->addOp(entry.phase, entry.id, {Context(target.kernelName)});
      pcEntry.upsertMetric(makePCSamplingMetric(accum));
    }
  }

  if (writeRawOutput)
    appendRawOutput(rawPCInfo.str());
  flushRawOutput();

  samplingState.withLock([&](SamplingState &state) {
    for (auto codeObjectId : snapshotCodeObjectIds)
      state.flushingCodeObjectIds.erase(codeObjectId);
  });

  for (auto codeObjectId : snapshotCodeObjectIds)
    tryReleaseCodeObject(codeObjectId);
}

} // namespace proton

#else

namespace proton {

RocprofSDKPCSampling::RocprofSDKPCSampling() = default;

RocprofSDKPCSampling::~RocprofSDKPCSampling() = default;

RocprofSDKPCSampling::MetadataState::MetadataState() = default;

RocprofSDKPCSampling::MetadataState::~MetadataState() = default;

void RocprofSDKPCSampling::configure(rocprofiler_buffer_tracing_cb_t callback) {
  (void)callback;
}

void RocprofSDKPCSampling::warnIfInvalidInterval() {}

void RocprofSDKPCSampling::warnIfSourceLocationsUnavailable() {}

void RocprofSDKPCSampling::setRawOutputPath(const std::string &path) {
  (void)path;
}

void RocprofSDKPCSampling::setAggregationEnabled(bool enabled) {
  (void)enabled;
}

void RocprofSDKPCSampling::recordCodeObjectLoad(
    const rocprofiler_callback_tracing_code_object_load_data_t &load) {
  (void)load;
}

void RocprofSDKPCSampling::recordCodeObjectUnload(uint64_t codeObjectId) {
  (void)codeObjectId;
}

void RocprofSDKPCSampling::recordKernelSymbol(
    const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
        &symbol) {
  (void)symbol;
}

void RocprofSDKPCSampling::start() {}

void RocprofSDKPCSampling::stop() {}

void RocprofSDKPCSampling::stopNoThrow() {}

void RocprofSDKPCSampling::flushBuffers() {}

void RocprofSDKPCSampling::flushBuffersNoThrow() {}

void RocprofSDKPCSampling::recordResolvedTarget(
    uint64_t dispatchId, uint64_t kernelId, const DataToEntryMap &dataToEntry,
    bool needsKernelChild) {
  (void)dispatchId;
  (void)kernelId;
  (void)dataToEntry;
  (void)needsKernelChild;
}

void RocprofSDKPCSampling::processBuffer(rocprofiler_record_header_t **headers,
                                         size_t numHeaders,
                                         uint64_t dropCount) {
  (void)headers;
  (void)numHeaders;
  (void)dropCount;
}

void RocprofSDKPCSampling::flushAccum(int64_t timestampOffsetNs) {
  (void)timestampOffsetNs;
}

void RocprofSDKPCSampling::appendRawOutput(const std::string &records) {
  (void)records;
}

void RocprofSDKPCSampling::appendRawClockInfo(int64_t timestampOffsetNs) {
  (void)timestampOffsetNs;
}

void RocprofSDKPCSampling::flushRawOutput() {}

} // namespace proton

#endif
