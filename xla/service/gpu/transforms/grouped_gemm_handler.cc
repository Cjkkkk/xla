/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/transforms/grouped_gemm_handler.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/shape_inference.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

namespace m = match;

int64_t NumberOfInstructionsInComp(const HloComputation* comp, HloOpcode op) {
  int64_t total_count = 0;
  for (const HloInstruction* inst : comp->instructions()) {
    if (inst->opcode() == op) {
      ++total_count;
    }
  }
  return total_count;
}

absl::Status UpdateDotAndConsumerConfig(HloInstruction* dot,
                                        int64_t stream_id) {
  auto dot_gpu_config = dot->backend_config<gpu::GpuBackendConfig>();
  HloInstruction* updater = dot->users()[0];
  auto updater_gpu_config = updater->backend_config<gpu::GpuBackendConfig>();
  dot_gpu_config->set_operation_queue_id(stream_id);
  updater_gpu_config->mutable_wait_on_operation_queues()->Add(stream_id);

  TF_RETURN_IF_ERROR(dot->set_backend_config(dot_gpu_config.value()));
  TF_RETURN_IF_ERROR(updater->set_backend_config(updater_gpu_config.value()));
  return absl::OkStatus();
}

absl::Status SetForceDelayForInstruction(HloInstruction* instr,
                                         bool force_delay) {
  auto gpu_config = instr->backend_config<gpu::GpuBackendConfig>();

  gpu_config->set_force_earliest_schedule(force_delay);

  TF_RETURN_IF_ERROR(instr->set_backend_config(gpu_config.value()));
  return absl::OkStatus();
}

absl::StatusOr<bool> HandleGroupedGemm(HloComputation* comp,
                                        int64_t max_streams) {
  bool changed = false;
  // If we have a einsum loop with only 1 dot, this means either
  // the loop is not unrolled or only 1 partition is available.
  // It's a no-op in either case.
  if (NumberOfInstructionsInComp(comp, HloOpcode::kDot) <= 1) {
    return changed;
  }
  int64_t stream_id = 0;
  for (auto inst : comp->MakeInstructionPostOrder()) {
    if (Match(inst, m::Dot())) {
      // Dispatch the dot to additional compute stream.
      TF_RETURN_IF_ERROR(UpdateDotAndConsumerConfig(inst, stream_id));
      ++stream_id;
      stream_id = stream_id % max_streams;
      changed = true;
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> GroupedGemmHandler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      5, "GroupedGemmHandler::Run(), before:\n" + module->ToString());
  bool changed = false;

  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    if (comp->name().find("grouped_gemm") == 0) {
      VLOG(5) << "Processing computation: " << comp->name();
      TF_ASSIGN_OR_RETURN(bool comp_result,
                          HandleGroupedGemm(comp, MaxStreams));
      changed = comp_result;
    }
  }

  XLA_VLOG_LINES(5,
                 "GroupedGemmHandler::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
