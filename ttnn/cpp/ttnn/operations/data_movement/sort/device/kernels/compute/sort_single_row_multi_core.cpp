// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"

#include "sort_common.hpp"

namespace NAMESPACE {
/*
This kernel implements a parallel Bitonic Sort for a single row of tiles, distributing the work across multiple cores
for efficiency.

### High-Level Workflow:
- The row to be sorted contains `Wt` tiles. Sorting is performed in stages, with each stage processing pairs of tiles.
- Multiple cores work in parallel, each handling one or more pairs per stage, depending on the number of available
cores.
- A coordinator core manages synchronization and ensures correct stage progression.

### Detailed Steps:
1. **Initialization**:
    - For each row, the coordinator core prepares the index tensor and copies input values to the output memory bank.
    - The output memory bank is used both for final results and as temporary storage between stages.

2. **Bitonic Sort Stages**:
    - The sort proceeds in multiple stages, as required by the Bitonic sort algorithm.
    - In each stage:
      - Cores are assigned pairs of tiles to process, based on their core ID and the current stage.
      - If there are fewer cores than pairs, some cores handle multiple pairs.
      - The coordinator core signals other cores to begin processing.
      - Each core reads its assigned tile pairs (both values and indices), sorts them according to the required
direction (ascending or descending), and writes the results back to the output memory bank.
      - After processing, each core signals completion to the coordinator.
      - The coordinator waits for all cores to finish before moving to the next stage.

3. **Completion**:
    - The process repeats for all stages until the row is fully sorted.
    - The entire procedure is repeated for each row in the `Ht` dimension.

### Additional Notes:
- The algorithm dynamically assigns work to ensure all cores are utilized efficiently.
- Synchronization between cores is handled using semaphores to guarantee correct ordering and data consistency.
- The coordinator core orchestrates stage transitions and manages memory access.
*/
void MAIN {
    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_tensor_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_tensor_transposed_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t input_tensor_output_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t Ht = get_compile_time_arg_val(7);
    constexpr uint32_t number_of_available_cores = get_compile_time_arg_val(8);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(9);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(10);
    constexpr bool descending = get_compile_time_arg_val(11);
    constexpr bool stable =
        get_compile_time_arg_val(12);  // TODO: In the future change LLK to have the option or add additional step with
                                       // checking values and indexes after the sorting
                                       // Issue: https://github.com/tenstorrent/tt-metal/issues/20625
    constexpr uint32_t log2Wt = get_compile_time_arg_val(13);

    // Constants
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;

    ckernel::topk_tile_init();
    transpose_wh_init(input_tensor_cb_index, input_tensor_transposed_cb_index);

    for (uint32_t h = 0; h < Ht; h++) {
        const bool ascending = !descending;
        // Get core start value
        const uint32_t core_start =
            get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Processing each row
        for (uint32_t stage = 1; stage <= log2Wt; stage++) {
            const uint32_t m_iter = stage - 1;
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);

                uint16_t pair_id = 0;
                uint32_t processing_pair_id = core_start;
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;
                    if (j > i) {
                        // Determine direction for this comparison block
                        const bool ascending_block = ((i >> stage) & 1) == 0;
                        const bool dir = ascending_block == ascending;

                        if (pair_id == processing_pair_id) {
                            // Get indexes of tiles to compare
                            const uint32_t left_tile_id = i;
                            const uint32_t right_tile_id = j;

                            // Wait for data from reader
                            cb_wait_front(input_tensor_cb_index, 2 * one_tile);
                            cb_wait_front(index_tensor_cb_index, 2 * one_tile);

                            tile_regs_acquire();
                            if (stage == 1 && sub == 1) {
                                // First stage and substage - transpose the data for for LLK
                                // Process value tiles
                                reconfig_data_format_srca(input_tensor_cb_index);
                                transpose_wh_init_short(input_tensor_cb_index);
                                transpose_wh_tile(input_tensor_cb_index, 0, input_dest_start);
                                transpose_wh_tile(input_tensor_cb_index, 1, input_dest_end);

                                // Process index tiles
                                reconfig_data_format_srca(index_tensor_cb_index);
                                transpose_wh_init_short(index_tensor_cb_index);
                                transpose_wh_tile(index_tensor_cb_index, 0, index_dest_start);
                                transpose_wh_tile(index_tensor_cb_index, 1, index_dest_end);
                            } else {
                                // Intermediate step - tiles are already transposed
                                // Process value tiles
                                reconfig_data_format_srca(input_tensor_cb_index);
                                copy_tile_to_dst_init_short(input_tensor_cb_index);
                                copy_tile(input_tensor_cb_index, 0, input_dest_start);
                                copy_tile(input_tensor_cb_index, 1, input_dest_end);

                                // Process index tiles
                                reconfig_data_format_srca(index_tensor_cb_index);
                                copy_tile_to_dst_init_short(index_tensor_cb_index);
                                copy_tile(index_tensor_cb_index, 0, index_dest_start);
                                copy_tile(index_tensor_cb_index, 1, index_dest_end);
                            }

                            cb_pop_front(input_tensor_cb_index, 2 * one_tile);
                            cb_pop_front(index_tensor_cb_index, 2 * one_tile);

                            uint32_t tile_input_low = input_dest_start;
                            uint32_t tile_input_high = input_dest_end;
                            uint32_t tile_index_low = index_dest_start;
                            uint32_t tile_index_high = index_dest_end;

                            if (sub == 1) {
                                // Use sort LLK only the last substage to sort the last pair of tiles - speed up
                                ckernel::topk_local_sort(/*idst=*/0, (int)dir, /*end_phase(log2(K))=*/5);
                            } else {
                                // For all other stages use topk_merge to put the top K values in one tile, and the
                                // bottom K values in another tile
                                ckernel::topk_merge(/*idst=*/0, m_iter, /*k=*/32);

                                // topk_merge puts smallest values in DEST[0] and largest in DEST[1]
                                // We swap their indices when using descending order
                                if (dir) {
                                    tile_input_low = input_dest_end;
                                    tile_input_high = input_dest_start;
                                    tile_index_low = index_dest_end;
                                    tile_index_high = index_dest_start;
                                }
                            }

                            tile_regs_commit();
                            tile_regs_wait();

                            if (stage == log2Wt && sub == 1) {
                                // Last step of the last stage - transpose tiles back to the original format

                                // Reserve space for temporary buffers
                                cb_reserve_back(input_tensor_transposed_cb_index, 2 * one_tile);
                                cb_reserve_back(index_tensor_transposed_cb_index, 2 * one_tile);

                                // Process value tiles
                                pack_reconfig_data_format(input_tensor_transposed_cb_index);
                                pack_tile(tile_input_low, input_tensor_transposed_cb_index);
                                pack_tile(tile_input_high, input_tensor_transposed_cb_index);

                                // Process index tiles
                                pack_reconfig_data_format(index_tensor_transposed_cb_index);
                                pack_tile(tile_index_low, index_tensor_transposed_cb_index);
                                pack_tile(tile_index_high, index_tensor_transposed_cb_index);

                                // Push tiles to synchronize unpacker and packer
                                cb_push_back(input_tensor_transposed_cb_index, 2 * one_tile);
                                cb_push_back(index_tensor_transposed_cb_index, 2 * one_tile);

                                tile_regs_release();

                                // Pack and push sorted values tensor tiles
                                acquire_dst();

                                cb_wait_front(input_tensor_transposed_cb_index, 2 * one_tile);
                                reconfig_data_format_srca(input_tensor_transposed_cb_index);
                                transpose_wh_init_short(input_tensor_transposed_cb_index);
                                transpose_wh_tile(input_tensor_transposed_cb_index, 0, input_dest_start);
                                transpose_wh_tile(input_tensor_transposed_cb_index, 1, input_dest_end);

                                cb_reserve_back(input_tensor_output_cb_index, 2 * one_tile);
                                pack_reconfig_data_format(input_tensor_output_cb_index);
                                pack_tile(input_dest_start, input_tensor_output_cb_index);
                                pack_tile(input_dest_end, input_tensor_output_cb_index);

                                // Push value tiles to writer and free transposed buffer
                                cb_pop_front(input_tensor_transposed_cb_index, 2 * one_tile);
                                cb_push_back(input_tensor_output_cb_index, 2 * one_tile);

                                release_dst();

                                // Pack and push adjusted index tensor tiles
                                acquire_dst();

                                cb_wait_front(index_tensor_transposed_cb_index, 2 * one_tile);
                                reconfig_data_format_srca(index_tensor_transposed_cb_index);
                                transpose_wh_init_short(index_tensor_transposed_cb_index);
                                transpose_wh_tile(index_tensor_transposed_cb_index, 0, input_dest_start);
                                transpose_wh_tile(index_tensor_transposed_cb_index, 1, input_dest_end);

                                cb_reserve_back(index_tensor_output_cb_index, 2 * one_tile);
                                pack_reconfig_data_format(index_tensor_output_cb_index);
                                pack_tile(input_dest_start, index_tensor_output_cb_index);
                                pack_tile(input_dest_end, index_tensor_output_cb_index);

                                // Push index tiles to writer and free transposed buffer
                                cb_pop_front(index_tensor_transposed_cb_index, 2 * one_tile);
                                cb_push_back(index_tensor_output_cb_index, 2 * one_tile);

                                release_dst();
                            } else {
                                // Intermediate step - pack and push transposed tiles to be saved for the next stage
                                cb_reserve_back(index_tensor_output_cb_index, 2 * one_tile);
                                cb_reserve_back(input_tensor_output_cb_index, 2 * one_tile);

                                // Process value tiles
                                pack_reconfig_data_format(input_tensor_output_cb_index);
                                pack_tile(tile_input_low, input_tensor_output_cb_index);
                                pack_tile(tile_input_high, input_tensor_output_cb_index);

                                // Process index tiles
                                pack_reconfig_data_format(index_tensor_output_cb_index);
                                pack_tile(tile_index_low, index_tensor_output_cb_index);
                                pack_tile(tile_index_high, index_tensor_output_cb_index);

                                // Push tiles to writer
                                cb_push_back(input_tensor_output_cb_index, 2 * one_tile);
                                cb_push_back(index_tensor_output_cb_index, 2 * one_tile);

                                tile_regs_release();
                            }

                            processing_pair_id += number_of_available_cores;
                        }  // if pair_id == processing_pair_id
                        pair_id++;
                    }  // if j > i
                }  // i loop
            }  // sub loop
        }  // stage loop
    }  // h loop
}
}  // namespace NAMESPACE
