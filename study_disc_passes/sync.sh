#!/bin/bash

DEST_DIR=~/workspace/learnmlir/study_disc_passes/

rsync -arvRP mlir/disc/IR/hlo_disc_ops.* ${DEST_DIR}

rsync -arvRP mlir/disc/IR/hlo_disc_enums.* ${DEST_DIR}

rsync -arvRP mlir/disc/IR/custom_call_base.* ${DEST_DIR}

rsync -arvRP mlir/disc/IR/lhlo_disc_ops.* ${DEST_DIR}

rsync -arvRP mlir/disc/IR/lhlo_disc_enums.* ${DEST_DIR}

rsync -arvRP mlir/disc/utils/cycle_detector.* ${DEST_DIR}

rsync -arvRP mlir/disc/transforms/placement_utils.* ${DEST_DIR}

rsync -arvRP mlir/disc/transforms/disc_map_hlo_to_lhlo_op.h ${DEST_DIR}

rsync -arvRP mlir/disc/transforms/disc_shape_optimization_utils.* ${DEST_DIR}

rsync -arvRP mlir/disc/transforms/PassDetail.h ${DEST_DIR}

rsync -arvRP mlir/disc/transforms/mhlo_disc_passes.* ${DEST_DIR}

rsync -arvRP mlir/disc/transforms/fusion_utils.* ${DEST_DIR}
rsync -arvRP mlir/disc/transforms/fusion_utils_* ${DEST_DIR}

rsync -arvRP mlir/disc/transforms/disc_supported_list.h.inc ${DEST_DIR}

rsync -arvRP mlir/disc/transforms/lhlo_elemental_utils.* ${DEST_DIR}

rsync -arvRP mlir/disc/transforms/codegen_utils.* ${DEST_DIR}

rsync -arvRP mlir/disc/transforms/input_inline_fusion_pattern.* ${DEST_DIR}