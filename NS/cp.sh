#!/bin/bash


# 定义源目录和目标目录
SOURCE_DIR=./
TARGET_DIR=../report/figures

# 定义文件列表
FILES=(
    "ns_traj_error_new.png"
    "error_grid_1s_new.png"
    "error_grid_1.5s_new.png"
    "error_grid_2s_new.png"
    "error_grid_2.5s_new.png"
    "omega_error_grid_1s_new.png"
    "omega_error_grid_1.5s_new.png"
    "omega_error_grid_2s_new.png"
    "omega_error_grid_2.5s_new.png"
    "uv_grid_1s_new.png"
    "uv_grid_1.5s_new.png"
    "uv_grid_2s_new.png"
    "uv_grid_2.5s_new.png"
    "omega_grid_1s_new.png"
    "omega_grid_1.5s_new.png"
    "omega_grid_2s_new.png"
    "omega_grid_2.5s_new.png"
)

# 遍历文件列表并复制文件
for FILE in "${FILES[@]}"; do
    cp "$SOURCE_DIR/$FILE" "$TARGET_DIR/ns_$FILE"
done

cp "$SOURCE_DIR/ns_traj_error_new.png" "$TARGET_DIR/ns_traj_error_new.png"

echo "所有文件已成功复制到 $TARGET_DIR"