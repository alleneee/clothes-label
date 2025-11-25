#!/bin/bash
# simple_api 启动脚本（使用指定 conda 环境并将日志输出到当前目录）

set -euo pipefail

PROJECT_DIR="/data/hx/model-train"
ENV_PATH="/root/.conda/envs/model-train"
PYTHON_BIN="$ENV_PATH/bin/python"
LOG_FILE="$PROJECT_DIR/simple_api.log"
PID_FILE="$PROJECT_DIR/simple_api.pid"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "项目目录不存在: $PROJECT_DIR"
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "找不到 Python 解释器: $PYTHON_BIN"
    exit 1
fi

cd "$PROJECT_DIR"

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" >/dev/null 2>&1; then
        echo "检测到正在运行的 simple_api (PID=$OLD_PID)，尝试停止..."
        kill "$OLD_PID"
        sleep 2
        if kill -0 "$OLD_PID" >/dev/null 2>&1; then
            echo "进程未正常退出，执行强制终止"
            kill -9 "$OLD_PID" || true
        fi
    fi
    rm -f "$PID_FILE"
fi

nohup "$PYTHON_BIN" simple_api.py >"$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "simple_api 已在后台启动"
echo "PID: $NEW_PID"
echo "日志: $LOG_FILE"
echo "PID 文件: $PID_FILE"
