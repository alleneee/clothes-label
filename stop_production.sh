#!/bin/bash
# 生产环境停止脚本

set -e

PID_FILE="/var/run/clothes-api.pid"
LOG_DIR="/var/log/clothes-api"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}⏹️  停止衣服分类API服务${NC}"
echo "=================================="

if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}⚠️  PID文件不存在，服务可能未运行${NC}"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  进程 $PID 不存在，清理PID文件${NC}"
    rm -f "$PID_FILE"
    exit 0
fi

echo -e "${YELLOW}🛑 正在停止进程 $PID...${NC}"

# 优雅停止
kill -TERM "$PID"

# 等待进程结束
for i in {1..10}; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo -e "${GREEN}✅ 服务已停止${NC}"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# 强制停止
echo -e "${YELLOW}⚠️  优雅停止超时，强制终止...${NC}"
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PID_FILE"

echo -e "${GREEN}✅ 服务已强制停止${NC}"
echo "=================================="
