#!/bin/bash
# 生产环境启动脚本

set -e

# 项目配置
PROJECT_DIR="/data/hx/model-train"
APP_MODULE="simple_api:app"
GUNICORN_CONFIG="gunicorn.conf.py"
LOG_DIR="/var/log/clothes-api"
PID_FILE="/var/run/clothes-api.pid"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 启动衣服分类API生产服务${NC}"
echo "=================================="

# 检查项目目录
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}❌ 项目目录不存在: $PROJECT_DIR${NC}"
    exit 1
fi

cd "$PROJECT_DIR"

# 创建日志目录
echo -e "${YELLOW}📁 创建日志目录...${NC}"
sudo mkdir -p "$LOG_DIR"
sudo chown -R $USER:$USER "$LOG_DIR"

# 检查依赖
echo -e "${YELLOW}🔍 检查Python依赖...${NC}"
if ! python -c "import gunicorn, uvicorn, fastapi, torch" 2>/dev/null; then
    echo -e "${RED}❌ 缺少必要依赖，请运行: pip install -r requirements.txt${NC}"
    exit 1
fi

# 检查模型文件
echo -e "${YELLOW}🤖 检查模型文件...${NC}"
if [ ! -f "model/best.ckpt" ]; then
    echo -e "${RED}❌ 模型文件不存在: model/best.ckpt${NC}"
    exit 1
fi

# 检查配置文件
if [ ! -f "config.yaml" ]; then
    echo -e "${RED}❌ 配置文件不存在: config.yaml${NC}"
    exit 1
fi

# 停止现有服务
if [ -f "$PID_FILE" ]; then
    echo -e "${YELLOW}⏹️  停止现有服务...${NC}"
    kill -TERM $(cat "$PID_FILE") 2>/dev/null || true
    sleep 2
fi

# 启动服务
echo -e "${GREEN}🚀 启动Gunicorn服务...${NC}"
gunicorn \
    --config "$GUNICORN_CONFIG" \
    "$APP_MODULE" \
    --daemon

# 检查服务状态
sleep 3
if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    PID=$(cat "$PID_FILE")
    echo -e "${GREEN}✅ 服务启动成功!${NC}"
    echo "   PID: $PID"
    echo "   地址: http://0.0.0.0:8000"
    echo "   文档: http://0.0.0.0:8000/docs"
    echo "   健康检查: http://0.0.0.0:8000/health"
    echo "   日志目录: $LOG_DIR"
else
    echo -e "${RED}❌ 服务启动失败${NC}"
    echo "请检查日志: $LOG_DIR/error.log"
    exit 1
fi

echo "=================================="
