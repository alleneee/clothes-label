# Gunicorn生产环境配置
import multiprocessing
import os

# 服务器套接字
bind = "0.0.0.0:8000"
backlog = 2048

# Worker进程
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# 超时设置
timeout = 300
keepalive = 2
graceful_timeout = 30

# 应用设置
preload_app = True
reload = False

# 日志设置
accesslog = "/var/log/clothes-api/access.log"
errorlog = "/var/log/clothes-api/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 进程管理
pidfile = "/var/run/clothes-api.pid"
user = None  # 设置为运行用户
group = None  # 设置为运行组

# 安全设置
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# 性能调优
worker_tmp_dir = "/dev/shm"  # 使用内存文件系统提升性能

# 启动前钩子
def on_starting(server):
    server.log.info("正在启动衣服分类API服务...")

def when_ready(server):
    server.log.info("衣服分类API服务已准备就绪")

def on_exit(server):
    server.log.info("衣服分类API服务正在关闭...")

# Worker进程钩子
def worker_int(worker):
    worker.log.info("Worker收到SIGINT信号")

def pre_fork(server, worker):
    server.log.info(f"Worker {worker.pid} 正在启动")

def post_fork(server, worker):
    server.log.info(f"Worker {worker.pid} 已启动")

def worker_abort(worker):
    worker.log.info(f"Worker {worker.pid} 异常终止")
