# Conda → Miniforge 迁移指南

## 一、安装 Miniforge

```bash
# 下载 (使用清华镜像加速)
wget https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh

# 安装 (可自定义安装路径)
bash Miniforge3-Linux-x86_64.sh -b -p /data/miniforge3

# 初始化
/data/miniforge3/bin/conda init bash
source ~/.bashrc
```

## 二、配置国内镜像

```bash
# conda-forge 镜像 (USTC)
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --set channel_priority strict
conda config --set show_channel_urls yes
```

## 三、创建环境

```bash
# 创建新环境
conda create -n <环境名> python=3.12 pip -y

# 激活
conda activate <环境名>
```

## 四、安装依赖

```bash
# 使用阿里云 PyPI 镜像
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 安装额外包
python -m pip install <包名> -i https://mirrors.aliyun.com/pypi/simple/
```

## 五、验证环境

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 六、更新启动脚本

将脚本中的环境路径改为新路径：
```bash
ENV_PATH="/data/miniforge3/envs/<环境名>"
```

## 七、清理旧环境（可选）

```bash
conda remove -n <旧环境名> --all -y
```

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 镜像 403/404 | 切换镜像源（阿里云/USTC/BFSU/清华） |
| NumPy ABI 不兼容 | 固定 `numpy>=1.24.0,<2.0.0` |
| pip 安装错环境 | 使用 `python -m pip` |
| 缺失包 | 手动安装后添加到 requirements.txt |

## 国内镜像汇总

**Conda 镜像：**
- USTC: `https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/`
- 清华: `https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/`

**PyPI 镜像：**
- 阿里云: `https://mirrors.aliyun.com/pypi/simple/`
- 清华: `https://pypi.tuna.tsinghua.edu.cn/simple/`
- USTC: `https://mirrors.ustc.edu.cn/pypi/simple/`
