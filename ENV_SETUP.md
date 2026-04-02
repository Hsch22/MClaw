# MClaw Environment Setup (uv)

本项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境。
两个环境通过 `pyproject.toml` 中的 **dependency groups** 区分。

## 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 环境 1: mclaw-train (训练节点)

```bash
cd /path/to/MClaw

# 创建 venv 并安装 core + train 依赖
uv sync --group train

# verl (AgentGym-RL fork) 不通过 pip 安装，需设置 PYTHONPATH
export PYTHONPATH="/path/to/AgentGym-RL/AgentGym-RL:$PYTHONPATH"
```

> **注意**: `flash-attn` 需要匹配的 CUDA toolkit，如果安装失败可先跳过：
> `uv sync --group train --exclude flash-attn` 然后手动编译安装。

## 环境 2: mclaw-envserver (环境服务器节点)

```bash
cd /path/to/MClaw

# 只安装 envserver 组的依赖（不含 core dependencies）
uv sync --only-group envserver
```

## 常用命令

```bash
# 添加新依赖到 train 组
uv add --group train <package>

# 添加新依赖到 envserver 组
uv add --group envserver <package>

# 锁定依赖（生成 uv.lock）
uv lock

# 从 lock 文件精确还原
uv sync --frozen --group train
```

## 原始环境快照

完整的 pip freeze 快照保存在 `envs/` 目录下，供参考：
- `envs/mclaw-train-freeze.txt`
- `envs/mclaw-envserver-freeze.txt`
