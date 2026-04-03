# MClaw Environment Setup (uv)

本项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境。
两个环境通过 `pyproject.toml` 中的 **dependency groups** 区分。

## 前置条件

| 依赖 | 说明 |
|------|------|
| uv | Python 包管理器 |
| CUDA Toolkit 12.4 | 编译 flash-attn 需要 nvcc |
| NVIDIA 驱动 >= 525 | 向下兼容 CUDA 12.4 |
| Python 3.12 | xformers 预编译 wheel 仅覆盖 cp310/cp311/cp312 |

## 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 完整部署步骤（新机器）

### 1. Clone 仓库

```bash
cd ~/husicheng
git clone https://github.com/Hsch22/MClaw.git
git clone https://github.com/Hsch22/AgentGym-RL.git
cd AgentGym-RL && git submodule update --init && cd ..
```

### 2. 安装 CUDA Toolkit 12.4

flash-attn 编译需要 nvcc。如果机器上没有 CUDA Toolkit（`which nvcc` 无输出），按以下方式安装：

**方式 A：从 NVIDIA 官网安装（推荐，需要 sudo）**
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent --toolkitpath=/path/to/cuda-12.4
```

**方式 B：无 sudo，用 .run 安装到用户目录**
```bash
sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent --toolkitpath=$HOME/cuda-12.4 --defaultroot=$HOME/cuda-12.4
```

**方式 C：conda 安装（如果网络通畅）**
```bash
conda install -c nvidia cuda-toolkit=12.4
export CUDA_HOME=$CONDA_PREFIX
```

安装后设置环境变量：
```bash
export CUDA_HOME=/path/to/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
nvcc -V  # 验证：应输出 release 12.4
```

### 3. 安装 train 环境

```bash
cd ~/husicheng/MClaw

# 如果 /tmp 空间不足（No usable temporary directory），指定临时目录
export TMPDIR=/mnt/kangshijia/husicheng/tmp
mkdir -p $TMPDIR

# 安装（flash-attn 已从 pyproject.toml 中注释掉，不会自动编译）
UV_HTTP_TIMEOUT=12000 uv sync --group train
```

### 4. 编译安装 flash-attn

flash-attn 必须从源码编译，且需要匹配 torch 的 CXX11 ABI：

```bash
# 确认 torch 的 ABI 设置
source .venv/bin/activate
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
# 输出 False → 用 -D_GLIBCXX_USE_CXX11_ABI=0
# 输出 True  → 用 -D_GLIBCXX_USE_CXX11_ABI=1

# 编译安装（约 5-15 分钟）
TORCH_CUDA_ARCH_LIST="8.0" \
CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
MAX_JOBS=4 \
uv pip install --no-deps --no-cache flash-attn==2.8.3 --no-build-isolation
```

> **关键**：`CXXFLAGS` 中的 ABI 值必须与 torch 一致，否则会报 `undefined symbol` 错误。
> 预编译 wheel 在 Python 3.12 + torch 2.6 下存在 ABI 不兼容问题，不建议使用。

验证：
```bash
python -c "from flash_attn.bert_padding import pad_input; print('flash_attn ok')"
```

### 5. 设置 PYTHONPATH 并验证

```bash
export PYTHONPATH="$HOME/husicheng/AgentGym-RL/AgentGym-RL:$PYTHONPATH"
export DS_BUILD_OPS=0  # 跳过 deepspeed 的 CUDA op 编译检查

python -c "
import torch; print(f'torch={torch.__version__}')
import vllm; print('vllm ok')
import xformers; print('xformers ok')
import flash_attn; print(f'flash_attn={flash_attn.__version__}')
import verl; print('verl ok')
"
```

### 6. 下载模型和数据

```bash
# 模型（从 ModelScope 下载，约 8GB）
pip install modelscope  # 或在单独的 conda 环境中
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY  # 取消代理加速下载
modelscope download --model Qwen/Qwen3-4B --local_dir ~/husicheng/AgentGym-RL/models/Qwen3-4B-Instruct-2507

# 训练数据（约 8MB，可从已有机器 scp）
scp -r user@source_machine:/path/to/AgentGym-RL-Data-ID ~/husicheng/
```

### 7. 运行 Smoke Test

**终端 1 — 环境服务器：**
```bash
cd ~/husicheng/MClaw
source .venv/bin/activate
export ENVSERVER_PYTHON="$HOME/husicheng/MClaw/.venv/bin/python"
export AGENTGYM_DIR="$HOME/husicheng/AgentGym-RL/AgentGym"
# envserver 需要 gymnasium，如未安装：
# uv pip install gymnasium agentenv-textcraft@git+https://github.com/WooooDyy/AgentGym.git@640f8bca#subdirectory=agentenv-textcraft
bash examples/smoke_test.sh env
```

**终端 2 — 训练：**
```bash
cd ~/husicheng/MClaw
source .venv/bin/activate
export TMPDIR=/mnt/kangshijia/husicheng/tmp
export CUDA_HOME=/path/to/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export DS_BUILD_OPS=0
export TRAIN_PYTHON="$HOME/husicheng/MClaw/.venv/bin/python"
export AGENTGYM_RL_SRC="$HOME/husicheng/AgentGym-RL/AgentGym-RL"
export MODEL_PATH="$HOME/husicheng/AgentGym-RL/models/Qwen3-4B-Instruct-2507"
export DATA_FILE="$HOME/husicheng/AgentGym-RL-Data-ID/train/textcraft_train.json"
export CUDA_VISIBLE_DEVICES=0  # 选择空闲 GPU
bash examples/smoke_test.sh train
```

---

## 踩坑记录与解决方案

### 1. uv 依赖解析：torch-tb-profiler 版本冲突

**现象**：`torch-tb-profiler>=0.4.3` 在 PyTorch CUDA 索引上只有 0.1.0。

**原因**：uv 默认只从第一个包含该包的索引获取版本（防止依赖混淆攻击）。

**解决**：`pyproject.toml` 中设置 `index-strategy = "unsafe-best-match"`。

### 2. Python 3.13 不兼容

**现象**：`xformers` 没有 cp313 的预编译 wheel。

**解决**：`pyproject.toml` 限制 `requires-python = ">=3.10,<3.13"`，并在 `.python-version` 中固定 `3.12`。

### 3. flash-attn 编译需要 CUDA Toolkit

**现象**：`CUDA_HOME environment variable is not set` 或 `nvcc not found`。

**原因**：NVIDIA 驱动 ≠ CUDA Toolkit。驱动只提供运行时，nvcc 编译器在 Toolkit 中。

**解决**：安装 CUDA Toolkit 12.4（见上方步骤 2）。

### 4. flash-attn ABI 不匹配（undefined symbol）

**现象**：
```
ImportError: flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2E...
```

**原因**：flash-attn 编译时的 `_GLIBCXX_USE_CXX11_ABI` 与 torch 不一致。预编译 wheel 在 Python 3.12 + torch 2.6.0+cu124 下普遍有此问题。

**解决**：从源码编译时显式设置 `CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"`（值与 `torch._C._GLIBCXX_USE_CXX11_ABI` 一致）。

### 5. transformers 版本冲突（train vs envserver）

**现象**：`transformers==5.x` 与 `vllm==0.8.5` 不兼容（dataclass 初始化报错）。

**原因**：train 组需要 `transformers==4.51.3`，envserver 需要 `transformers==5.4.0`。uv 默认同时解析所有依赖组。

**解决**：
- 在 `pyproject.toml` 中写死各组的 transformers 版本
- 使用 `[tool.uv] conflicts` 声明 train 和 envserver 互斥

### 6. deepspeed 导入时检查 CUDA_HOME

**现象**：`MissingCUDAException: CUDA_HOME does not exist`（transformers 导入 deepspeed 触发）。

**解决**：设置 `export DS_BUILD_OPS=0` 跳过检查；同时设置 `CUDA_HOME` 指向已安装的 toolkit（或 pip 包中的 nvidia cuda_runtime 目录 + 假 nvcc）。

### 7. /home 磁盘空间不足

**现象**：`No space left on device` 或 `No usable temporary directory`。

**解决**：
- 将工作目录迁移到 `/mnt`：`mv ~/husicheng /mnt/kangshijia/husicheng && ln -s /mnt/kangshijia/husicheng ~/husicheng`
- 指定临时目录：`export TMPDIR=/mnt/kangshijia/husicheng/tmp`

### 8. conda 镜像源失效

**现象**：`UnavailableInvalidChannel: HTTP 404 Not Found for channel anaconda/pkgs/msys2`

**解决**：覆盖 `~/.condarc`：
```bash
echo -e "channels:\n  - nvidia\n  - defaults" > ~/.condarc
```

### 9. 版本漂移（跨机器环境不一致）

**现象**：不同时间 `uv sync` 解析出不同版本（如 transformers 4.x vs 5.x）。

**解决**：在 `pyproject.toml` 中写死所有关键依赖的精确版本（`==`），不使用范围约束。

---

## 环境 2: mclaw-envserver (环境服务器节点)

```bash
cd /path/to/MClaw
uv sync --only-group envserver
```

> envserver 和 train 使用不同的 transformers 版本，不能装在同一个 venv 中。
> 如果在同一个 venv 中运行 env server（如 smoke test），需要额外安装：
> `uv pip install gymnasium agentenv-textcraft@git+https://github.com/WooooDyy/AgentGym.git@640f8bca#subdirectory=agentenv-textcraft`

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
