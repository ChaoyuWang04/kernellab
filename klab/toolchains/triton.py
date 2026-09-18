"""Triton 工具链:uv 建 venv,torch 从 PyTorch 官方 CUDA 索引装,其余从 PyPI。"""
from __future__ import annotations

from pathlib import Path

NAME = "triton"


def setup_script(root: str, python: str, repo_dir: str) -> str:
    env = f"{root}/envs/{NAME}"
    req = f"{repo_dir}/envs/{NAME}/requirements.txt"
    index = f"{repo_dir}/envs/{NAME}/torch-index.txt"
    return f"""
set -e
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] 安装 uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"
fi
mkdir -p {root}/envs {root}/runs
if [ ! -x {env}/bin/python ]; then
  echo "[setup] 建 venv {env} (python {python})"; uv venv --python {python} {env}
fi
IDX=$(head -1 {index})
echo "[setup] 安装 torch(索引 $IDX)"
uv pip install --python {env}/bin/python --index-url "$IDX" torch
echo "[setup] 安装其余依赖"
uv pip install --python {env}/bin/python -r {req}
{env}/bin/python - <<'PY'
import torch, triton
print("[setup] torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
print("[setup] device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
"""
