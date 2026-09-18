"""纯 pip 工具链的通用装法:uv venv + torch(官方 CUDA 索引)+ envs/<name>/requirements.txt。"""
from __future__ import annotations


def setup_script(name: str, root: str, python: str, repo_dir: str) -> str:
    env = f"{root}/envs/{name}"
    req = f"{repo_dir}/envs/{name}/requirements.txt"
    index = f"{repo_dir}/envs/{name}/torch-index.txt"
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
echo "[setup:{name}] 安装 torch(索引 $IDX)"
uv pip install --python {env}/bin/python --index-url "$IDX" torch
echo "[setup:{name}] 安装其余依赖"
uv pip install --python {env}/bin/python -r {req}
{env}/bin/python - <<'PY'
import importlib, torch
print("[setup] torch", torch.__version__, "cuda", torch.version.cuda)
for m in ("triton", "tilelang"):
    try:
        print("[setup]", m, importlib.import_module(m).__version__)
    except Exception as e:
        print("[setup]", m, "未装:", type(e).__name__)
print("[setup] device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
"""
