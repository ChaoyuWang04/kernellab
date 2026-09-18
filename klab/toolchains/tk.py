"""ThunderKittens 工具链:header-only C++20 库,编译方式同 cuda 工具链,多一个 include 路径。

TK 源码位置由环境变量 KLAB_TK_ROOT 指定:
  SSH 后端  Mac 上克隆到仓库的 envs/tk/ThunderKittens(gitignore),随 rsync 同步;Target.env_prefix 指向 <repo>/envs/tk/ThunderKittens
  Modal     镜像构建时从 github 克隆到 /opt/ThunderKittens,镜像 env 里写死
"""
from klab.toolchains._pip import setup_script as _pip_setup

NAME = "tk"
TK_GIT = "https://github.com/HazyResearch/ThunderKittens.git"
APT = ["git", "build-essential"]
MODAL_RUN_COMMANDS = [f"git clone --depth 1 {TK_GIT} /opt/ThunderKittens"]
MODAL_ENV = {"KLAB_TK_ROOT": "/opt/ThunderKittens"}


def local_prepare(root) -> None:
    """SSH 后端不一定能连 github(5090home 就不能),所以在 Mac 上克隆一份放进 envs/tk/,随仓库 rsync 过去。"""
    import subprocess
    from pathlib import Path

    dst = Path(root) / "envs" / NAME / "ThunderKittens"
    if (dst / "include" / "kittens.cuh").exists():
        subprocess.run(["git", "-C", str(dst), "pull", "--ff-only", "-q"], check=False)
        return
    print(f"[setup:tk] 在本机克隆 ThunderKittens 到 {dst}")
    subprocess.run(["git", "clone", "--depth", "1", TK_GIT, str(dst)], check=True)


def setup_script(root: str, python: str, repo_dir: str) -> str:
    tk = f"{repo_dir}/envs/{NAME}/ThunderKittens"
    check = f"""
test -f {tk}/include/kittens.cuh && echo "[setup:tk] ThunderKittens 已随仓库同步到 {tk}" || echo "[setup:tk] 缺 {tk},本机 klab setup 会先克隆"
echo "[setup:tk] gcc $(gcc --version | head -1)"
"""
    return _pip_setup(NAME, root, python, repo_dir) + check
