from klab.config import TargetConfig
from klab.targets.base import Target


def make_target(cfg: TargetConfig) -> Target:
    if cfg.kind == "ssh":
        from klab.targets.ssh import SshTarget
        return SshTarget(cfg)
    if cfg.kind == "local":
        from klab.targets.local import LocalTarget
        return LocalTarget(cfg)
    raise SystemExit(f"未实现的后端类型 {cfg.kind!r}(target {cfg.name})")
