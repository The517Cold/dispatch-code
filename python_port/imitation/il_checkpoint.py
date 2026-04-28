import os
from typing import List, Tuple


def normalize_il_mode(mode: str) -> str:
    value = str(mode or "").strip().lower()
    if value in {"bc", "dagger", "auto"}:
        return value
    return "auto"


def classify_il_artifact(base_dir: str, path: str, net_stem: str = "", scene_id: str = "") -> Tuple[str, str]:
    if not path:
        return "none", "none"
    full = os.path.abspath(path)
    name = os.path.basename(full)
    if net_stem:
        if name == "dagger_" + net_stem + ".pt" or name == "dagger_" + net_stem + "_result.txt":
            return "dagger", "net"
        if name == "bc_" + net_stem + ".pt" or name == "bc_" + net_stem + "_result.txt":
            return "bc", "net"
    if scene_id:
        if name == "dagger_scene_" + str(scene_id) + ".pt" or name == "dagger_scene_" + str(scene_id) + "_result.txt":
            return "dagger", "scene"
        if name == "bc_scene_" + str(scene_id) + ".pt" or name == "bc_scene_" + str(scene_id) + "_result.txt":
            return "bc", "scene"
    if name in {"dagger_pretrain_latest.pt", "dagger_pretrain_result.txt"}:
        return "dagger", "global"
    if name in {"bc_pretrain_latest.pt", "bc_pretrain_result.txt"}:
        return "bc", "global"
    if "dagger" in name.lower():
        return "dagger", "custom"
    if "bc" in name.lower():
        return "bc", "custom"
    return "custom", "custom"


def resolve_il_checkpoint(base_dir: str, mode: str, net_stem: str = "", scene_id: str = "", explicit: str = "") -> str:
    return _resolve_il_artifact(base_dir, mode, net_stem=net_stem, scene_id=scene_id, explicit=explicit, kind="checkpoint")


def resolve_il_result(base_dir: str, mode: str, net_stem: str = "", scene_id: str = "", explicit: str = "") -> str:
    return _resolve_il_artifact(base_dir, mode, net_stem=net_stem, scene_id=scene_id, explicit=explicit, kind="result")


def _resolve_il_artifact(base_dir: str, mode: str, net_stem: str, scene_id: str, explicit: str, kind: str) -> str:
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(base_dir, explicit)
    normalized = normalize_il_mode(mode)
    for _, _, candidate in _iter_candidates(base_dir, normalized, net_stem=net_stem, scene_id=scene_id, kind=kind):
        if os.path.exists(candidate):
            return candidate
    return ""


def _iter_candidates(base_dir: str, mode: str, net_stem: str, scene_id: str, kind: str) -> List[Tuple[str, str, str]]:
    methods = ["dagger", "bc"] if mode == "auto" else [mode]
    out = []
    for method in methods:
        if kind == "checkpoint":
            if net_stem:
                out.append((method, "net", os.path.join(base_dir, "checkpoints", method + "_" + net_stem + ".pt")))
            if scene_id:
                out.append((method, "scene", os.path.join(base_dir, "checkpoints", method + "_scene_" + str(scene_id) + ".pt")))
            out.append((method, "global", os.path.join(base_dir, "checkpoints", method + "_pretrain_latest.pt")))
        else:
            if net_stem:
                out.append((method, "net", os.path.join(base_dir, "results", method + "_" + net_stem + "_result.txt")))
            if scene_id:
                out.append((method, "scene", os.path.join(base_dir, "results", method + "_scene_" + str(scene_id) + "_result.txt")))
            out.append((method, "global", os.path.join(base_dir, "results", method + "_pretrain_result.txt")))
    return out
