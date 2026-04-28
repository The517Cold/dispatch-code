import os
import re
from typing import List


_SCENE_NET_PATTERN = re.compile(r"^(\d+)(?:-\d+)+\.txt$")


def infer_scene_id(name_or_path: str) -> str:
    stem = os.path.basename(str(name_or_path).strip())
    match = _SCENE_NET_PATTERN.match(stem)
    if match:
        return match.group(1)
    stem_no_ext = os.path.splitext(stem)[0]
    match = re.match(r"^(\d+)(?:-\d+)+$", stem_no_ext)
    if match:
        return match.group(1)
    return ""


def list_dash_net_files(resources_dir: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(resources_dir):
        for name in sorted(files):
            if _SCENE_NET_PATTERN.match(name):
                out.append(os.path.join(root, name))
    out.sort()
    return out


def list_scene_net_files(resources_dir: str, scene_id: str) -> List[str]:
    scene = str(scene_id).strip()
    if not scene:
        return list_dash_net_files(resources_dir)
    out: List[str] = []
    for path in list_dash_net_files(resources_dir):
        if infer_scene_id(path) == scene:
            out.append(path)
    return out
