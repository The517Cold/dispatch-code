import os
import hashlib
import torch


def _safe_name(text):
    raw = str(text) if text is not None else ""
    chars = []
    last_is_sep = False
    for ch in raw:
        ok = ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9")
        if ok:
            chars.append(ch)
            last_is_sep = False
            continue
        if not last_is_sep:
            chars.append("-")
            last_is_sep = True
    out = "".join(chars).strip("-")
    if not out:
        return "unknown"
    return out[:48]


def build_signature(path, context):
    pre = context["pre"]
    post = context["post"]
    end = context["end"]
    min_delay_p = context["min_delay_p"]
    min_delay_t = context["min_delay_t"]
    max_residence_time = context.get("max_residence_time")
    capacity = context.get("capacity")
    payload = (
        str(pre)
        + "|"
        + str(post)
        + "|"
        + str(end)
        + "|"
        + str(min_delay_p)
        + "|"
        + str(min_delay_t)
        + "|"
        + str(max_residence_time)
        + "|"
        + str(capacity)
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    trans_count = len(pre[0]) if pre else 0
    return {
        "file": os.path.basename(path),
        "file_stem": _safe_name(os.path.splitext(os.path.basename(path))[0]),
        "place_count": len(pre),
        "trans_count": trans_count,
        "digest": digest,
    }


def build_profile(context):
    pre = context["pre"]
    post = context["post"]
    max_residence_time = context["max_residence_time"]
    capacity = context.get("capacity") or []
    place_count = len(pre)
    trans_count = len(pre[0]) if pre else 0
    constrained_count = 0
    capacity_count = 0
    for val in max_residence_time:
        if val < 2 ** 31 - 1:
            constrained_count += 1
    for val in capacity:
        if val < 2 ** 31 - 1:
            capacity_count += 1
    pre_nnz = 0
    post_nnz = 0
    for i in range(place_count):
        for j in range(trans_count):
            if pre[i][j] > 0:
                pre_nnz += 1
            if post[i][j] > 0:
                post_nnz += 1
    denom = max(1, place_count * trans_count)
    return {
        "place_count": place_count,
        "trans_count": trans_count,
        "constrained_count": constrained_count,
        "capacity_count": capacity_count,
        "pre_nnz": pre_nnz,
        "post_nnz": post_nnz,
        "pre_density": float(pre_nnz) / float(denom),
        "post_density": float(post_nnz) / float(denom),
    }


def checkpoint_dir(base_dir):
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def checkpoint_path(base_dir, prefix, digest):
    if isinstance(digest, dict):
        sig = digest
        file_stem = _safe_name(sig.get("file_stem", sig.get("file", "unknown")))
        return os.path.join(checkpoint_dir(base_dir), prefix + "_" + file_stem + "_" + sig["digest"] + ".pt")
    return os.path.join(checkpoint_dir(base_dir), prefix + "_" + str(digest) + ".pt")


def _legacy_checkpoint_path(base_dir, prefix, digest):
    return os.path.join(checkpoint_dir(base_dir), prefix + "_" + digest + ".pt")


def _norm_gap(a, b):
    return abs(float(a) - float(b)) / max(1.0, abs(float(a)), abs(float(b)))


def _profile_score(curr, other):
    score = 0.0
    score += 3.0 * _norm_gap(curr["place_count"], other.get("place_count", curr["place_count"]))
    score += 3.0 * _norm_gap(curr["trans_count"], other.get("trans_count", curr["trans_count"]))
    score += 2.0 * _norm_gap(curr["constrained_count"], other.get("constrained_count", curr["constrained_count"]))
    score += 2.0 * _norm_gap(curr["capacity_count"], other.get("capacity_count", curr["capacity_count"]))
    score += 1.0 * _norm_gap(curr["pre_nnz"], other.get("pre_nnz", curr["pre_nnz"]))
    score += 1.0 * _norm_gap(curr["post_nnz"], other.get("post_nnz", curr["post_nnz"]))
    score += 1.5 * _norm_gap(curr["pre_density"], other.get("pre_density", curr["pre_density"]))
    score += 1.5 * _norm_gap(curr["post_density"], other.get("post_density", curr["post_density"]))
    return score


def find_checkpoint(base_dir, prefix, signature, profile, allow_similar=True):
    exact = checkpoint_path(base_dir, prefix, signature)
    legacy_exact = _legacy_checkpoint_path(base_dir, prefix, signature["digest"])
    if os.path.exists(exact):
        return {"path": exact, "mode": "exact", "score": 0.0}
    if os.path.exists(legacy_exact):
        return {"path": legacy_exact, "mode": "exact", "score": 0.0}
    if not allow_similar:
        return {"path": exact, "mode": "none", "score": -1.0}
    ckpt_dir = checkpoint_dir(base_dir)
    best_path = ""
    best_score = 10 ** 9
    if not os.path.exists(ckpt_dir):
        return {"path": exact, "mode": "none", "score": -1.0}
    for name in os.listdir(ckpt_dir):
        if not name.startswith(prefix + "_") or not name.endswith(".pt"):
            continue
        path = os.path.join(ckpt_dir, name)
        try:
            saved = torch.load(path, map_location="cpu")
        except BaseException:
            continue
        saved_sig = saved.get("signature", {})
        if saved_sig.get("digest") == signature["digest"]:
            return {"path": path, "mode": "exact", "score": 0.0}
        saved_profile = saved.get("profile")
        if saved_profile is None:
            saved_profile = {
                "place_count": saved_sig.get("place_count", profile["place_count"]),
                "trans_count": saved_sig.get("trans_count", profile["trans_count"]),
                "constrained_count": profile["constrained_count"],
                "pre_nnz": profile["pre_nnz"],
                "post_nnz": profile["post_nnz"],
                "pre_density": profile["pre_density"],
                "post_density": profile["post_density"],
            }
        score = _profile_score(profile, saved_profile)
        if score < best_score:
            best_score = score
            best_path = path
    if not best_path:
        return {"path": exact, "mode": "none", "score": -1.0}
    return {"path": best_path, "mode": "similar", "score": float(best_score)}


def load_compatible_state(module, saved_state):
    if not isinstance(saved_state, dict):
        return
    current = module.state_dict()
    compatible = {}
    for key, value in saved_state.items():
        if key not in current:
            continue
        if current[key].shape != value.shape:
            continue
        compatible[key] = value
    module.load_state_dict(compatible, strict=False)
