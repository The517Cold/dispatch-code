import argparse
import os
import re
import sys
import traceback

#python python_port/check_trains.py --net 1-1.txt --trans 2->21->2


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_port.petri_net_io.utils.net_loader import build_ttpn_with_residence, load_petri_net_context


def goal_distance(marking, end):
    p_info = marking.get_p_info()
    dist = 0
    for idx, token in enumerate(p_info):
        if end[idx] == -1:
            continue
        dist += abs(token - end[idx])
    return dist


def format_marking(marking):
    p_info = ",".join(str(v) for v in marking.get_p_info())
    return "(" + p_info + ")"


def enabled_transitions(net):
    out = []
    for tran in range(net.get_trans_count()):
        if net.enable(tran):
            out.append(tran)
    return out


def parse_sequence_text(text):
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []
    line_patterns = [
        r"(?:^|\n)\s*trans_sequence\s*:\s*([0-9\-\>\s,]+)",
        r"(?:^|\n)\s*policy_trans_sequence\s*:\s*([0-9\-\>\s,]+)",
        r"(?:^|\n)\s*expert_trans_sequence\s*:\s*([0-9\-\>\s,]+)",
        r"(?:^|\n)\s*trans\s*:\s*([0-9\-\>\s,]+)",
    ]
    for pattern in line_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            text = match.group(1)
            break
    nums = re.findall(r"\d+", text)
    return [int(x) for x in nums]


def resolve_net_path(base_dir, net_arg):
    if os.path.isabs(net_arg):
        return net_arg
    direct = os.path.join(base_dir, net_arg)
    if os.path.exists(direct):
        return direct
    resource_path = os.path.join(base_dir, "resources", net_arg)
    return resource_path


def validate_sequence(net, end, sequence):
    initial = net.get_marking().clone()
    curr = initial
    markings = [curr]
    used = []
    for step_idx, tran in enumerate(sequence):
        if tran < 0 or tran >= net.get_trans_count():
            return {
                "ok": False,
                "reason": "transition_out_of_range",
                "step_idx": step_idx,
                "transition": tran,
                "used": used,
                "markings": markings,
                "curr_marking": curr,
                "enabled": enabled_transitions(net),
            }
        if not net.enable(tran):
            return {
                "ok": False,
                "reason": "transition_not_enabled",
                "step_idx": step_idx,
                "transition": tran,
                "used": used,
                "markings": markings,
                "curr_marking": curr,
                "enabled": enabled_transitions(net),
            }
        nxt = net.launch(tran)
        net.set_marking(nxt)
        curr = nxt
        used.append(tran)
        markings.append(curr)
    return {
        "ok": True,
        "reason": "sequence_valid",
        "step_idx": len(sequence),
        "transition": -1,
        "used": used,
        "markings": markings,
        "curr_marking": curr,
        "enabled": enabled_transitions(net),
        "reach_goal": goal_distance(curr, end) == 0,
        "goal_distance": goal_distance(curr, end),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Validate whether a transition sequence is executable on a Petri-net file.")
    parser.add_argument("--net", required=True, help="Net file path or a file name under python_port/resources")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trans", help="Transition sequence text, e.g. 2->10->2->11")
    group.add_argument("--trans-file", help="A text file containing the sequence or a result file with trans_sequence:")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    base_dir = os.path.dirname(__file__)
    net_path = resolve_net_path(base_dir, args.net)
    if not os.path.exists(net_path):
        raise FileNotFoundError("missing net file: " + net_path)

    if args.trans_file:
        trans_file = args.trans_file if os.path.isabs(args.trans_file) else os.path.join(base_dir, args.trans_file)
        if not os.path.exists(trans_file):
            raise FileNotFoundError("missing transition file: " + trans_file)
        with open(trans_file, "r", encoding="utf-8") as f:
            seq_text = f.read()
    else:
        seq_text = args.trans

    sequence = parse_sequence_text(seq_text)
    if not sequence:
        raise ValueError("no transition ids found in the provided sequence input")

    context = load_petri_net_context(net_path)
    net = build_ttpn_with_residence(context)
    result = validate_sequence(net, context["end"], sequence)

    print("net_file:", net_path, sep="")
    print("sequence_length:", len(sequence), sep="")
    print("sequence:", "->".join(str(x) for x in sequence), sep="")
    print("trans_count_in_net:", net.get_trans_count(), sep="")

    if result["ok"]:
        last = result["curr_marking"]
        print("status:valid")
        print("used_length:", len(result["used"]), sep="")
        print("final_makespan:", float(last.get_prefix()), sep="")
        print("reach_goal:", bool(result["reach_goal"]), sep="")
        print("goal_distance:", int(result["goal_distance"]), sep="")
        print("final_marking:", format_marking(last), sep="")
        print("enabled_after_replay:", ",".join(str(x) for x in result["enabled"]), sep="")
        return 0

    curr = result["curr_marking"]
    print("status:invalid")
    print("reason:", result["reason"], sep="")
    print("failed_step_idx:", result["step_idx"], sep="")
    print("failed_transition:", result["transition"], sep="")
    print("valid_prefix_length:", len(result["used"]), sep="")
    print("valid_prefix_sequence:", "->".join(str(x) for x in result["used"]), sep="")
    print("current_makespan:", float(curr.get_prefix()), sep="")
    print("current_marking:", format_marking(curr), sep="")
    print("enabled_now:", ",".join(str(x) for x in result["enabled"]), sep="")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        err = "ERROR\n" + traceback.format_exc()
        print(err, flush=True)
        raise
