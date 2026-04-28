import traceback
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from python_port.petri_net_io.utils.net_loader import load_petri_net_context, build_ttpn_with_residence
from python_port.petri_net_platform.search.ga import GAWithTrans
from python_port.petri_net_platform.search.ant import AntClonyOptimization


def main():
    base_dir = os.path.dirname(__file__)
    out_path = os.path.join(base_dir, "results", "ga_result.txt")
    progress_path = os.path.join(base_dir, "results", "ga_progress.txt")
    try:
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        path = os.path.join(base_dir, "resources", "1-2.txt")
        context = load_petri_net_context(path)
        end = context["end"]
        petri_net = build_ttpn_with_residence(context)
        with open(progress_path, "w", encoding="utf-8") as f:
            f.write("")
        fast_mode = os.environ.get("GA_FAST", "1") == "1"
        if fast_mode:
            round_count = 20
            ant_count = 30
            genes_count = 50
            mode_line = "GA mode: fast"
        else:
            round_count = 70
            ant_count = 300
            genes_count = 300
            mode_line = "GA mode: full"
        print(mode_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(mode_line + "\n")
        def on_ant_round(i, total, best_prefix):
            line = "ant_round " + str(i) + "/" + str(total) + " best_prefix=" + str(best_prefix)
            print(line, flush=True)
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        AntClonyOptimization.default_on_round = on_ant_round
        search = GAWithTrans(petri_net, end, round_count, ant_count, genes_count)
        print("GA started", flush=True)
        def on_iter(i, total, elapsed, makespan):
            line = "round " + str(i) + "/" + str(total) + " time=" + format(elapsed, ".3f") + "s makespan=" + str(makespan)
            print(line, flush=True)
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        search.on_iteration = on_iter
        result = search.search()
        trans = result.get_trans()
        markings = result.get_markings()
        out = "trans_count:" + str(len(trans)) + "\n" + "last_prefix:" + str(markings[-1].get_prefix()) + "\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out)
        print(out, flush=True)
    except BaseException:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("ERROR\n" + traceback.format_exc())


if __name__ == "__main__":
    main()
