import traceback
import os
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from python_port.petri_net_io.utils.net_loader import load_petri_net_context, build_ttpn_with_residence
from python_port.petri_net_platform.search.a_star import AStar, OpenTable, EvaluationFunction, CreateEFLine


def main():
    base_dir = os.path.dirname(__file__)
    out_path = os.path.join(base_dir, "results", "a_star_result.txt")
    progress_path = os.path.join(base_dir, "results", "a_star_progress.txt")
    try:
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        max_search_seconds = float(os.environ.get("A_STAR_MAX_SECONDS", "60"))
        build_efline = os.environ.get("A_STAR_BUILD_EFLINE", "0") == "1"
        path = os.path.join(base_dir, "resources/resources_new/train/family1", "1-5-16.txt")
        if not os.path.exists(path):
            msg = "ERROR\nmissing input file: " + path + "\n请将资源文件放到 python_port/resources"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(msg + "\n")
            print(msg, flush=True)
            return
        context = load_petri_net_context(path)
        petri_net_file = context["petri_net_file"]
        matrix_translator = context["matrix_translator"]
        p_info = context["p_info"]
        end = context["end"]
        a_matrix = context["a_matrix"]
        petri_net = build_ttpn_with_residence(context)
        with open(progress_path, "w", encoding="utf-8") as f:
            f.write("")
        start_line = "AStar started"
        print(start_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(start_line + "\n")
        if build_efline and not petri_net_file.EFline:
            is_resource = matrix_translator.sets.get("isResource")
            if is_resource is None:
                is_resource = [False] * len(p_info)
            try:
                creator = CreateEFLine(petri_net.clone(), end.copy(), p_info, is_resource.copy(), [])
                ef_line = creator.ef_line(a_matrix, matrix_translator.p_map_v)
                if ef_line:
                    petri_net_file.EFline = ef_line
                    ef_msg = "EFline generated"
                    print(ef_msg, flush=True)
                    with open(progress_path, "a", encoding="utf-8") as f:
                        f.write(ef_msg + "\n")
            except BaseException:
                pass
        elif not build_efline and not petri_net_file.EFline:
            skip_msg = "EFline skipped"
            print(skip_msg, flush=True)
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(skip_msg + "\n")
        open_table = None
        if petri_net_file.EFline:
            evaluation_function = EvaluationFunction(petri_net_file)
            open_table = OpenTable(a_matrix, evaluation_function)
        # 默认启用限时搜索：先保住一个可行解，再在时间预算内尽量继续改进。
        search = AStar(petri_net, end, open_table, max_search_seconds=max_search_seconds)
        start_time = time.perf_counter()
        result = search.search()
        elapsed = time.perf_counter() - start_time
        if result is None:
            end_line = "AStar finished: NO_RESULT"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("NO_RESULT\n")
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(end_line + "\n")
            print(end_line, flush=True)
            return
        trans = result.get_trans()
        trans_line = "trans:" + "->".join(str(t) for t in trans) if trans else "trans:"
        markings = result.get_markings()
        makespan = markings[-1].get_prefix() if markings else 0
        extra_info = search.get_extra_info()
        out = "elapsed:" + format(elapsed, ".6f") + "s\n" + trans_line + "\n" + "makespan:" + str(makespan) + "\n" + "extra:" + str(extra_info) + "\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(out)
            f.write("AStar finished\n")
        print(out, flush=True)
    except BaseException:
        err = "ERROR\n" + traceback.format_exc()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(err)
        print(err, flush=True)


if __name__ == "__main__":
    main()
