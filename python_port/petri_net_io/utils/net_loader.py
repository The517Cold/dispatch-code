"""Helpers that turn a Petri-net text file into tensors and simulator objects."""

try:
    from .file_to_object import PntTranslator
    from .object_to_petri_net_info import CustomMatrixTranslator
    from petri_net_platform.petri_net import TTPPNHasResidenceTime
except ImportError:
    from python_port.petri_net_io.utils.file_to_object import PntTranslator
    from python_port.petri_net_io.utils.object_to_petri_net_info import CustomMatrixTranslator
    from python_port.petri_net_platform.petri_net import TTPPNHasResidenceTime


def load_petri_net_context(path):
    # Parse once and keep both symbolic info and matrices for later stages.
    translator = PntTranslator.get_pnt_translator()
    translator.translate_to_petri_net_file(path)
    petri_net_file = translator.get_petri_net_file()
    matrix_translator = CustomMatrixTranslator(petri_net_file)
    matrix_translator.translate()
    vectors = matrix_translator.vectors
    sets = matrix_translator.sets
    groups = matrix_translator.groups
    p_info = vectors.get("pInfo")
    min_delay_p = vectors.get("minDelayP")
    min_delay_t = vectors.get("minDelayT")
    max_residence_time = vectors.get("maxResidenceTime")
    if max_residence_time is None:
        max_residence_time = [2 ** 31 - 1] * len(p_info)
    capacity = vectors.get("capacity")
    if capacity is None:
        capacity = vectors.get("capicity")
    end = vectors.get("end")
    pre = matrix_translator.pre
    post = matrix_translator.post
    a_matrix = matrix_translator.a_matrix
    is_resource = sets.get("isResource")
    if is_resource is None:
        is_resource = [False] * len(p_info)
    place_from_places = groups.get("placeFromPlaces")
    if place_from_places is None:
        place_from_places = [[] for _ in range(len(p_info))]
    return {
        "petri_net_file": petri_net_file,
        "matrix_translator": matrix_translator,
        "vectors": vectors,
        "sets": sets,
        "groups": groups,
        "p_info": p_info,
        "min_delay_p": min_delay_p,
        "min_delay_t": min_delay_t,
        "max_residence_time": max_residence_time,
        "capacity": capacity,
        "end": end,
        "pre": pre,
        "post": post,
        "a_matrix": a_matrix,
        "is_resource": is_resource,
        "place_from_places": place_from_places,
    }


def build_ttpn_with_residence(context):
    # The imitation pipeline uses the simulator object for both A* search and policy rollout.
    return TTPPNHasResidenceTime(
        context["p_info"],
        context["a_matrix"],
        context["pre"],
        context["min_delay_p"],
        context["min_delay_t"],
        context["max_residence_time"],
        context["capacity"],
        context["place_from_places"],
        context["is_resource"],
    )
