import os
import sys
import json
import logging
import traceback
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_port.petri_net_io.utils.net_loader import load_petri_net_context, build_ttpn_with_residence
from python_port.petri_net_io.utils.checkpoint_selector import load_compatible_state
from python_port.petri_net_platform.search.petri_net_gcn_ppo import PetriNetGCNPPOEnhancedHQ
from python_port.petri_net_platform.utils.result import Result
from python_port.scene_utils import list_scene_net_files

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BatchInferenceEvaluator")


@dataclass
class InferenceResult:
    method: str
    success: bool
    makespan: float
    trans_count: int
    trans_sequence: str
    steps: int
    error: Optional[str] = None
    stop_reason: Optional[str] = None
    deadlock_reason: Optional[str] = None


@dataclass
class FileInferenceOutput:
    file_name: str
    file_path: str
    status: str
    il_result: Optional[Dict[str, Any]] = None
    il_rl_result: Optional[Dict[str, Any]] = None
    optimal_method: Optional[str] = None
    optimal_metrics: Optional[Dict[str, Any]] = None
    error_msg: Optional[str] = None


def _compute_step_schedule(context: Dict[str, Any], expert_steps: int = 0) -> Dict[str, Any]:
    place_count = len(context["p_info"])
    pre = context["pre"]
    trans_count = len(pre[0]) if pre else 0
    constrained_count = 0
    for val in context["max_residence_time"]:
        if val < 2 ** 31 - 1:
            constrained_count += 1
    complexity = max(place_count, trans_count)
    heuristic_min_steps = min(220, max(120, 90 + complexity))
    heuristic_max_steps = min(900, max(heuristic_min_steps + 260, 480 + complexity * 8 + constrained_count * 6))
    
    min_steps = heuristic_min_steps
    max_steps = heuristic_max_steps
    step_reference_source = "heuristic"
    
    if expert_steps > 0:
        min_steps = max(24, int(round(float(expert_steps) * 0.75)))
        max_steps = max(48, min_steps + 24, int(round(float(expert_steps) * 1.80)))
        step_reference_source = "expert"
    
    inference_max_steps = max_steps
    if expert_steps > 0:
        inference_max_steps = max(expert_steps + 16, int(round(float(expert_steps) * 2.0)))
    
    return {
        "min_steps": min_steps,
        "max_steps": max_steps,
        "inference_max_steps": inference_max_steps,
        "step_reference_source": step_reference_source,
        "place_count": place_count,
        "trans_count": trans_count,
        "constrained_count": constrained_count,
    }


def _build_search_instance(
    context: Dict[str, Any],
    petri_net: Any,
    schedule: Dict[str, Any],
    verbose: bool = False,
) -> Any:
    return PetriNetGCNPPOEnhancedHQ(
        petri_net=petri_net,
        end=context["end"],
        pre=context["pre"],
        post=context["post"],
        min_delay_p=context["min_delay_p"],
        train_iterations=1,
        rollout_episodes_per_iter=1,
        ppo_update_epochs=1,
        min_steps_per_episode=schedule["min_steps"],
        max_steps_per_episode=schedule["max_steps"],
        inference_max_steps_per_episode=schedule["inference_max_steps"],
        goal_eval_rollouts=1,
        goal_min_success_rate=0.5,
        extra_train_iterations=0,
        use_reward_scaling=True,
        reward_time_scale=1000.0,
        use_reward_clip=True,
        reward_clip_abs=20.0,
        verbose=verbose,
        log_interval=1,
        controller_representation_enabled=True,
    )


def _run_single_inference(
    context: Dict[str, Any],
    petri_net: Any,
    schedule: Dict[str, Any],
    actor_state: Dict[str, Any],
    critic_state: Dict[str, Any],
    method_name: str,
    verbose: bool = False,
) -> InferenceResult:
    result = InferenceResult(
        method=method_name,
        success=False,
        makespan=float('inf'),
        trans_count=0,
        trans_sequence="",
        steps=0,
    )
    
    try:
        search = _build_search_instance(context, petri_net, schedule, verbose=verbose)
        
        if actor_state:
            load_compatible_state(search.model.actor_net, actor_state)
        if critic_state:
            load_compatible_state(search.model.value_head, critic_state)
        
        search.is_trained = True
        
        inference_result = search.search()
        extra_info = search.get_extra_info()
        
        trans = inference_result.get_trans() if inference_result else []
        markings = inference_result.get_markings() if inference_result else []
        
        result.success = bool(extra_info.get("reachGoal", False))
        result.trans_count = len(trans)
        result.trans_sequence = "->".join(str(t) for t in trans) if trans else ""
        result.steps = len(trans)
        
        if markings:
            result.makespan = markings[-1].get_prefix()
        else:
            result.makespan = -1
        
        result.stop_reason = str(extra_info.get("inferenceStopReason", ""))
        result.deadlock_reason = str(extra_info.get("inferenceDeadlockReason", ""))
        
    except Exception as e:
        result.error = f"{method_name} inference error: {str(e)}"
        logger.error(f"{method_name} inference failed: {e}\n{traceback.format_exc()}")
    
    return result


def _compare_results(il_res: InferenceResult, rl_res: InferenceResult) -> str:
    if il_res.success and not rl_res.success:
        return "IL"
    if not il_res.success and rl_res.success:
        return "IL+RL"
    if not il_res.success and not rl_res.success:
        return "Tie (Both Failed)"
    if il_res.makespan < rl_res.makespan:
        return "IL"
    if rl_res.makespan < il_res.makespan:
        return "IL+RL"
    if il_res.trans_count < rl_res.trans_count:
        return "IL"
    if rl_res.trans_count < il_res.trans_count:
        return "IL+RL"
    return "Tie (Identical Performance)"


class BatchInferenceEvaluator:
    def __init__(
        self,
        il_checkpoint_path: str,
        il_rl_checkpoint_path: str,
        max_workers: int = 4,
        verbose: bool = False,
    ):
        self.il_checkpoint_path = il_checkpoint_path
        self.il_rl_checkpoint_path = il_rl_checkpoint_path
        self.max_workers = max_workers
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.il_actor_state: Dict[str, Any] = {}
        self.il_critic_state: Dict[str, Any] = {}
        self.il_rl_actor_state: Dict[str, Any] = {}
        self.il_rl_critic_state: Dict[str, Any] = {}
        
        self._load_checkpoints()
    
    def _load_checkpoints(self):
        self.il_actor_state, self.il_critic_state = self._load_checkpoint_states(
            self.il_checkpoint_path, "IL"
        )
        self.il_rl_actor_state, self.il_rl_critic_state = self._load_checkpoint_states(
            self.il_rl_checkpoint_path, "IL+RL"
        )
    
    def _load_checkpoint_states(
        self, 
        checkpoint_path: str, 
        model_type: str
    ) -> tuple:
        actor_state: Dict[str, Any] = {}
        critic_state: Dict[str, Any] = {}
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logger.warning(f"Warning: {model_type} checkpoint not found ({checkpoint_path})")
            return actor_state, critic_state
        
        try:
            logger.info(f"Loading {model_type} checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            if isinstance(checkpoint, dict):
                actor_state = (
                    checkpoint.get("actor_state")
                    or checkpoint.get("model_state")
                    or checkpoint.get("policy_state")
                    or {}
                )
                critic_state = checkpoint.get("critic_state", {})
            
            logger.info(f"Successfully loaded {model_type} checkpoint")
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} checkpoint: {e}")
        
        return actor_state, critic_state
    
    def _process_single_file(self, file_path: str) -> FileInferenceOutput:
        file_name = os.path.basename(file_path)
        logger.info(f"Processing file: {file_name}")
        
        output = FileInferenceOutput(
            file_name=file_name,
            file_path=file_path,
            status="Processed",
        )
        
        if not os.path.isfile(file_path):
            output.status = "Failed"
            output.error_msg = f"File not found: {file_path}"
            return output
        
        try:
            context = load_petri_net_context(file_path)
            petri_net = build_ttpn_with_residence(context)
            schedule = _compute_step_schedule(context)
            
            il_result = _run_single_inference(
                context=context,
                petri_net=petri_net.clone(),
                schedule=schedule,
                actor_state=self.il_actor_state,
                critic_state=self.il_critic_state,
                method_name="IL",
                verbose=self.verbose,
            )
            
            il_rl_result = _run_single_inference(
                context=context,
                petri_net=petri_net.clone(),
                schedule=schedule,
                actor_state=self.il_rl_actor_state,
                critic_state=self.il_rl_critic_state,
                method_name="IL+RL",
                verbose=self.verbose,
            )
            
            output.il_result = asdict(il_result)
            output.il_rl_result = asdict(il_rl_result)
            
            best_method = _compare_results(il_result, il_rl_result)
            output.optimal_method = best_method
            
            if best_method == "IL":
                output.optimal_metrics = {
                    "makespan": il_result.makespan,
                    "trans_count": il_result.trans_count,
                }
            elif best_method == "IL+RL":
                output.optimal_metrics = {
                    "makespan": il_rl_result.makespan,
                    "trans_count": il_rl_result.trans_count,
                }
            elif "Tie" in best_method and il_result.success:
                output.optimal_metrics = {
                    "makespan": il_result.makespan,
                    "trans_count": il_result.trans_count,
                }
            
        except Exception as e:
            output.status = "Error"
            output.error_msg = f"Processing error: {str(e)}"
            logger.error(f"Error processing {file_name}: {e}\n{traceback.format_exc()}")
        
        logger.info(f"Completed: {file_name} | Optimal: {output.optimal_method}")
        return output
    
    def process_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, path): path 
                for path in file_paths
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    res = future.result()
                    results.append(asdict(res))
                except Exception as exc:
                    logger.error(f"Uncaught exception for {file_path}: {exc}")
                    results.append({
                        "file_name": os.path.basename(file_path),
                        "file_path": file_path,
                        "status": "Fatal Error",
                        "error_msg": str(exc),
                    })
        
        return results
    
    def process_scene(
        self, 
        resources_dir: str, 
        scene_id: str = "",
        net_limit: int = 0,
    ) -> List[Dict[str, Any]]:
        net_files = list_scene_net_files(resources_dir, scene_id)
        if net_limit > 0:
            net_files = net_files[:net_limit]
        
        if not net_files:
            logger.warning(f"No network files found for scene_id={scene_id}")
            return []
        
        logger.info(f"Found {len(net_files)} network files for processing")
        return self.process_files(net_files)


def main():
    base_dir = os.path.dirname(__file__)
    resources_dir = os.path.join(base_dir, "resources")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(results_dir, exist_ok=True)
    
    il_ckpt = os.path.join(checkpoints_dir, "bc_scene_1.pt")
    il_rl_ckpt = os.path.join(checkpoints_dir, "ppo_scene_1.pt")
    
    test_files = [
        os.path.join(resources_dir, "1-1-4.txt"),
        os.path.join(resources_dir, "1-1-9.txt"),
        os.path.join(resources_dir, "1-1-11.txt"),
        os.path.join(resources_dir, "1-1-14.txt"),
        os.path.join(resources_dir, "1-2-4.txt"),
        os.path.join(resources_dir, "1-2-9.txt"),
        os.path.join(resources_dir, "1-2-11.txt"),
        os.path.join(resources_dir, "1-2-14.txt"),
        os.path.join(resources_dir, "1-3-4.txt"),
        os.path.join(resources_dir, "1-3-9.txt"),
        os.path.join(resources_dir, "1-3-11.txt"),
        os.path.join(resources_dir, "1-3-14.txt"),
    ]
    
    valid_test_files = [f for f in test_files if os.path.exists(f)]
    if not valid_test_files:
        logger.warning("Test files not found, using scene files instead")
        valid_test_files = list_scene_net_files(resources_dir, "1")[:2]
    
    if not valid_test_files:
        logger.error("No valid network files found for testing")
        return
    
    evaluator = BatchInferenceEvaluator(
        il_checkpoint_path=il_ckpt,
        il_rl_checkpoint_path=il_rl_ckpt,
        max_workers=2,
        verbose=False,
    )
    
    final_results = evaluator.process_files(valid_test_files)
    
    output_path = os.path.join(results_dir, "batch_inference_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("Batch Inference Results Report")
    print("=" * 60)
    print(f"Output saved to: {output_path}")
    print("-" * 60)
    
    for res in final_results:
        print(f"\nFile: {res['file_name']}")
        print(f"  Status: {res['status']}")
        if res.get('optimal_method'):
            print(f"  Optimal Method: {res['optimal_method']}")
            if res.get('optimal_metrics'):
                print(f"  Makespan: {res['optimal_metrics'].get('makespan')}")
                print(f"  Trans Count: {res['optimal_metrics'].get('trans_count')}")
        if res.get('error_msg'):
            print(f"  Error: {res['error_msg']}")


if __name__ == "__main__":
    main()