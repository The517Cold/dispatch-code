import sys, os, time
sys.path.insert(0, '..')
from petri_net_io.utils.net_loader import load_petri_net_context, build_ttpn_with_residence
from petri_gcn_ppo_4_1 import PetriNetGCNPPOPro

def make_env(net_file, name):
    net_path = os.path.join('resources/resources_new/resources', net_file)
    if not os.path.exists(net_path):
        net_path = os.path.join('resources', net_file)
    ctx = load_petri_net_context(net_path)
    pn = build_ttpn_with_residence(ctx)
    return {
        'petri_net': pn,
        'initial_marking': pn.get_marking().clone(),
        'end': ctx['end'],
        'pre': ctx['pre'],
        'post': ctx['post'],
        'min_delay_p': ctx['min_delay_p'],
        'max_residence_time': ctx['max_residence_time'],
        'name': name,
        'path': net_path,
        'context': ctx,
        'complexity_score': float(len(ctx['pre']) + len(ctx['pre'][0])),
    }

env1 = make_env('1-1-9.txt', 'env_1-1-9')
env2 = make_env('1-2-9.txt', 'env_1-2-9')
env3 = make_env('1-3-9.txt', 'env_1-3-9')

print('=== Test: Dynamic Curriculum + Priority Sampling ===')
s = PetriNetGCNPPOPro(
    petri_net=env1['petri_net'], end=env1['end'], pre=env1['pre'], post=env1['post'],
    min_delay_p=env1['min_delay_p'], env_pool=[env1, env2, env3],
    max_train_steps=10000, verbose=True, beam_depth=50,
    mixed_rollout=True, cross_env_gae=True, async_collection=False,
    dynamic_curriculum=True, curriculum_warmup_ratio=0.3,
    steps_per_epoch=2048,
)

# 测试优先级采样权重
print('\n--- Priority Weights at different progress ---')
for p in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
    weights = s._compute_priority_sampling_weights(p)
    print(f'  progress={p:.1f}: {dict((k, f"{v:.3f}") for k, v in weights.items())}')

s.train_model()
extra = s.get_extra_info()
print(f'\n=== Results ===')
print(f'trainSteps: {extra.get("trainSteps")}')
print(f'is_trained: {s.is_trained}')
for name, rec in s.best_records.items():
    m = rec['makespan']
    print(f'  {name}: makespan={m if m < 2**31-1 else -1}, trans_count={len(rec["trans"])}')

print('DYNAMIC CURRICULUM TEST PASSED')
