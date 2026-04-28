import sys, os, time
sys.path.insert(0, '..')
from petri_net_io.utils.net_loader import load_petri_net_context, build_ttpn_with_residence
from petri_gcn_ppo_4_1 import PetriNetGCNPPOPro

net_path = 'resources/resources_new/resources/1-1-9.txt'
ctx = load_petri_net_context(net_path)
pn = build_ttpn_with_residence(ctx)
env = {
    'petri_net': pn,
    'initial_marking': pn.get_marking().clone(),
    'end': ctx['end'],
    'pre': ctx['pre'],
    'post': ctx['post'],
    'min_delay_p': ctx['min_delay_p'],
    'max_residence_time': ctx['max_residence_time'],
    'name': 'test',
    'path': net_path,
    'context': ctx,
    'complexity_score': 1.0,
}

print('=== Regression Test: Sequential Mode (mixed_rollout=False) ===')
s = PetriNetGCNPPOPro(petri_net=pn, end=ctx['end'], pre=ctx['pre'], post=ctx['post'],
    min_delay_p=ctx['min_delay_p'], env_pool=[env], max_train_steps=100, verbose=True,
    beam_depth=50, mixed_rollout=False, dynamic_curriculum=False)
s.train_model()
extra = s.get_extra_info()
print(f'trainSteps: {extra.get("trainSteps")}')
print(f'bestTrainMakespan: {extra.get("bestTrainMakespan")}')
print(f'is_trained: {s.is_trained}')
print('REGRESSION TEST PASSED')
