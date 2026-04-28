import sys, os, time
sys.path.insert(0, '..')
from petri_net_io.utils.net_loader import load_petri_net_context, build_ttpn_with_residence
from petri_gcn_ppo_4_1 import PetriNetGCNPPOPro

print('Loading net file...')
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

print('Testing greedy search...')
s = PetriNetGCNPPOPro(petri_net=pn, end=ctx['end'], pre=ctx['pre'], post=ctx['post'], 
    min_delay_p=ctx['min_delay_p'], env_pool=[env], max_train_steps=0, verbose=False, 
    beam_depth=100, search_strategy='greedy')
s.is_trained = True

t1 = time.perf_counter()
r = s.search(strategy='greedy')
e1 = time.perf_counter() - t1
extra = s.get_extra_info()
print(f'Greedy: time={e1:.4f}s, trans={len(r.get_trans())}, makespan={extra.get("inferenceMakespan")}, strategy={extra.get("searchStrategy")}')

print('Testing beam search...')
t2 = time.perf_counter()
r2 = s.search(strategy='beam')
e2 = time.perf_counter() - t2
extra2 = s.get_extra_info()
print(f'Beam: time={e2:.4f}s, trans={len(r2.get_trans())}, makespan={extra2.get("inferenceMakespan")}, strategy={extra2.get("searchStrategy")}')

print(f'Speed: greedy is {e2/e1:.2f}x faster than beam')
print('TEST PASSED')
