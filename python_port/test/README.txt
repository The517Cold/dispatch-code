Put unseen test nets in this folder for RL inference/training input.

Example:
- 1-6.txt
- 2-6.txt

The RL entry script `run_gcn_dqn_enhanced_hq.py` now reads from `python_port/test`
by default via `DEFAULT_GCN_ENH_HQ_INPUT_SUBDIR = "test"`.
