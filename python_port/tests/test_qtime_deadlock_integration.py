from __future__ import annotations

import sys
import os
import unittest
from collections import deque
from typing import List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from petri_net_platform.search.deadlock_controller import DeadlockController, DeadlockAnalysis
from petri_net_platform.representation.features import PetriStateFeatureEncoder, _safe_log1p
from petri_net_platform.representation.graph import PetriNetGraph


class _Token:
    def __init__(self, token_id, timer, residence_time, tran_delay=0, last_tran_delay=0):
        self._id = token_id
        self.timer = timer
        self.residence_time = residence_time
        self.tran_delay = tran_delay
        self.last_tran_delay = last_tran_delay

    def get_id(self):
        return self._id


class _MockMarking:
    def __init__(self, p_info, prefix=0, qtime_map=None, over_max_residence_time=False, t_info=None, curr_delay_t=None):
        self._p_info = list(p_info)
        self.prefix = prefix
        self.qtime_map = qtime_map if qtime_map is not None else {}
        self.over_max_residence_time = over_max_residence_time
        self.t_info = t_info if t_info is not None else [deque() for _ in p_info]
        self.curr_delay_t = curr_delay_t if curr_delay_t is not None else []
        self.is_enable = []
        self.residence_time_info = []
        self._deadlock_controller_cache = None

    def get_p_info(self):
        return list(self._p_info)

    def get_prefix(self):
        return self.prefix

    def is_over_residece_time(self):
        return self.over_max_residence_time


class _MockPetriNet:
    def __init__(self, marking, trans_count, enabled_transitions=None):
        self._marking = marking
        self._trans_count = trans_count
        self._enabled = enabled_transitions if enabled_transitions is not None else list(range(trans_count))

    def get_marking(self):
        return self._marking

    def set_marking(self, marking):
        self._marking = marking

    def get_trans_count(self):
        return self._trans_count

    def enable(self, tran):
        return tran in self._enabled

    def launch(self, tran):
        new_p_info = list(self._marking.get_p_info())
        new_qtime_map = dict(self._marking.qtime_map)
        new_prefix = self._marking.prefix + 1
        new_t_info = [deque(tokens) for tokens in self._marking.t_info]
        return _MockMarking(
            new_p_info,
            prefix=new_prefix,
            qtime_map=new_qtime_map,
            t_info=new_t_info,
            curr_delay_t=self._marking.curr_delay_t,
        )


PRE = [[1, 0], [0, 1]]
POST = [[0, 1], [1, 0]]
END = [0, 0]
TWO_TRANS = 2


class TestDeadlockAnalysisQtimeFields(unittest.TestCase):
    def test_default_qtime_fields(self):
        analysis = DeadlockAnalysis(
            enabled_actions=[0, 1],
            safe_actions=[0],
            controller_actions=[0],
            hard_blocked_actions=[1],
            soft_risk_actions=[],
            reason_by_action={1: "immediate_deadlock"},
            fbm_candidate=True,
            state_deadlock=False,
            state_deadlock_reason="alive",
            controller_empty_fallback=False,
            controller_source="rule_safe",
            lookahead_ran=False,
        )
        self.assertFalse(analysis.qtime_warning)
        self.assertEqual(analysis.qtime_min_margin, float("inf"))

    def test_qtime_warning_set(self):
        analysis = DeadlockAnalysis(
            enabled_actions=[0],
            safe_actions=[],
            controller_actions=[0],
            hard_blocked_actions=[],
            soft_risk_actions=[],
            reason_by_action={},
            fbm_candidate=False,
            state_deadlock=False,
            state_deadlock_reason="alive",
            controller_empty_fallback=False,
            controller_source="rule_safe",
            lookahead_ran=False,
            qtime_warning=True,
            qtime_min_margin=-5.0,
        )
        self.assertTrue(analysis.qtime_warning)
        self.assertEqual(analysis.qtime_min_margin, -5.0)


class TestDeadlockControllerQtimeWarning(unittest.TestCase):
    def _make_controller(self, enable_qtime_warning=True, qtime_warning_ratio=0.8, qtime_places=None, qtime=None):
        return DeadlockController(
            pre=PRE,
            post=POST,
            end=END,
            enable_lookahead=False,
            enable_qtime_warning=enable_qtime_warning,
            qtime_warning_ratio=qtime_warning_ratio,
            qtime_places=qtime_places,
            qtime=qtime,
        )

    def test_qtime_warning_disabled_by_default_when_no_qtime(self):
        ctrl = self._make_controller(enable_qtime_warning=True, qtime=None)
        marking = _MockMarking([1, 1], prefix=10, qtime_map={0: 0})
        net = _MockPetriNet(marking, TWO_TRANS)
        analysis = ctrl.analyze_state(net, marking)
        self.assertFalse(analysis.qtime_warning)

    def test_qtime_warning_not_triggered_when_margin_sufficient(self):
        ctrl = self._make_controller(
            enable_qtime_warning=True,
            qtime_warning_ratio=0.8,
            qtime_places=[False, True],
            qtime=100.0,
        )
        marking = _MockMarking([1, 1], prefix=10, qtime_map={0: 5.0})
        net = _MockPetriNet(marking, TWO_TRANS)
        analysis = ctrl.analyze_state(net, marking)
        self.assertFalse(analysis.qtime_warning)

    def test_qtime_warning_triggered_when_margin_negative(self):
        ctrl = self._make_controller(
            enable_qtime_warning=True,
            qtime_warning_ratio=0.8,
            qtime_places=[False, True],
            qtime=10.0,
        )
        marking = _MockMarking([1, 1], prefix=50, qtime_map={0: 5.0})
        net = _MockPetriNet(marking, TWO_TRANS)
        analysis = ctrl.analyze_state(net, marking)
        self.assertTrue(analysis.qtime_warning)
        self.assertLess(analysis.qtime_min_margin, 0)

    def test_qtime_risk_hard_block(self):
        ctrl = self._make_controller(
            enable_qtime_warning=True,
            qtime_warning_ratio=0.5,
            qtime_places=[False, True],
            qtime=10.0,
        )
        marking = _MockMarking([1, 1], prefix=50, qtime_map={0: 5.0})
        net = _MockPetriNet(marking, TWO_TRANS)
        analysis = ctrl.analyze_state(net, marking)
        blocked_reasons = list(analysis.reason_by_action.values())
        self.assertIn("qtime_risk", blocked_reasons)

    def test_no_qtime_risk_when_warning_disabled(self):
        ctrl = self._make_controller(
            enable_qtime_warning=False,
            qtime_warning_ratio=0.5,
            qtime_places=[False, True],
            qtime=10.0,
        )
        marking = _MockMarking([1, 1], prefix=50, qtime_map={0: 5.0})
        net = _MockPetriNet(marking, TWO_TRANS)
        analysis = ctrl.analyze_state(net, marking)
        blocked_reasons = list(analysis.reason_by_action.values())
        self.assertNotIn("qtime_risk", blocked_reasons)

    def test_qtime_margin_computation_empty_qtime_map(self):
        ctrl = self._make_controller(qtime=100.0)
        marking = _MockMarking([1, 1], prefix=10, qtime_map={})
        margin = ctrl._compute_qtime_margin(marking)
        self.assertEqual(margin, float("inf"))

    def test_qtime_margin_computation_with_entries(self):
        ctrl = self._make_controller(qtime=100.0)
        marking = _MockMarking([1, 1], prefix=50, qtime_map={0: 10.0, 1: 20.0})
        margin = ctrl._compute_qtime_margin(marking)
        expected = 100.0 - (50.0 - 10.0)
        self.assertAlmostEqual(margin, expected)

    def test_qtime_check_risk_with_ratio(self):
        ctrl = self._make_controller(qtime_warning_ratio=0.8, qtime=100.0)
        marking = _MockMarking([1, 1], prefix=95, qtime_map={0: 10.0})
        self.assertTrue(ctrl._check_qtime_risk(marking))

    def test_qtime_check_no_risk_below_ratio(self):
        ctrl = self._make_controller(qtime_warning_ratio=0.8, qtime=100.0)
        marking = _MockMarking([1, 1], prefix=15, qtime_map={0: 10.0})
        self.assertFalse(ctrl._check_qtime_risk(marking))

    def test_backward_compat_no_qtime_params(self):
        ctrl = DeadlockController(
            pre=PRE,
            post=POST,
            end=END,
            enable_lookahead=False,
        )
        marking = _MockMarking([1, 1], prefix=10)
        net = _MockPetriNet(marking, TWO_TRANS)
        analysis = ctrl.analyze_state(net, marking)
        self.assertFalse(analysis.qtime_warning)
        self.assertEqual(analysis.qtime_min_margin, 0.0)


class TestMarkingSignatureQtimeMap(unittest.TestCase):
    def test_signature_includes_qtime_map(self):
        ctrl = DeadlockController(pre=PRE, post=POST, end=END, enable_lookahead=False)
        marking = _MockMarking([1, 1], prefix=10, qtime_map={0: 5.0, 1: 8.0})
        sig = ctrl._marking_signature(marking)
        self.assertIn((0, 5.0), sig[-1])
        self.assertIn((1, 8.0), sig[-1])

    def test_signature_without_qtime_map(self):
        ctrl = DeadlockController(pre=PRE, post=POST, end=END, enable_lookahead=False)
        marking = _MockMarking([1, 1], prefix=10)
        delattr(marking, "qtime_map")
        sig = ctrl._marking_signature(marking)
        last_elem = sig[-1]
        self.assertNotIsInstance(last_elem, tuple)

    def test_signature_distinguishes_different_qtime_maps(self):
        ctrl = DeadlockController(pre=PRE, post=POST, end=END, enable_lookahead=False)
        m1 = _MockMarking([1, 1], prefix=10, qtime_map={0: 5.0})
        m2 = _MockMarking([1, 1], prefix=10, qtime_map={0: 8.0})
        self.assertNotEqual(ctrl._marking_signature(m1), ctrl._marking_signature(m2))


class TestFeatureEncoderQtimeInjection(unittest.TestCase):
    def _make_graph(self):
        return PetriNetGraph.from_components(
            pre=PRE,
            post=POST,
            end=END,
        )

    def test_place_features_without_qtime(self):
        graph = self._make_graph()
        encoder = PetriStateFeatureEncoder(graph)
        marking = _MockMarking([1, 1], prefix=10)
        features = encoder.encode_places(marking)
        self.assertEqual(features.shape[1], encoder.place_feature_dim)
        self.assertNotIn("is_qtime_place", encoder.place_feature_names)
        self.assertNotIn("qtime_margin", encoder.place_feature_names)

    def test_place_features_with_qtime(self):
        graph = self._make_graph()
        encoder = PetriStateFeatureEncoder(graph)
        encoder.bind_qtime(qtime_places=[False, True], qtime=100.0)
        t_info = [deque([_Token(0, 5, 0)]), deque([_Token(1, 3, 0)])]
        marking = _MockMarking([1, 1], prefix=10, qtime_map={0: 5.0, 1: 3.0}, t_info=t_info)
        features = encoder.encode_places(marking)
        self.assertIn("is_qtime_place", encoder.place_feature_names)
        self.assertIn("qtime_margin", encoder.place_feature_names)
        self.assertEqual(features.shape[1], encoder.place_feature_dim)

    def test_transition_features_without_qtime(self):
        graph = self._make_graph()
        encoder = PetriStateFeatureEncoder(graph)
        marking = _MockMarking([1, 1], prefix=10)
        features = encoder.encode_transitions(marking)
        self.assertEqual(features.shape[1], encoder.transition_feature_dim)
        self.assertNotIn("qtime_warning", encoder.transition_feature_names)
        self.assertNotIn("qtime_margin", encoder.transition_feature_names)

    def test_transition_features_with_qtime(self):
        graph = self._make_graph()
        encoder = PetriStateFeatureEncoder(graph)
        encoder.bind_qtime(qtime_places=[False, True], qtime=100.0)
        marking = _MockMarking([1, 1], prefix=10, qtime_map={0: 5.0})
        features = encoder.encode_transitions(marking)
        self.assertIn("qtime_warning", encoder.transition_feature_names)
        self.assertIn("qtime_margin", encoder.transition_feature_names)
        self.assertEqual(features.shape[1], encoder.transition_feature_dim)

    def test_qtime_margin_zero_when_no_qtime_map(self):
        graph = self._make_graph()
        encoder = PetriStateFeatureEncoder(graph)
        encoder.bind_qtime(qtime_places=[False, True], qtime=100.0)
        marking = _MockMarking([1, 1], prefix=10, qtime_map={})
        margin = encoder._compute_global_qtime_margin(marking)
        self.assertEqual(margin, 0.0)

    def test_qtime_margin_computation(self):
        graph = self._make_graph()
        encoder = PetriStateFeatureEncoder(graph)
        encoder.bind_qtime(qtime_places=[False, True], qtime=100.0)
        marking = _MockMarking([1, 1], prefix=50, qtime_map={0: 10.0, 1: 20.0})
        margin = encoder._compute_global_qtime_margin(marking)
        expected = 100.0 - (50.0 - 10.0)
        self.assertAlmostEqual(margin, expected)

    def test_place_qtime_margin_for_qtime_place(self):
        graph = self._make_graph()
        encoder = PetriStateFeatureEncoder(graph)
        encoder.bind_qtime(qtime_places=[False, True], qtime=100.0)
        t_info = [deque(), deque([_Token(1, 3, 0)])]
        marking = _MockMarking([1, 1], prefix=50, qtime_map={1: 20.0}, t_info=t_info)
        margin = encoder._compute_place_qtime_margin(marking, 1, marking.qtime_map, marking.prefix)
        expected = 100.0 - (50.0 - 20.0)
        self.assertAlmostEqual(margin, expected)

    def test_place_qtime_margin_for_non_qtime_place(self):
        graph = self._make_graph()
        encoder = PetriStateFeatureEncoder(graph)
        encoder.bind_qtime(qtime_places=[False, True], qtime=100.0)
        marking = _MockMarking([1, 1], prefix=50, qtime_map={0: 10.0})
        margin = encoder._compute_place_qtime_margin(marking, 0, marking.qtime_map, marking.prefix)
        self.assertEqual(margin, 0.0)

    def test_encode_full_pipeline_with_qtime(self):
        graph = self._make_graph()
        encoder = PetriStateFeatureEncoder(graph)
        encoder.bind_qtime(qtime_places=[False, True], qtime=100.0)
        t_info = [deque([_Token(0, 5, 0)]), deque([_Token(1, 3, 0)])]
        marking = _MockMarking([1, 1], prefix=10, qtime_map={0: 5.0, 1: 3.0}, t_info=t_info)
        result = encoder.encode(marking)
        self.assertEqual(result.place_features.shape[1], encoder.place_feature_dim)
        self.assertEqual(result.transition_features.shape[1], encoder.transition_feature_dim)

    def test_encode_full_pipeline_without_qtime(self):
        graph = self._make_graph()
        encoder = PetriStateFeatureEncoder(graph)
        marking = _MockMarking([1, 1], prefix=10)
        result = encoder.encode(marking)
        self.assertEqual(result.place_features.shape[1], encoder.place_feature_dim)
        self.assertEqual(result.transition_features.shape[1], encoder.transition_feature_dim)


class TestEdgeCases(unittest.TestCase):
    def test_qtime_zero(self):
        ctrl = DeadlockController(
            pre=PRE, post=POST, end=END,
            enable_lookahead=False,
            enable_qtime_warning=True,
            qtime=0.0,
        )
        marking = _MockMarking([1, 1], prefix=50, qtime_map={0: 5.0})
        margin = ctrl._compute_qtime_margin(marking)
        self.assertEqual(margin, float("inf"))

    def test_qtime_negative(self):
        ctrl = DeadlockController(
            pre=PRE, post=POST, end=END,
            enable_lookahead=False,
            enable_qtime_warning=True,
            qtime=-10.0,
        )
        marking = _MockMarking([1, 1], prefix=50, qtime_map={0: 5.0})
        margin = ctrl._compute_qtime_margin(marking)
        self.assertEqual(margin, float("inf"))

    def test_qtime_warning_ratio_clamped(self):
        ctrl = DeadlockController(
            pre=PRE, post=POST, end=END,
            enable_lookahead=False,
            qtime_warning_ratio=1.5,
        )
        self.assertEqual(ctrl.qtime_warning_ratio, 1.0)
        ctrl2 = DeadlockController(
            pre=PRE, post=POST, end=END,
            enable_lookahead=False,
            qtime_warning_ratio=-0.5,
        )
        self.assertEqual(ctrl2.qtime_warning_ratio, 0.0)

    def test_marking_without_qtime_map_attribute(self):
        ctrl = DeadlockController(
            pre=PRE, post=POST, end=END,
            enable_lookahead=False,
            enable_qtime_warning=True,
            qtime=100.0,
        )
        marking = _MockMarking([1, 1], prefix=10)
        delattr(marking, "qtime_map")
        margin = ctrl._compute_qtime_margin(marking)
        self.assertEqual(margin, float("inf"))

    def test_state_deadlock_analysis_has_qtime_fields(self):
        ctrl = DeadlockController(
            pre=PRE, post=POST, end=END,
            enable_lookahead=False,
        )
        marking = _MockMarking([0, 0], prefix=10)
        net = _MockPetriNet(marking, TWO_TRANS, enabled_transitions=[])
        analysis = ctrl.analyze_state(net, marking)
        self.assertTrue(analysis.state_deadlock)
        self.assertFalse(analysis.qtime_warning)
        self.assertEqual(analysis.qtime_min_margin, 0.0)

    def test_encoder_bind_qtime_updates_dimensions(self):
        graph = PetriNetGraph.from_components(pre=PRE, post=POST, end=END)
        encoder = PetriStateFeatureEncoder(graph)
        orig_place_dim = encoder.place_feature_dim
        orig_trans_dim = encoder.transition_feature_dim
        encoder.bind_qtime(qtime_places=[False, True], qtime=100.0)
        self.assertEqual(encoder.place_feature_dim, orig_place_dim + 2)
        self.assertEqual(encoder.transition_feature_dim, orig_trans_dim + 2)

    def test_encoder_bind_qtime_none_does_not_change_dims(self):
        graph = PetriNetGraph.from_components(pre=PRE, post=POST, end=END)
        encoder = PetriStateFeatureEncoder(graph)
        orig_place_dim = encoder.place_feature_dim
        orig_trans_dim = encoder.transition_feature_dim
        encoder.bind_qtime(qtime_places=None, qtime=None)
        self.assertEqual(encoder.place_feature_dim, orig_place_dim)
        self.assertEqual(encoder.transition_feature_dim, orig_trans_dim)


class TestIntegrationControllerWithFeatures(unittest.TestCase):
    def test_controller_qtime_warning_flows_to_features(self):
        graph = PetriNetGraph.from_components(pre=PRE, post=POST, end=END)
        encoder = PetriStateFeatureEncoder(graph)
        encoder.bind_qtime(qtime_places=[False, True], qtime=10.0)

        ctrl = DeadlockController(
            pre=PRE, post=POST, end=END,
            enable_lookahead=False,
            enable_qtime_warning=True,
            qtime_warning_ratio=0.5,
            qtime_places=[False, True],
            qtime=10.0,
        )

        marking = _MockMarking([1, 1], prefix=50, qtime_map={0: 5.0})
        net = _MockPetriNet(marking, TWO_TRANS)

        encoder.bind_deadlock_controller(net, ctrl)
        analysis = ctrl.analyze_state(net, marking)

        self.assertTrue(analysis.qtime_warning)

        t_info = [deque([_Token(0, 5, 0)]), deque([_Token(1, 3, 0)])]
        marking.t_info = t_info
        features = encoder.encode_transitions(marking)
        qtime_warning_idx = encoder.transition_feature_names.index("qtime_warning")
        self.assertEqual(features[0, qtime_warning_idx].item(), 1.0)

    def test_no_qtime_warning_flows_zero_to_features(self):
        graph = PetriNetGraph.from_components(pre=PRE, post=POST, end=END)
        encoder = PetriStateFeatureEncoder(graph)
        encoder.bind_qtime(qtime_places=[False, True], qtime=100.0)

        ctrl = DeadlockController(
            pre=PRE, post=POST, end=END,
            enable_lookahead=False,
            enable_qtime_warning=True,
            qtime_warning_ratio=0.8,
            qtime_places=[False, True],
            qtime=100.0,
        )

        marking = _MockMarking([1, 1], prefix=10, qtime_map={0: 5.0})
        net = _MockPetriNet(marking, TWO_TRANS)

        encoder.bind_deadlock_controller(net, ctrl)
        analysis = ctrl.analyze_state(net, marking)

        self.assertFalse(analysis.qtime_warning)

        t_info = [deque([_Token(0, 5, 0)]), deque([_Token(1, 3, 0)])]
        marking.t_info = t_info
        features = encoder.encode_transitions(marking)
        qtime_warning_idx = encoder.transition_feature_names.index("qtime_warning")
        self.assertEqual(features[0, qtime_warning_idx].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
