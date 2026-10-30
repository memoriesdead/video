"""CPU fixtures for opt-in VSA graph state; GPU parity is a separate gate."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import PropertyMock, patch

import torch


RUNTIME = Path(os.environ["WANGP_TEST_RUNTIME"]).resolve()
SOURCE = RUNTIME / "models/minimax_h3/vsa/graphs.py"
MMGP = RUNTIME / "mmgp/cuda_graphs.py"


def graph_api():
    tree = ast.parse(MMGP.read_text(encoding="utf-8"))
    names = {"register_state", "release_state", "statistics"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": torch, "_recording": False, "_graphs": lambda owner: owner.manager}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MMGP), "exec"), namespace)
    api = SimpleNamespace(**{n.name: namespace[n.name] for n in nodes})
    api.keys, api.gates = [], []
    api.register_key = api.keys.append
    api.register_gate = api.gates.append
    return api


class Manager:
    def __init__(self):
        self.state, self.holders, self.state_ids, self.holder_ids = {}, {}, set(), set()
        self.entries = {"old": object()}
        self.stats = {"captures": 2, "replays": 5}
        self.disabled = None
        self.cleared = 0

    def clear(self):
        self.entries.clear()
        self.cleared += 1


class VSAGraphTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.is_file(), "VSA graph state integration has not been implemented")
        self.api = graph_api()
        mmgp = ModuleType("mmgp")
        mmgp.cuda_graphs = self.api
        mmgp.offload = SimpleNamespace(shared_state={"_attention": "vsa"})
        spec = importlib.util.spec_from_file_location("h3_vsa_graph_fixture", SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"mmgp": mmgp}):
            spec.loader.exec_module(self.module)
        self.owner = SimpleNamespace(manager=Manager())
        self.layout = SimpleNamespace(perm=torch.arange(8), valid_count=torch.tensor([8]),
            block_valid=torch.tensor([8]), tile=128, n_prefix_tiles=1,
            regions=(SimpleNamespace(start=1, stop=4, keep=1),), seq_len=384)
        self.policy = SimpleNamespace(enabled=True, layout=self.layout, signature=("layout-a", "cuda:0"),
            tile=128, sparsity=0.9, ref_keep_rate=0.1, int8_qk=True, cuda_kind=None,
            index_cache={})
        self.env = patch.dict(os.environ, {"WGP_H3_VSA_GRAPHS": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        # Exercise actual MMGP state registration using CPU tensors; no CUDA call is made.
        self.cuda_flag = patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True)
        self.cuda_flag.start()
        self.addCleanup(self.cuda_flag.stop)

    def warm(self):
        self.policy.index_cache = {("heads", 0, 7): torch.arange(7),
            ("prefix", 7): (torch.zeros(7, 2, 8, dtype=torch.int32), torch.full((7, 2), 8, dtype=torch.int32))}

    def test_first_forward_stays_eager_until_complete_prefix_cache_exists(self):
        self.assertFalse(self.module.prepare(self.owner, self.policy))
        self.assertIsNotNone(self.api.gates[0]())
        self.policy.index_cache[("heads", 0, 7)] = torch.arange(7)
        self.assertFalse(self.module.prepare(self.owner, self.policy))
        self.assertNotIn("h3_vsa", self.owner.manager.state)
        self.warm()
        self.assertTrue(self.module.prepare(self.owner, self.policy))
        self.assertIsNone(self.api.gates[0]())
        self.assertEqual(len(self.owner.manager.state["h3_vsa"]), 6)

    def test_same_layout_reuses_state_and_layout_replacement_clears_old_graphs(self):
        self.warm()
        self.module.prepare(self.owner, self.policy)
        key = self.api.keys[0]()
        self.module.prepare(self.owner, self.policy)
        self.assertEqual(self.api.keys[0](), key)
        self.assertEqual(self.owner.manager.cleared, 0)
        old_perm = self.layout.perm
        self.layout.perm = torch.arange(8)
        self.module.prepare(self.owner, self.policy)
        self.assertNotEqual(self.api.keys[0](), key)
        self.assertEqual(self.owner.manager.cleared, 1)
        self.assertNotIn(id(old_perm), self.owner.manager.state_ids)

    def test_new_layout_warmup_releases_previous_registered_state(self):
        self.warm()
        self.module.prepare(self.owner, self.policy)
        self.policy.signature = ("layout-b", "cuda:0")
        self.policy.index_cache = {}
        self.assertFalse(self.module.prepare(self.owner, self.policy))
        self.assertEqual(self.owner.manager.state, {})
        self.assertEqual(self.owner.manager.entries, {})
        self.assertIsNone(self.api.keys[0]())

    def test_changed_tile_selection_metadata_changes_graph_key(self):
        self.warm()
        self.module.prepare(self.owner, self.policy)
        key = self.api.keys[0]()
        self.layout.regions[0].keep = 2
        self.module.prepare(self.owner, self.policy)
        self.assertNotEqual(self.api.keys[0](), key)

    def test_off_and_unsupported_precision_branches_fail_closed(self):
        self.warm()
        self.module.prepare(self.owner, self.policy)
        for changes in ({"tile": 64}, {"int8_qk": False}, {"cuda_kind": "fp8"}):
            with self.subTest(changes=changes), patch.multiple(self.policy, **changes):
                self.assertFalse(self.module.prepare(self.owner, self.policy))
                self.assertIsNotNone(self.api.gates[0]())
                self.assertEqual(self.owner.manager.state, {})
        with patch.dict(os.environ, {"WGP_H3_VSA_GRAPHS": "0"}):
            self.assertFalse(self.module.prepare(self.owner, self.policy))
            self.assertIsNotNone(self.api.gates[0]())

    def test_other_attention_is_not_blocked_by_vsa_gate(self):
        self.module.prepare(self.owner, self.policy)
        with patch.dict(self.module.offload.shared_state, {"_attention": "sage2"}):
            self.assertIsNone(self.api.gates[0]())

    def test_statistics_reports_actual_manager_counts_and_owner_readiness(self):
        self.warm()
        self.module.prepare(self.owner, self.policy)
        result = self.module.statistics(self.owner)
        self.assertEqual((result["captures"], result["replays"]), (2, 5))
        self.assertTrue(result["state_ready"])
        other = SimpleNamespace(manager=Manager())
        self.assertFalse(self.module.statistics(other)["state_ready"])


class ReplayCounterTests(unittest.TestCase):
    def test_counter_increments_only_after_successful_graph_execution(self):
        tree = ast.parse(MMGP.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BlockGraphs")
        method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_launch_replay"), None)
        self.assertIsNotNone(method, "successful graph replay accounting is not implemented")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(MMGP), "exec"), namespace)
        manager = SimpleNamespace(stats={"captures": 0, "replays": 0})
        entry = SimpleNamespace(graph=SimpleNamespace(replay=lambda: None))
        namespace["_launch_replay"](manager, entry)
        self.assertEqual(manager.stats["replays"], 1)
        entry.graph.replay = lambda: (_ for _ in ()).throw(RuntimeError("capture launch failed"))
        with self.assertRaisesRegex(RuntimeError, "capture launch failed"):
            namespace["_launch_replay"](manager, entry)
        self.assertEqual(manager.stats["replays"], 1)


if __name__ == "__main__":
    unittest.main()
