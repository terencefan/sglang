"""Exercise the real transfer-worker loop under deterministic scheduler cleanup.

CPU-only: GPU writes are mocked; queue, status and cleanup control flow are real.
"""

import ast
import logging
import time
import unittest
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import Mock


class Poll(IntEnum):
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


def worker():
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/disaggregation/mooncake/conn.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MooncakeKVManager"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "transfer_worker"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    ns = {
        "KVPoll": Poll,
        "torch": SimpleNamespace(Tensor=type("Tensor", (), {})),
        "logger": logging.getLogger(__name__),
        "time": time,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns["transfer_worker"]


class RetiringDict(dict):
    """Scheduler retires a room after a worker observes its presence/value."""

    def __contains__(self, key):
        found = super().__contains__(key)
        if found:
            super().pop(key)
        return found

    def get(self, key, default=None):
        result = super().get(key, default)
        super().pop(key, None)
        return result


class CleanupRace(unittest.TestCase):
    def run_worker(self, race=None, terminal=Poll.Success, last=True):
        rooms = [11, 22]
        manager = SimpleNamespace(
            enable_trace=False,
            enable_staging=False,
            enable_deferred_decode_kv_release=False,
            bootstrap_port=18998,
            request_status={r: Poll.WaitingForInput for r in rooms},
            transfer_infos={
                r: {0: SimpleNamespace(room=r, is_dummy=True)} for r in rooms
            },
            req_to_decode_prefix_len={r: 0 for r in rooms},
            _staging_outstanding=defaultdict(int),
            finished=[],
        )
        manager.check_status = lambda room: manager.request_status[room]
        manager.update_status = lambda room, value: manager.request_status.__setitem__(
            room, value
        )
        manager.poison_deferred_ack_room = Mock()
        manager.record_failure = Mock()

        def finished(room, record):
            manager.finished.append(room)
            if race == "fatal":
                raise RuntimeError("unsettled GPU write")
            if terminal != Poll.Success:
                manager.request_status[room] = terminal
            if room == 11 and race == "status":
                manager.request_status = RetiringDict(manager.request_status)
            if room == 11 and race == "metadata":
                manager.transfer_infos = RetiringDict(manager.transfer_infos)
            if room == 11 and race == "already_cleared":
                manager.request_status.pop(room, None)
                manager.transfer_infos.pop(room, None)

        manager.note_send_finished = finished
        queue = SimpleQueue()
        for r in rooms:
            queue.put(
                SimpleNamespace(
                    room=r,
                    staging_counted=False,
                    wait_event=None,
                    indices_event=None,
                    prefill_kv_indices=[],
                    is_last_chunk=last,
                    send_record=None,
                )
            )
        queue.put(None)
        worker()(manager, queue, None)
        self.assertEqual(
            manager.finished,
            rooms,
            "one retired room must not kill subsequent transfers",
        )
        manager.record_failure.assert_not_called()
        manager.poison_deferred_ack_room.assert_not_called()
        return manager

    def test_real_transfer_failure_is_not_silenced(self):
        with self.assertRaisesRegex(
            RuntimeError, "Transfer thread failed.*unsettled GPU write"
        ):
            self.run_worker("fatal")

    def test_status_retired_between_presence_check_and_read(self):
        m = self.run_worker("status")
        self.assertFalse(m.transfer_infos)

    def test_metadata_retired_between_presence_check_and_pop(self):
        m = self.run_worker("metadata")
        self.assertFalse(m.transfer_infos)

    def test_scheduler_already_cleared_room(self):
        m = self.run_worker("already_cleared")
        self.assertFalse(m.transfer_infos)

    def test_successful_room_is_cleaned(self):
        self.assertFalse(self.run_worker().transfer_infos)

    def test_failed_last_chunk_is_cleaned(self):
        self.assertFalse(self.run_worker(terminal=Poll.Failed).transfer_infos)

    def test_failed_nonlast_chunk_keeps_metadata(self):
        self.assertEqual(
            len(self.run_worker(terminal=Poll.Failed, last=False).transfer_infos), 2
        )

    def test_unfinished_room_keeps_metadata(self):
        self.assertEqual(
            len(self.run_worker(terminal=Poll.Transferring, last=False).transfer_infos),
            2,
        )


if __name__ == "__main__":
    unittest.main()
