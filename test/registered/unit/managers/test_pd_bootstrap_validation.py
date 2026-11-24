"""Invalid PD transfer IDs must fail before state creation or dispatch."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode, validate_bootstrap_room
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPDBootstrapValidation(CustomTestCase):
    def validate(self, obj, mode=DisaggregationMode.PREFILL, backend="mooncake"):
        obj.normalize_batch_and_arguments()
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.disaggregation_mode = mode
        manager.rid_to_state = {}
        with patch(
            "sglang.srt.managers.tokenizer_manager.get_disagg",
            return_value=SimpleNamespace(disaggregation_transfer_backend=backend),
        ):
            try:
                manager._validate_pd_bootstrap_request(obj)
            finally:
                self.assertEqual(manager.rid_to_state, {})

    def test_rejects_missing_room_in_both_roles_even_with_rank(self):
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            for rank in (None, 0):
                with self.subTest(mode=mode, rank=rank):
                    with self.assertRaisesRegex(ValueError, "bootstrap_room"):
                        self.validate(
                            GenerateReqInput(text="hello", routed_dp_rank=rank), mode
                        )

    def test_http_validation_precedes_state_creation_and_dispatch(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.auto_create_handle_loop = MagicMock()
        manager._set_default_priority = MagicMock()
        manager._init_req_state = MagicMock()
        manager._send_one_request = MagicMock()

        async def reject():
            with self.assertRaisesRegex(ValueError, "bootstrap_room"):
                await manager.generate_request(GenerateReqInput(text="bad")).__anext__()

        with patch(
            "sglang.srt.managers.tokenizer_manager.get_disagg",
            return_value=SimpleNamespace(disaggregation_transfer_backend="mooncake"),
        ):
            asyncio.run(reject())
        manager._init_req_state.assert_not_called()
        manager._send_one_request.assert_not_called()

    def test_valid_request_after_invalid_request_and_zero_room(self):
        with self.assertRaises(ValueError):
            self.validate(GenerateReqInput(text="bad"))
        for room in (0, 42, -(1 << 63), (1 << 63) - 1):
            self.validate(GenerateReqInput(text="good", bootstrap_room=room))

    def test_rejects_entire_batch_when_later_item_has_no_room(self):
        with self.assertRaisesRegex(ValueError, "bootstrap_room"):
            self.validate(GenerateReqInput(text=["a", "b"], bootstrap_room=[42, None]))
        self.validate(GenerateReqInput(text=["a", "b"], bootstrap_room=42))

    def test_plain_and_fake_keep_existing_behavior(self):
        self.validate(GenerateReqInput(text="plain"), DisaggregationMode.NULL)
        self.validate(GenerateReqInput(text="fake"), backend="fake")

    def test_non_integer_and_out_of_range_ids(self):
        for room in (None, True, "42", 1.5, 1 << 63, -(1 << 63) - 1):
            with self.subTest(room=room):
                with self.assertRaises(ValueError):
                    validate_bootstrap_room(room)


if __name__ == "__main__":
    unittest.main()
