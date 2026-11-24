"""Disaggregation mode, kept free of heavy imports for request and IPC code."""

from enum import Enum

FAKE_BOOTSTRAP_HOST = "2.2.2.2"


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode: str) -> str:
        if mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        elif mode == DisaggregationMode.DECODE.value:
            return "decode"
        return "unified"


def validate_bootstrap_room(room) -> None:
    """Validate one PD transfer ID before it reaches rank dispatch.

    Zero is valid (including health/warmup requests). A routed rank cannot
    substitute for the ID used to pair prefill and decode KV transfers.
    """
    if type(room) is not int or not -(1 << 63) <= room < (1 << 63):
        raise ValueError(
            "PD generation requires bootstrap_room to be a signed 64-bit integer. "
            "Send inference requests through the PD router."
        )
