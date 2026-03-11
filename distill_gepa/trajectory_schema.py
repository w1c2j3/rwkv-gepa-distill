from __future__ import annotations

from dataclasses import dataclass

from .common import build_shuffle_key


@dataclass(frozen=True)
class SlotKey:
    question_id: str
    target_model: str
    sample_index: int


def build_slot_id(*, question_id: str, target_model: str, sample_index: int) -> str:
    return f"slot::{question_id}::{target_model}::{sample_index}"


def build_complex_trajectory_id(*, slot_id: str) -> str:
    return f"{slot_id}::complex"


def build_rewrite_trajectory_id(*, slot_id: str, rewrite_variant_index: int) -> str:
    return f"{slot_id}::rewrite::{rewrite_variant_index}"


def parse_slot_id(slot_id: str) -> SlotKey:
    parts = slot_id.split("::")
    if len(parts) < 5 or parts[0] != "slot":
        raise ValueError(f"Invalid slot_id: {slot_id!r}")
    question_id = "::".join(parts[1:-2]).strip()
    target_model = parts[-2].strip()
    sample_index = int(parts[-1])
    if not question_id or not target_model:
        raise ValueError(f"Invalid slot_id: {slot_id!r}")
    return SlotKey(
        question_id=question_id,
        target_model=target_model,
        sample_index=sample_index,
    )


def build_slot_shuffle_key(*, question_id: str, target_model: str, sample_index: int) -> str:
    return build_shuffle_key("slot", question_id, target_model, sample_index)
