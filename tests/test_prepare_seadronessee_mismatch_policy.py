import pytest

from scripts.prepare_seadronessee import resolve_length_mismatch_policy


def test_length_mismatch_policy_aligned():
    action, pair_count, extra_frames, extra_annotations = resolve_length_mismatch_policy(
        num_frames=10,
        num_annotations=10,
        policy="truncate",
    )
    assert action == "aligned"
    assert pair_count == 10
    assert extra_frames == 0
    assert extra_annotations == 0


def test_length_mismatch_policy_truncate():
    action, pair_count, extra_frames, extra_annotations = resolve_length_mismatch_policy(
        num_frames=7,
        num_annotations=5,
        policy="truncate",
    )
    assert action == "truncate"
    assert pair_count == 5
    assert extra_frames == 2
    assert extra_annotations == 0


def test_length_mismatch_policy_skip():
    action, pair_count, extra_frames, extra_annotations = resolve_length_mismatch_policy(
        num_frames=5,
        num_annotations=8,
        policy="skip",
    )
    assert action == "skip"
    assert pair_count == 0
    assert extra_frames == 0
    assert extra_annotations == 3


def test_length_mismatch_policy_fail():
    with pytest.raises(RuntimeError):
        resolve_length_mismatch_policy(
            num_frames=3,
            num_annotations=4,
            policy="fail",
        )

