"""Tiny exact-integer reward for synthetic arithmetic, not medical evaluation."""

import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Score the final integer; real model output may earn zero.

    Signature matches the legacy NaiveRewardManager custom reward interface.
    This deliberately simple smoke reward is not a training-quality benchmark.
    """
    if data_source != "flagos_smoke_arithmetic":
        raise ValueError("This smoke reward only supports flagos_smoke_arithmetic data")
    integers = re.findall(r"(?<![\w.])-?\d+(?![\w.])", solution_str)
    if not integers:
        return 0.0
    return float(integers[-1] == str(ground_truth).strip())
