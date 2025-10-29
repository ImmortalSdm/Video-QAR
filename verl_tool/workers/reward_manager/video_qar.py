import regex as re
from pathlib import Path
from typing import Union

from verl.workers.reward_manager import register

from .torl import ToRLRewardManager


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _normalize_answer(text: Union[str, None]) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = text.replace("\\boxed", "").replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text)
    # remove common punctuation while keeping unicode letters/numbers
    text = re.sub(r"[，。！？、,.!?;:\"'`·～~\-——_/\\≈≠<>（）()\[\]{}|]", "", text)
    text = text.lower()
    return text


def video_qar_compute_score(solution_str: str, ground_truth) -> float:
    """
    Simple exact/substring match on the content inside <answer></answer>.
    """
    if isinstance(ground_truth, list):
        return max(video_qar_compute_score(solution_str, gt) for gt in ground_truth)

    match = ANSWER_PATTERN.search(solution_str)
    if match:
        pred = match.group(1)
    else:
        # fall back to the tail of the string
        pred = solution_str.split("</query>")[-1]
    norm_pred = _normalize_answer(pred)
    norm_gt = _normalize_answer(ground_truth)
    if not norm_gt:
        return 0.0
    if norm_pred == norm_gt:
        return 1.0
    if norm_gt in norm_pred or norm_pred in norm_gt:
        return 1.0
    return 0.0


@register("video_qar")
class VideoQARRewardManager(ToRLRewardManager):
    """
    Reward manager for multi-turn video understanding.
    Enforces the `<think>` / `<answer>` format and evaluates the final answer string match.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs):
        super().__init__(tokenizer, num_examine, compute_score=compute_score, reward_fn_key=reward_fn_key, **kwargs)
        self.compute_score = video_qar_compute_score
        # Encourage proper formatting for the multi-turn template
        self.add_format_think_penalty = True
        self.add_format_answer_penalty = True
        if "record_dir" in kwargs:
            self.record_dir = Path(kwargs['record_dir'])
            self.record_dir.mkdir(parents=True, exist_ok=True)
