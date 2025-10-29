import asyncio
import concurrent.futures
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import regex as re

from .base import BaseTool, register_tool
from verl_tool.llm_agent.vision_utils import encode_image_url


def _flatten_nested_list(nested: Sequence[Union[str, Dict[str, Any], Sequence]]) -> List[Union[str, Dict[str, Any]]]:
    """Flatten a nested list structure."""
    result: List[Union[str, Dict[str, Any]]] = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            result.extend(_flatten_nested_list(item))
        else:
            result.append(item)
    return result


def _extract_path(item: Union[str, Dict[str, Any]]) -> Optional[str]:
    """Extract a filesystem path from the item if available."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("path", "image", "frame", "file", "url"):
            if key in item and isinstance(item[key], str):
                return item[key]
    return None


def _parse_frame_indices(query: str) -> List[int]:
    """
    Parse requested frame indices from query text.
    Supports individual indices ('第3帧', 'frame 5') and ranges ('5-8', '5~8', '5到8').
    Returns zero-based indices.
    """
    indices: set[int] = set()
    # handle ranges first
    range_pattern = re.compile(
        r"(?P<start>\d+)\s*(?:[-~到toTO～~至]|到)\s*(?P<end>\d+)",
        re.IGNORECASE,
    )
    for match in range_pattern.finditer(query):
        start = int(match.group("start"))
        end = int(match.group("end"))
        if end < start:
            start, end = end, start
        indices.update(i - 1 for i in range(start, end + 1))

    # individual numbers
    for number in re.findall(r"\d+", query):
        idx = int(number) - 1
        if idx >= 0:
            indices.add(idx)

    return sorted(indices)


def _normalize_turn_frames(
    turn_frames: Optional[Sequence[Sequence[Union[int, str, Dict[str, Any]]]]],
    flat_frames: List[str],
) -> Optional[List[List[str]]]:
    """Normalize per-turn frame specification into explicit frame path lists."""
    if not turn_frames:
        return None

    normalized: List[List[str]] = []
    for group in turn_frames:
        group_paths: List[str] = []
        for item in group:
            if isinstance(item, int):
                idx = item - 1
                if 0 <= idx < len(flat_frames):
                    group_paths.append(flat_frames[idx])
            else:
                path = _extract_path(item)
                if path:
                    group_paths.append(path)
        if group_paths:
            normalized.append(group_paths)
    return normalized or None


@register_tool
class VideoQARMultiTurnTool(BaseTool):
    """
    Tool that supports multi-turn video understanding.
    After the agent issues a `<query>...</query>`, the tool responds with `<response>` content that
    embeds the requested video frames as `<image>` placeholders.
    """

    tool_type = "video_qar"
    stop_tokens = ["</query>"]

    def __init__(self, num_workers: int = 1):
        super().__init__(num_workers=num_workers)
        max_workers = min(32, (os.cpu_count() or 1) + 4)  # type: ignore[name-defined]
        self.image_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="video_qar_image_worker",
        )

    def get_usage_inst(self) -> str:
        return (
            "Format each reasoning turn as:\n"
            "<think>...</think>\n"
            "<query>Describe which frames or time range you need</query>\n"
            "The tool will reply with <response>...</response> along with <image> tokens for the selected frames.\n"
            "After inspecting the frames, continue with <summary>...</summary>. "
            "Finish with a final <answer>...</answer>."
        )

    def load_env(self, trajectory_id: str) -> Dict[str, Any]:
        env = super().load_env(trajectory_id)
        if "initialized" not in env:
            env.update(
                {
                    "initialized": False,
                    "frames": [],
                    "turn_frames": None,
                    "cursor": 0,
                    "default_chunk": 4,
                    "query_history": [],
                    "encoded_cache": {},
                }
            )
        return env

    def delete_env(self, trajectory_id: str):
        env = self.env_cache.pop(trajectory_id, None)
        if env and "encoded_cache" in env:
            env["encoded_cache"].clear()

    def parse_action(self, action: str) -> Tuple[str, bool]:
        match = re.search(r"<query>(.*?)</query>", action, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            return "", False
        query = match.group(1).strip()
        if not query:
            return "", False
        return query, True

    def _ensure_env_initialized(self, env: Dict[str, Any], extra_field: Dict[str, Any]):
        if env["initialized"]:
            return

        frame_sources: List[str] = []
        turn_frames: Optional[List[List[str]]] = None

        if not extra_field:
            env["initialized"] = True
            env["frames"] = []
            env["turn_frames"] = None
            return

        potential_sources: List[Union[str, Dict[str, Any]]] = []
        for key in ("frame_paths", "frames", "frame_list", "images", "thumbnails"):
            if key in extra_field:
                value = extra_field[key]
                if isinstance(value, (list, tuple)):
                    potential_sources.extend(_flatten_nested_list(value))

        for item in potential_sources:
            path = _extract_path(item)
            if path:
                frame_sources.append(path)

        turn_frames = _normalize_turn_frames(
            extra_field.get("turn_frames") or extra_field.get("frame_groups"),
            frame_sources,
        )

        env["frames"] = frame_sources
        env["turn_frames"] = turn_frames
        env["default_chunk"] = int(extra_field.get("frames_per_query", extra_field.get("default_frame_chunk", 4)) or 4)
        env["initialized"] = True

    def _select_frames(
        self,
        env: Dict[str, Any],
        query: str,
    ) -> Tuple[List[str], List[int]]:
        frames: List[str] = env["frames"]
        if not frames:
            return [], []

        if env["turn_frames"]:
            turn_idx = len(env["query_history"])
            if turn_idx >= len(env["turn_frames"]):
                return [], []
            selected = env["turn_frames"][turn_idx]
            indices = []
            for path in selected:
                try:
                    indices.append(frames.index(path) + 1)
                except ValueError:
                    indices.append(-1)
            return selected, indices

        requested_indices = _parse_frame_indices(query)
        if requested_indices:
            valid_indices = [idx for idx in requested_indices if 0 <= idx < len(frames)]
        else:
            start = env["cursor"]
            end = min(start + env["default_chunk"], len(frames))
            valid_indices = list(range(start, end))
            env["cursor"] = end

        if not requested_indices:
            env["cursor"] = min(len(frames), env["cursor"])

        selected_paths = [frames[idx] for idx in valid_indices if 0 <= idx < len(frames)]
        return selected_paths, [i + 1 for i in valid_indices]

    async def _encode_frames(self, env: Dict[str, Any], frame_paths: List[str]) -> List[str]:
        loop = asyncio.get_event_loop()
        tasks = []
        for path in frame_paths:
            tasks.append(loop.run_in_executor(self.image_executor, self._encode_single_frame, env, path))
        return await asyncio.gather(*tasks)

    @staticmethod
    def _encode_single_frame(env: Dict[str, Any], frame_path: str) -> str:
        cache = env.setdefault("encoded_cache", {})
        if frame_path in cache:
            return cache[frame_path]
        encoded = encode_image_url(frame_path)
        cache[frame_path] = encoded
        return encoded

    async def _conduct_action_async(
        self,
        trajectory_id: str,
        action: str,
        extra_field: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool, bool]:
        query, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        self._ensure_env_initialized(env, extra_field)

        if not is_valid:
            observation = {
                "obs": "<response>查询格式无效，请在<query>标签内描述需要查看的帧。</response>",
            }
            valid = False
            done = False
        else:
            frame_paths, frame_indices = self._select_frames(env, query)
            if not frame_paths:
                observation = {
                    "obs": "<response>未找到可展示的帧，请尝试调整查询或结束推理。</response>",
                }
                valid = False
                done = False
            else:
                encoded_images = await self._encode_frames(env, frame_paths)
                lines = []
                for order, (idx, _) in enumerate(zip(frame_indices, encoded_images), start=1):
                    if idx >= 0:
                        lines.append(f"帧 {idx}: <image>")
                    else:
                        lines.append(f"帧 {order}: <image>")
                response_text = "<response>\n" + "\n".join(lines) + "\n</response>"
                observation = {
                    "obs": response_text,
                    "image": encoded_images,
                    "frame_indices": frame_indices,
                }
                valid = True
                done = False

        env["query_history"].append(
            {
                "query": query,
                "observation": observation,
                "valid": valid,
            }
        )
        self.update_env(trajectory_id, env, query, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        return observation, done, valid

    async def aget_observations(
        self,
        trajectory_ids: List[str],
        actions: List[str],
        extra_fields: List[Dict[str, Any]],
    ):
        tasks = [
            self._conduct_action_async(traj_id, action, extra)
            for traj_id, action, extra in zip(trajectory_ids, actions, extra_fields)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        observations: List[Union[str, Dict[str, Any]]] = []
        dones: List[bool] = []
        valids: List[bool] = []

        for result in results:
            if isinstance(result, Exception):
                observations.append("<response>工具处理失败，请重试。</response>")
                dones.append(False)
                valids.append(False)
            else:
                obs, done, valid = result
                observations.append(obs)
                dones.append(done)
                valids.append(valid)
        return observations, dones, valids

    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._conduct_action_async(trajectory_id, action, extra_field))
        finally:
            loop.close()

    def __del__(self):
        if hasattr(self, "image_executor"):
            self.image_executor.shutdown(wait=False)
