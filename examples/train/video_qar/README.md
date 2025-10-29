## Multi-turn Video QAR (GRPO)

This recipe demonstrates how to train a multi-round video understanding agent with GRPO, where the policy iteratively asks for visual evidence using `<query>...</query>` tags and receives frame snapshots back inside `<response>...</response>` sections before continuing the reasoning.

### Expected Dialogue Template

For each reasoning round, the model should follow:

```text
<think>推理想法</think>
<query>需要查看的视觉线索</query>
<response>（系统在这里填充帧，包含若干个 <image> 占位符）</response>
<summary>结合当前视觉信息的结论</summary>
```

After all rounds, the policy must output

```text
<think>最终推理</think>
<answer>最终答案</answer>
```

When the actor emits `</query>`, the `video_qar` tool server will return the requested frame thumbnails (or the next batch if no explicit indices are found). This ensures the rollout prompt automatically contains the visual evidence before the model continues generating `<response>` and `<summary>`.

### Data Preparation

Prepare a parquet dataset with the following fields (see `VerlToolRLHFDataset` for details):

- `prompt`: chat template containing the system/user turns. The user message should include instructions about the multi-turn format above.
- `images`: list of frame image paths (absolute paths recommended). If you already grouped frames per round, also include `extra_info.turn_frames` as a list of lists.
- `reward_model.ground_truth`: target answers for reward computation.
- `extra_info`: must include `images`, `is_video` (optional), and any metadata needed during rollout.

You can reuse the Pixel Reasoner data preparation script as a reference and extend it to add multi-round annotations if needed.

### Training

Update `dataset_name` and `model_name` inside `train_grpo.sh`, then launch:

```bash
bash examples/train/video_qar/train_grpo.sh
```

This script:

- Spins up the `video_qar` tool server.
- Uses `</query>` as the action stop token so that rollouts pause after each visual query.
- Enables multi-turn layout (`enable_mtrl=True`) so inserted observations become new chat turns.

Tune batch sizes, sequence lengths, or max turns according to your GPU budget and dataset length.

### Evaluation

You can reuse the evaluation harness from other recipes by pointing the tool server to `video_qar` and enabling `enable_mtrl` so that the evaluator injects frame observations in the same way as training.
