"""Tests for video recording with mediapy."""

from pathlib import Path
from unittest.mock import Mock

import mediapy as media
import numpy as np
import torch


def _make_mock_env(num_envs: int = 1):
  """Create a mock environment that produces random RGB frames."""
  env = Mock()
  env.render_mode = "rgb_array"
  env.metadata = {"render_fps": 30}
  env.render.return_value = np.random.randint(0, 255, (num_envs, 64, 64, 3), np.uint8)
  env.step.return_value = (
    torch.zeros(num_envs),  # obs
    torch.zeros(num_envs),  # reward
    torch.zeros(num_envs, dtype=torch.bool),  # terminated
    torch.zeros(num_envs, dtype=torch.bool),  # truncated
    {},  # info
  )
  env.close.return_value = None
  env.unwrapped = env
  return env


def test_step_trigger_writes_video(tmp_path: Path):
  """VideoRecorder writes a readable mp4 when the step trigger fires."""
  from mjlab.utils.wrappers.video_recorder import VideoRecorder

  env = _make_mock_env()
  recorder = VideoRecorder(
    env,
    video_folder=tmp_path,
    step_trigger=lambda step: step == 0,
    video_length=5,
    disable_logger=True,
  )

  action = torch.zeros(1)
  for _ in range(6):
    recorder.step(action)

  recorder.close()

  videos = list(tmp_path.glob("*.mp4"))
  assert len(videos) == 1

  # Verify the file is a valid video readable by mediapy.
  frames = media.read_video(str(videos[0]))
  assert len(frames) == 5
  assert frames[0].shape == (64, 64, 3)


def test_accepts_string_path(tmp_path: Path):
  """VideoRecorder accepts a string path for video_folder."""
  from mjlab.utils.wrappers.video_recorder import VideoRecorder

  env = _make_mock_env()
  folder = str(tmp_path / "vids")
  recorder = VideoRecorder(
    env,
    video_folder=folder,
    step_trigger=lambda step: step == 0,
    video_length=3,
    disable_logger=True,
  )

  action = torch.zeros(1)
  for _ in range(4):
    recorder.step(action)

  recorder.close()

  assert list(Path(folder).glob("*.mp4"))


def _camera_recorder(tmp_path: Path, episode_step: int):
  """A task-layer CameraVideoRecorder over a mock env whose recorded env (env[0])
  reports ``episode_step`` as its per-env step count."""
  from mjlab.tasks.shared.wrapper.viewer import CameraVideoRecorder

  env = _make_mock_env()
  env.episode_length_buf = np.array([episode_step], dtype=np.int64)
  env.render_with_overrides = Mock(
    return_value=np.zeros((1, 64, 64, 3), dtype=np.uint8)
  )
  recorder = CameraVideoRecorder(
    env,
    video_folder=tmp_path,
    step_trigger=lambda step: step == 0,
    video_length=5,
    disable_logger=True,
  )
  recorder.trigger_type = "step"  # normally set by the base step() before recording
  return recorder


def test_camera_recorder_captures_reset_frame(tmp_path: Path):
  """At a reset boundary (episode_length_buf[0] == 0) the first recorded frame is
  observation_0, captured at recording start."""
  recorder = _camera_recorder(tmp_path, episode_step=0)
  try:
    recorder._start_recording()
    assert len(recorder.current_video_frames) == 1  # the reset frame
  finally:
    recorder._finish_recording()  # close the streaming ffmpeg writer


def test_camera_recorder_rejects_non_reset_start(tmp_path: Path):
  """Recording must begin right after a reset (the first frame is labelled step 0);
  starting mid-episode fails loudly instead of mislabelling a frame."""
  import pytest

  recorder = _camera_recorder(tmp_path, episode_step=5)
  with pytest.raises(RuntimeError, match="non-reset env"):
    recorder._start_recording()
  assert not recorder.current_video_frames  # nothing captured
