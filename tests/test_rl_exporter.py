"""Tests for RL exporter utilities."""

import json
import os
import tempfile

import mujoco
import onnx
import pytest
from conftest import get_test_device

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
  list_to_csv_str,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg


def test_list_to_csv_str():
  """Test CSV string conversion utility."""
  # Test with floats.
  result = list_to_csv_str([1.23456, 2.34567, 3.45678], decimals=3)
  assert result == "1.235,2.346,3.457"

  # Test with integers.
  result = list_to_csv_str([1, 2, 3], decimals=2)
  assert result == "1.00,2.00,3.00"

  # Test with mixed types.
  result = list_to_csv_str([1.5, "hello", 2.5], decimals=1)
  assert result == "1.5,hello,2.5"

  # Test custom delimiter.
  result = list_to_csv_str([1.0, 2.0, 3.0], decimals=1, delimiter=";")
  assert result == "1.0;2.0;3.0"


def test_attach_metadata_to_onnx():
  """Test that metadata can be attached to ONNX models."""
  # Create a dummy ONNX model.
  with tempfile.TemporaryDirectory() as tmpdir:
    onnx_path = os.path.join(tmpdir, "test_policy.onnx")

    # Create minimal ONNX model.
    input_tensor = onnx.helper.make_tensor_value_info(
      "input", onnx.TensorProto.FLOAT, [1, 2]
    )
    output_tensor = onnx.helper.make_tensor_value_info(
      "output", onnx.TensorProto.FLOAT, [1, 2]
    )
    node = onnx.helper.make_node("Identity", ["input"], ["output"])
    graph = onnx.helper.make_graph(
      [node], "test_graph", [input_tensor], [output_tensor]
    )
    model = onnx.helper.make_model(graph)
    onnx.save(model, onnx_path)

    # Attach metadata.
    metadata = {
      "run_path": "test/run/path",
      "joint_names": ["joint_a", "joint_b"],
      "joint_stiffness": [20.0, 10.0],
      "joint_damping": [1.0, 1.0],
      "extra_field": "extra_value",
    }
    attach_metadata_to_onnx(onnx_path, metadata)

    # Load and verify metadata was attached.
    loaded_model = onnx.load(onnx_path)
    metadata_props = {prop.key: prop.value for prop in loaded_model.metadata_props}

    # Check all metadata fields are present.
    assert "run_path" in metadata_props
    assert "joint_names" in metadata_props
    assert "joint_stiffness" in metadata_props
    assert "extra_field" in metadata_props

    # Check values are correct.
    assert metadata_props["run_path"] == "test/run/path"
    assert metadata_props["extra_field"] == "extra_value"

    # Check list was converted to CSV string.
    joint_names = metadata_props["joint_names"].split(",")
    assert len(joint_names) == 2
    assert "joint_a" in joint_names
    assert "joint_b" in joint_names

    # Check stiffness values are in natural joint order.
    stiffness_values = [float(x) for x in metadata_props["joint_stiffness"].split(",")]
    assert stiffness_values == [20.0, 10.0]  # Natural order: joint_a (20), joint_b (10)


# Robot with 2 joints but only 1 actuator (underactuated).
ROBOT_XML_UNDERACTUATED = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
      <body name="link2" pos="0 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link2_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="actuator1" joint="joint2" gear="1.0"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def test_get_base_metadata_skips_non_actuated_joints(device):
  """get_base_metadata handles non-actuated joints without KeyError."""
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_UNDERACTUATED),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=1.0,
      entities={"robot": robot_cfg},
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "joint_pos": ObservationTermCfg(
            func=lambda env: env.scene["robot"].data.joint_pos
          ),
        },
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=1,
    episode_length_s=1.0,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  metadata = get_base_metadata(env, run_path="dummy/run")

  robot = env.scene["robot"]

  # All joints (including non-actuated) should be listed in joint_names metadata.
  joint_names_meta = metadata["joint_names"]
  assert isinstance(joint_names_meta, list)
  assert joint_names_meta == list(robot.joint_names)
  assert "joint1" in joint_names_meta
  assert "joint2" in joint_names_meta

  # Stiffness/damping are only defined for actuated joints, in natural joint order.
  stiffness_meta = metadata["joint_stiffness"]
  damping_meta = metadata["joint_damping"]
  assert isinstance(stiffness_meta, list)
  assert isinstance(damping_meta, list)
  assert len(stiffness_meta) == len(robot.spec.actuators)
  assert len(damping_meta) == len(robot.spec.actuators)

  # Everything a deploy runtime would otherwise have to hardcode. These exist
  # so a consumer can read the policy's own shape and limits off the artifact
  # instead of keeping a private copy that silently goes stale.
  n_act = len(robot.spec.actuators)
  for key in ("joint_torque_limits", "joint_lower_limits", "joint_upper_limits"):
    limits = metadata[key]
    assert isinstance(limits, list) and len(limits) == n_act, key
  timing = {k: metadata[k] for k in ("control_dt", "control_hz", "physics_dt")}
  assert all(isinstance(v, float) for v in timing.values())
  assert timing["control_dt"] == pytest.approx(0.01)  # timestep * decimation
  assert timing["control_hz"] == pytest.approx(100.0)
  assert timing["physics_dt"] == pytest.approx(0.01)
  # Layout: one name per term, one flattened width per term, same order.
  dims, names = metadata["observation_dims"], metadata["observation_names"]
  hist = metadata["observation_history_lengths"]
  assert isinstance(dims, list) and isinstance(names, list) and isinstance(hist, list)
  assert len(dims) == len(names) == len(hist)
  assert all(d > 0 for d in dims)
  # Per term, not one number for the group: the group config may leave each
  # term to its own depth, and with mixed depths the flattened widths alone
  # cannot say how any single term was stacked.
  assert all(h >= 1 for h in hist)

  env.close()


def test_metadata_numbers_survive_the_round_trip():
  """Full precision, because a deploy runtime acts on these numbers.

  Rounding to 3 decimals would turn the metadata into a checksum rather than a
  source: the G1 wrist action scale would come back as 0.075 against a trained
  0.0745.
  """
  exact = [0.07450087032950714, 40.17923863450712, -0.312, 2.557889775413375]
  assert [float(x) for x in list_to_csv_str(exact).split(",")] == exact
  # Explicit rounding still available for anything that wants it.
  assert list_to_csv_str(exact, decimals=3) == "0.075,40.179,-0.312,2.558"


def test_observation_params_name_each_terms_frame_and_scope(device):
  """The obs key alone cannot say which frame a partner term is expressed in.

  Two cells of the partner-observation matrix differ only in
  ``base_body_name`` and export the same ``observation_names``, so without
  the params a deploy runtime has to guess whether to project into the torso
  or the pelvis.
  """
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_UNDERACTUATED),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)
    ),
  )
  joint_pos = lambda env, **kwargs: env.scene["robot"].data.joint_pos  # noqa: E731

  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=1.0,
      entities={"robot": robot_cfg},
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "human_ref_feet_pos_b": ObservationTermCfg(
            func=joint_pos,
            params={
              "command_name": "codancing",
              "body_names": ["pelvis", "left_knee_link"],
              "base_body_name": "pelvis",
            },
          ),
          "robot_ref_joint_state": ObservationTermCfg(
            func=joint_pos,
            params={
              "command_name": "codancing",
              "ref_face": "free",
              "body_set": None,
              # Stands in for the deploy `source` object: real, and not
              # something JSON can carry.
              "source": lambda: None,
            },
          ),
        },
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=1,
    episode_length_s=1.0,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  metadata = get_base_metadata(env, run_path="dummy/run")

  # One JSON string for the whole group, one entry per term, term order.
  raw = metadata["observation_params"]
  assert isinstance(raw, str)
  params = json.loads(raw)
  assert params == [
    {"body_names": ["pelvis", "left_knee_link"], "base_body_name": "pelvis"},
    {"ref_face": "free", "body_set": None},
  ]
  names = metadata["observation_names"]
  assert isinstance(names, list)
  assert len(params) == len(names)

  env.close()


def test_contact_slot_names_come_from_the_provider():
  """The one key the codancing exporter adds on top of the base metadata.

  Slot order is a property of the DATA release, not of the code, so it has to
  travel with the artifact: a deploy runtime that assumes it sends a per-hand
  stiffness to the other wrist. Duck-typed against the command manager rather
  than built on a full env, because the lookup IS the logic under test.
  """
  from mjlab.tasks.codancing.rl.exporter import _contact_slot_names

  class _Term:
    def __init__(self, provider):
      self._contact_provider = provider

  class _Manager:
    def __init__(self, terms):
      self._terms = terms
      self.active_terms = list(terms)

    def get_term(self, name):
      return self._terms[name]

  class _Env:
    def __init__(self, terms):
      self.command_manager = _Manager(terms)

  wrists = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
  provider = type("P", (), {"_slot_link_names": wrists})()
  assert _contact_slot_names(_Env({"codancing": _Term(provider)})) == wrists

  # A run with no force field, and a term that is not a codancing command at
  # all: both mean "nothing to write", not an error.
  assert _contact_slot_names(_Env({"codancing": _Term(None)})) is None
  assert _contact_slot_names(_Env({"pose": object()})) is None
