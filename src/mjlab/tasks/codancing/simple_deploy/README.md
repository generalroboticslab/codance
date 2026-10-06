# simple_deploy: running codancing policies on a real G1

Install the deploy extra on the robot's control computer first.
It adds the G1 SDK binding and onnxruntime, and the runners use the resulting venv directly (`.venv/bin/python3`):

```sh
make sync-deploy   # uv sync --extra deploy
```

Two control paths share one hardware layer.

- **Solo stand** (`stand.py`, `constants.py`, driven by `scripts/deploy_solo_stand.py`) assembles the observation directly from sensors and the numbers stored in the ONNX file.
  The ONNX file is all it needs: no simulator, no training config, no checkpoint.
- **Codance** (`codance.py`, self-driven) runs the waltz policies the same way, from the ONNX file alone: the moving reference is a plain numpy table played on the training pipeline's own schedule, and the partner term comes from live VICON (`vicon.py`).

Both give `loop.py` the same interface: `start(frame)` and `step(frame)` returning a `ControlStep`, plus `kp`, `kd`, `default_joint_pos`, `clamp_to_limits`, `reference_joint_pos`, and `max_episode_ticks`.
The robot and room tools around them are `python -m mjlab.tasks.codancing.simple_deploy
<tool>` (see [Robot and room tools](#robot-and-room-tools)).

## Solo stand

The 805-dim actor observation is stationary.
All eight terms are proprioception, a constant reference, or a constant compliance channel; there is no phase, clock, or cursor anywhere in the vector.
So none of the clip-cursor machinery is needed here, and episode length is an operator and safety choice rather than something the observation demands.

```sh
# in this order, every time; <nic> is the robot's wired interface
.venv/bin/python3 scripts/check_g1_remote.py --net-if <nic>
.venv/bin/python3 scripts/deploy_solo_stand.py --onnx data/checkpoints/stand.onnx \
    --stiffness-linear 140 --net-if <nic> --dry-run
# e-stop drills, then:
.venv/bin/python3 scripts/deploy_solo_stand.py --onnx data/checkpoints/stand.onnx \
    --stiffness-linear 140 --net-if <nic>
```

`--stiffness-linear` has no default on purpose: it is the compliance a run tells the policy to expect, so every run states it.

**Launch with the venv python, not `uv run`.**
Measured: on Ctrl-C the uv wrapper takes the SIGINT too and kills the process 0.92 s in, truncating the terminal damp hold that exists precisely so the robot is left latched on a damp command.
Under the venv python, SIGINT, SIGTERM and SIGHUP (an SSH link dropping) all complete the full 1.2 s hold and exit clean.

`check_g1_remote.py` confirms the wireless remote decodes the way the runner expects, before anything can move.
It is read-only and provably so: it constructs the comms with `release_motion=False`, so the onboard service keeps holding the robot, and it never calls `write_targets` or `damp`, so the command buffer stays empty and the wire thread publishes nothing at all.
Press B, then Y, then L1+A, then R1+A, then up and down, then A alone (which must fire nothing).
It exits non-zero and names what is missing if any combination never fired.

`--stiffness-step N` (stand runner only) maps the remote's up and down to plus and minus N N/m on both wrists' linear stiffness while READY or RUNNING, clamped to the trained 10 to 1000 N/m, so a force-gauge sweep is one run: start at `--stiffness-linear 40`, step by 100.
Each change is printed and logged as a run event, and the observation's stiffness block carries the value every tick.
`check_g1_remote.py` verifies the two buttons alongside the others.

Every run without `--dry-run` (either runner) writes `logs/simple_deploy/runs/<stamp>/`: `run.json` (the ONNX file, reference clip or clip plan, stiffness, gains, host, git HEAD, and the VICON calibration when the partner is live), `ticks.npz` with one row per control tick while the policy runs (`t_wall` epoch seconds in float64, `t_run`, `io_tick`, encoders, IMU, torques, raw action, target, target sent, the full observation, buttons) and, with `--partner vicon`, the world-frame VICON channels (`vicon_human_pos`, one row per object as tracked; `vicon_pelvis_pos`; the raw `vicon_robot_obj_pos` / `vicon_robot_obj_quat`; `vicon_stale`, the held-sample count), and `events.json` with every state change.

`--dry-run` reads, assembles, infers, and publishes damp only, never a position target.
With the robot hung: move one joint by hand and check the joint the readout names is the one you moved, then bend the waist and watch the reference-orientation error grow while the tilt stays put.

Reference and control numbers.
The reference motion NPZ (`data/reference_motion_stand/20260814_001_robot.npz`) is read with plain numpy; it is constant, so the reference never changes while it plays.
The control numbers (kp, kd, action scale, the knees-bent default pose, the joint ranges) come from the ONNX file's metadata, which the export writes at full precision.

The compliance channel is a command with no physical consumer: nothing becomes mechanically softer when it reads 140 N/m.
It only shapes what the policy expects, so it is fixed for a run and tuned between runs.

Episode length is a pure operator choice here, and `--episode-s inf` is supported: the reference is constant, so `at()` clamps at the last frame and serves the same pose forever, `reference_done` never fires on a static clip, and `StandController.max_episode_ticks` is `None`.
That wall clock is the only thing that ends a stand episode, so removing it leaves Y, the joint-space trips, and the e-stop as the ways out.
The cost of an unbounded run is memory: the tick record is held in RAM until exit, about 1 GB per hour at 50 Hz.

`--no-joint-trips` disarms the two joint-space trips (deviation vs the reference, joint-limit clamp streak) for a run where someone pulls on a wrist.
The stand reference IS the standing keyframe, so a hand pulled far enough to get the yield the policy was trained to give reads as a deviation, and a hand held at its stop keeps a joint clamped for as long as it is held.
Targets are still clamped every tick, and tilt, joint velocity, staleness and overrun stay armed.
The codance runner takes the same flag with the same meaning.

## Codance: the waltz policies from the ONNX file alone

`codance.py` deploys the six waltz policies the way `stand.py` deploys the standing one: onnxruntime plus numpy, no mjlab env.
The observation layout is read from the ONNX metadata: `observation_names` orders the nine terms, `observation_history_lengths` gives per-term depths (they are mixed: the partner term keeps 50 frames while everything else keeps 5), and `observation_params` sizes the partner term (`body_names`, `base_body_name`) and names the reference face each term trained against.
An ONNX file this runner cannot drive is refused by name.

```sh
# in this order, every time (venv python, not uv run; same SIGINT reason)
.venv/bin/python3 scripts/check_g1_remote.py --net-if <nic>
.venv/bin/python3 -m mjlab.tasks.codancing.simple_deploy.codance \
    --onnx data/checkpoints/decoupled.onnx --motion fwd --net-if <nic> \
    --dry-run --partner zeros
# e-stop drills, then, with <tracker> the VICON tracker's address:
.venv/bin/python3 -m mjlab.tasks.codancing.simple_deploy.codance \
    --onnx data/checkpoints/decoupled.onnx --motion fwd --net-if <nic> \
    --vicon-ip <tracker> --partner-profile partner-a [--demo transition] [--rounds N]
```

The reference is the fwd/rev waltz pair (`data/reference_motion_edits_g1_g1_waltz/20260224_001_robot.npz` and `20260408_001_robot.npz`, 487 frames at 50 Hz; rev is the time-reversed fwd) read with plain numpy on the schedule the training pipeline itself serves: the engage tick observes frame 1 (frame 0 is the spawn pose, the waltz HOLD, which deploy reproduces by ramping to that very frame; it is NOT the default keyframe: the arms differ by up to 1.5 rad), a transition swaps to the other clip's frame 0 the tick after the first clip's last frame (continuous by construction, so no re-anchor math), and `reference_done` fires after the planned one or two clips.

`--motion fwd|rev` picks the direction, `--demo transition` chains it with the other one.
The partner term defaults to live VICON: the Tracker objects derive from the partner body list in the ONNX metadata, through the wearer named by `--partner-profile` (whose own `vicon_objects` map wins, else the default `human_*` set: pelvis or torso_link both read the one trunk cluster `human_root`, and the knees read `human_{left,right}_knee`; `--vicon-objects` overrides, in the term's `body_names` order).
The waltz policies' `human_ref_pelvis_knees_pos_b` observes `pelvis, left_knee_link, right_knee_link`, so a run reads three objects: with `--partner-profile partner-b` those are `partner_b_root, partner_b_left_knee, partner_b_right_knee`.
A body the wearer has no object for is refused by name rather than falling back to someone else's markers, so a policy that observes other bodies stops before the robot moves.
The run then blocks up to 180 s until every object is tracked, so you can launch the run and then walk into the capture volume wearing the marker sets (`--no-wait-human` waits for `g1_pelvis` only and releases right after launch; the humans then just need to be tracked by R1+A).
The human objects carry usable POSITIONS only (only positions are read, so their axes need not be aligned to anything; the one quaternion read is `g1_pelvis`'s).
Two fixed corrections apply per tick: the base pose passes through the mounting transform (`vicon.py` `MOUNT_*`; identity by default, which assumes the robot object was built aligned to the pelvis link), and each human object's world z is shifted by the wearer's per-body offsets (the trained partner is a G1 proxy whose bodies differ in height from the worn clusters by a per-wearer amount).
The offsets are a named partner profile in `partner_profiles.yaml` (`partner-a` and `partner-b` are measured; an entry with `measured: null`, like `partner-c`, is refused until it is measured): `--partner-profile <name>` is required with a live partner, and `run.json` records the name and the numbers.
The same entry carries the wearer's Tracker objects, so a new wearer is one entry, not a code change.
Measure a wearer with `python -m mjlab.tasks.codancing.simple_deploy measure --profile <name>` (they stand still in the volume wearing the sets; the mean object heights minus the proxy's body heights become their entry).
`measure` reads the objects off that entry too; a new wearer whose objects are not the `human_*` set names them with `--objects TRUNK,LEFT_KNEE,RIGHT_KNEE` (the trunk set serves pelvis and torso_link), and the entry keeps them as its `vicon_objects`, so re-measuring someone is just the same command again.
A probe not taken on a standing wearer is refused rather than written: sets left on the floor read near 0.02 m and would derive about -0.7 m of offset, which looks deployable and wrecks the run, so any offset past 0.5 m fails the write.
`--partner zeros` is accepted with `--dry-run` only, because a zero vector means a partner standing inside the robot's pelvis.
`--no-joint-trips` disarms the two joint-space trips (deviation vs reference, clamp streak) for contact-heavy demos; targets stay clamped and tilt/velocity/staleness/overrun stay armed.

`--onnx` is the exported policy.
The dataset ships one per released policy as `data/checkpoints/<policy>.onnx`, and training writes one next to every checkpoint it saves.

## Operator procedure

States: `DAMP_IDLE` -> (L1+A) `RAMP`, lerping to the controller's pre-engage pose (stand: the default keyframe; codance: the clip's frame-0 waltz hold) at stiff holding gains over about 4 s -> `READY` -> (R1+A) a single tick that brings gains, policy, cursor and histories up together, expect the trained first-motion squat -> `RUNNING` -> reference end -> back to `RAMP`.
B damps from any state and is terminal: damp keeps publishing kp 0 / kd 8 until the process exits, with at least a 1 s hold.
Y in `RUNNING` is an operator stop back to the ramp.
The firmware remote damp (L2+B) is the independent e-stop layer, and software damp only covers live-link failures.

Never resume a stopped episode.
Restarting is a full start, which declares the heading to be the clip's frame-0 yaw again and refills the histories, and that is the only thing the policy was trained to see.

Comms is our nanobind `unitree-sdk2-bind` (the `deploy` extra), which works on Python 3.13, behind one small RobotIO interface; `--comms dummy` swaps in a stand-in with no DDS, for rehearsing without the robot.
Constructing the comms releases the onboard service, so the robot must already be hung or supported when a process starts.

`--net-if` names the NIC the robot is on (`ip -br link` lists them).
`--comms bind` requires it and checks that the link has carrier before building a participant, `codance.py` first of all, before loading the ONNX file and before the VICON walk-in wait; `--comms dummy` ignores it.
CycloneDDS reports an unplugged robot with the same words it reports a misspelled name, so without that check a dead cable reads as a typo in the flag.

## Robot and room tools

```sh
.venv/bin/python3 -m mjlab.tasks.codancing.simple_deploy <tool> [args]

bringup   # stand the robot up in software (L2+B -> L2+UP -> R1+X)
status    # read-only: which service holds the robot, and its mode
restore   # hand the robot back to its onboard service after a run
measure   # a wearer's partner profile from VICON (see the Codance section)
vicon     # the standalone room probe: tracker rate and the partner vector
```

`bringup`, `status` and `restore` take `--net-if`; `measure` and `vicon` take `--ip`, the VICON tracker's address.

## Known gap: releasing from the service's damp

Releasing the onboard `ai` service (`release_mode()`) while it holds the robot in its damp state switches the joints from the service's damping straight to the runner's damp command, with no staged transition, and the service's damping has not been measured.
Before doing this on a loaded robot, hang it and back-drive the joints by hand under each damp to confirm they resist alike.
