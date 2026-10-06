# scripts/clip_viz

Tools for looking at an augmented clip: forward kinematics only, no simulation.
Run them from the repo root with `uv run python scripts/clip_viz/<script>.py --help`.

- `render_clip_video.py`: render an adapted clip to mp4 with the forced links marked.
- `plot_force_curve.py`: per-slot |F| against frame, force events shaded.
- `plot_stiffness_curve.py`: force and per-slot stiffness on a shared frame axis.

Shared: `palette.py` (colors, figure defaults), `clip_evidence.py` and `g1_render.py` (clip readers and the G1 frame renderer the three tools build on).
