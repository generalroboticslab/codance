"""Simple deploy: running CoDance policies on a real Unitree G1.

Two runners assemble the observation straight from sensors and the policy's
exported ONNX, with no simulator: ``codance.py`` for the waltz policies (the
partner from live VICON) and ``scripts/deploy_solo_stand.py`` for the
standing policy. ``python -m mjlab.tasks.codancing.simple_deploy`` carries
the robot and room tools around them. See README.md.
"""
