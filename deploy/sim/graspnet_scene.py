"""Stub for the (not-yet-open-sourced) GraspNet object-injection module.

The upstream HANDOFF deploy stack references this module from
``sim_policy_node._load_spec_with_graspnet`` when ``WBC_MJLAB_G1_XML`` is
set to a scene that defines cameras/objects, but the actual GraspNet asset
injection code was not included in this repo's public release. This stub
keeps that code path importable while adding no objects to the scene.
"""

from __future__ import annotations

import mujoco


def inject_graspnet_objects(
    spec: mujoco.MjSpec,
    object_ids="default",
    num_random: int = 2,
    seed: int | None = None,
) -> list[str]:
    """No-op: adds nothing to `spec`, returns an empty list of added object names."""
    return []
