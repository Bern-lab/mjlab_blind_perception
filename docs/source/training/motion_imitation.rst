.. _motion-imitation:

Motion Imitation
================

mjlab can train humanoid policies to imitate reference motions. This page
covers motion data preprocessing and training.

WandB registry setup
--------------------

mjlab uses `Weights & Biases <https://wandb.ai/>`_ to store and load
reference motions. Before preprocessing any motions, create a WandB registry
by following the
`BeyondMimic instructions <https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup>`_
(only the registry creation step).

Motion data
-----------

Reference motions are retargeted CSV files in Unitree's generalized
coordinate convention (base position, base quaternion in xyzw, then joint
angles).

This branch no longer keeps the old CSV conversion utility. Use an existing
motion NPZ from the tracking pipeline or upload a preprocessed NPZ to your
WandB registry before launching tracking training.

.. warning::

   The NPZ must use mjlab/MuJoCo body ordering. Converters from other
   frameworks such as IsaacLab can produce incompatible body orderings. A
   mismatched NPZ will map tracking targets to the wrong bodies and training
   will not converge.

Training
--------

.. code-block:: bash

   uv run train Mjlab-Tracking-Flat-Unitree-G1 \
       --registry-name your-org/motions/motion-name \
       --env.scene.num-envs 4096

Evaluation
----------

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1 \
       --wandb-run-path your-org/mjlab/run-id
