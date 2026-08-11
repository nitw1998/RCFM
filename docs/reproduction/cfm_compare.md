# Preprocessing-matched CFM comparator

`train_cfm_compare.py` is the controlled CFM entry used to isolate the region
weight in RCFM. It is separate from `train_cfm_basic.py`, which remains the
historical reproduction entry and is not modified by this comparator.

The CFM comparator and its paired RCFM run share the dataset artifact, split
hash, record-level normalization, source and target leads, model architecture,
optimizer, scheduler, batch size, seed, fixed validation noise, inference NFE,
checkpoint selection, local metric files, and W&B metric names. The declared
training-factor difference is:

- CFM compare: `region_weight=0`, minibatch OT disabled;
- RCFM: `region_weight=0.01`, minibatch OT disabled.

ROI MSE, non-ROI MSE, and mask occupancy are still recorded for CFM as
diagnostics. With zero region weight, `train/total_loss` must equal
`train/velocity_mse`; the mask does not affect optimization or inference.

The comparator is not the legacy-paper CFM reproduction. Results from the two
protocols must remain labelled separately because the legacy entry uses a
different split, normalization, optimizer schedule, seed, and evidence schema.
