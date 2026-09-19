"""Synthetic retail world generator and fault oracle (Milestone A0).

The generator emits versioned snapshots of a simulated retail transactional
dataset together with a ground-truth manifest describing every injected fault,
its affected entities, the pipeline stage where it first appears and the
expected decision class.

Synthetic truth is a valid oracle for the deterministic and temporal layers.
It is *not* evidence for semantic-model quality; see docs/synthetic-data.md.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
