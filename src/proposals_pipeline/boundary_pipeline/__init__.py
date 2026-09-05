"""Modular temporal action-boundary proposal pipeline.

Three proposal sources (ABD on V-JEPA embeddings, optical-flow motion,
SAM3 hand-object contact) each turn a cached per-video signal into scored
boundary proposals; the proposals are merged with temporal NMS and every
threshold is tuned against ground truth (`tuning.py`).
"""
