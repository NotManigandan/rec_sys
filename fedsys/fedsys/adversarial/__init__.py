"""
fedsys.adversarial
==================
Plug-in adversarial layer for the fedsys FL framework.

Two independent sub-packages — use either or both without touching
the core networking/training code:

  fedsys.adversarial.attack
      Data-poisoning attack via synthetic user profile injection.
      Ported from recsys/federated/attack.py and benchmark.py.

  fedsys.adversarial.defense
      Robust aggregation methods that replace the plain FedAvg step:
      clip_mean | clip_trimmed_mean | focus_clip_mean | focus_clip_trimmed_mean
      Ported from recsys/federated/bpr.py (aggregate_server_states).

  fedsys.adversarial.eval
      Target-exposure evaluation: Hit@K / NDCG@K for benign users +
      target_hitrate@K for the victim genre segment.
      Ported from recsys/federated/eval.py.
"""
