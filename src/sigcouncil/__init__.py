"""Signal Council — evidence-first stock research, ranking and prediction.

Core invariants (enforced in code, tested in CI):
1. Point-in-time: features may only use data with observed_at <= prediction time.
2. The prediction ledger is append-only; history is never rewritten.
3. Probabilities trace to empirical distributions or are clamped (cold start).
4. Every displayed datum is tagged FACT / MODEL_ESTIMATE / AI_INTERPRETATION
   with source + timestamp.
5. "NO HIGH-CONVICTION OPPORTUNITIES TODAY" is a first-class output.
"""

__version__ = "0.1.0"
MODEL_VERSION = "rulecomposite-0.1.0"
