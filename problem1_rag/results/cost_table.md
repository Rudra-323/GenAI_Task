| Vectors | Self-hosted Chroma | Managed pods p1 (always-on) | Managed pods s1 | Managed serverless, 100K queries/mo | Managed serverless, 5M queries/mo | Saving vs p1 pods |
|---|---|---|---|---|---|---|
| 100K | $32 (t3.medium) | $70 (1 pod) | $70 | $50 | $50 | +54% |
| 1M | $63 (t3.large) | $70 (1 pod) | $70 | $50 | $164 | +10% |
| 10M | $191 (r6i.xlarge) | $350 (5 pod) | $70 | $50 | $1,644 | +45% |
