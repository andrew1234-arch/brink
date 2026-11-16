# ROI: closed API vs Brink-protected Tiny Aya

1,000,000 requests a month of 400 input and 128 output tokens.
Serving speed measured in experiment 04: 0.61 s per reply on the evaluation GPU
(batched, greedy, plain `transformers` generation; a serving engine such as vLLM would be faster). The per-1k
cost of the self-hosted model assumes the GPU is kept busy; the monthly cost counts an always-on GPU.

| | Model | Per 1,000 replies | Monthly cost |
|---|---|---|---|
| Closed API | gpt-6-luna | $0.104 | $104 |
| Closed API | gpt-6.1-terra | $1.040 | $1,040 |
| Self-hosted | CohereLabs/tiny-aya-global + Brink adapter on NVIDIA T4 16 GB (168 GPU-h, 1 GPU) | $0.084 | $360 |

- Per 1,000 replies, gpt-6-luna costs 1.2× as much
  as the self-hosted model.
- At this traffic, the always-on GPU costs **$256 a month more** than gpt-6-luna.
- An always-on GPU beats gpt-6-luna above **3,461,538 requests a month**.
- One-off fine-tune with Brink: $0.80; no payback at this traffic.

Prices: gpt-6-luna: https://developers.openai.com/api/docs/pricing (standard tier, read 2026-11-16); gpt-6.1-terra: https://developers.openai.com/api/docs/pricing (standard tier, read 2026-11-16); GPU: https://huggingface.co/pricing (Inference Endpoints, NVIDIA T4 on AWS, read 2026-11-16).
