# Scaling book exercises + Chinchilla laws + MoE Pallas kernel

This repo contains the main deliverables for the assignment specified in [this blog post](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html).

## Contents

| What | Where |
|------|-------|
| Scaling law report (dense vs MoE Chinchilla) | [`reports/scaling_law_report.pdf`](reports/scaling_law_report.pdf) |
| Scaling book exercises | [`reports/scaling_book_exercises.pdf`](reports/scaling_book_exercises.pdf) |
| Fused up/down MoE Pallas kernel | [`moe_kernel/kernels.ipynb`](moe_kernel/kernels.ipynb) |
| From-scratch JAX transformer | [`nishrelan/chinchilla`](https://github.com/nishrelan/chinchilla) |

### Notes

- The training code used for the scaling-law experiments is written in pure JAX (no flax, optax, etc.).
- For the chinchilla report, all training was done on a v5e-1 TPU on Google Colab.

## AI Assistance

For the full learning experience, I did the bulk of this work "manually". I used AI assistance for this project in the following ways:

- Boilerplate profiling / testing code for the Chapter 10 notebook
- Sanity checks for bugs in my from-scratch implementations
- Building a pipeline to convert my handwritten book problem solutions to LaTeX
- LaTeX formatting for both the scaling book problems and the scaling law report (all content is my own)

