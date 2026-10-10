# This fork: Strata on 2x CMP 50HX (sm_75) + 2x Xeon E5-2670 v2 (AVX1)

Upstream is [Niko1221/Strata](https://github.com/Niko1221/Strata); this fork is
[xrip/Strata](https://github.com/xrip/Strata), a direct GitHub fork of it:

- `main`: an exact copy of upstream `main`, nothing of ours (GitHub's "Sync fork" works on it without conflicts);
- `cmp50hx`: this branch, upstream `main` (0.1.41, `fb58e0d`) plus a short list of patches for this box;
- `pr/*`: each generic patch on top of upstream `main`, sent upstream as a PR.

The numbers for each patch are in its commit message; the older notes in [AVX1_SM75_PLAN.md](AVX1_SM75_PLAN.md) can
be out of date. The earlier fork (xrip/strata-dual-cmp50hx, branch `avx1-sm75-opt`) is retired.

## Syncing with upstream

```sh
git fetch upstream
git merge upstream/main          # on cmp50hx; the history is shared, so a plain merge works
```

Rules that keep the next merge small:

- A patch here is small and on its own: new files where possible, a few lines in upstream files.
- A patch that upstream takes (or replaces with its own) is dropped here.
- Each generic patch is also kept as a branch on top of upstream `main` (`pr/*`), ready to send upstream.

## Patches on top of upstream

| commit | what | switch |
|---|---|---|
| `8aa74eb` | Q2_0 expert rows on AVX-only CPUs (`q2_avx1.cpp`) instead of ggml-cpu's scalar dot | always, on an AVX-only CPU |
| `7b07ac7` | tokenizer: a long message's ids remembered whole, on top of upstream's piece cache | always |
| `11a4f15`, `18ec242` | QSA prompt-path block scores: a thread per query head, four lanes per query (Turing) | `STRATA_SCORES_THREAD=0` = old |
| `1eb1010` | prompt attention, int8 K/V: next chunk in registers, K and V in one buffer (Turing) | none (same bits) |
| `94292d6` | gate/up MMQ reads the experts in place (a patched copy of llama.cpp's `mmq.cuh` in the build tree) | `STRATA_MMQ_INPLACE=0` = old |
| `e8f68e0`, `cc3fd60`, `3a5e69e`, `8f93be3` | `dual3090/verify/`: agentbench (gate and compare), flashbench corpus, prefill_probe; `prefill_mmq_bench` | tools |

## Dropped in the sync (from the `avx1-sm75-opt` branch)

| commit | why |
|---|---|
| `87fa125` no-AVX2 support | upstream's older-CPU build (`STRATA_ISA_FLOOR=avx`) does the same |
| `22f2f64` `STRATA_HC_FP16` | upstream's `STRATA_BF16_TC=1` runs these and more of the prompt path's BF16 products on FP16 tensor cores |
| `8de94b5` residency table race | upstream fixed it (`res_put` waits for each copy) |
| `b7bf59d`, `c2e0e79`, `240744b` decode kernels | upstream reworked the same kernels (and stages the IQ2 codebooks itself); about +5 % decode together, not worth the merge conflicts. Bring one back only if the A/B below shows upstream slower |
| `b44bb09` MMQ IQ3_XXS codebook | -0.3 % prompt read in the agent session: too small for a patch to llama.cpp's MMQ |
| dual-3090 peer mode, Docker files | not used here: `--layer-split auto` is much faster on this PCIe Gen2 box |

xrip/llama.cpp-avx1-numa-sm75 was checked too: its SM75 MMQ tuning is for dense Q4_K/IQ4_XS only, its MMVQ and
codebook work is in files Strata does not use (Strata has its own decode kernels), and its AVX1 panel kernels are
repack paths Strata does not call. Strata keeps the pinned llama.cpp (`3cf03257`).

## A/B on the box (2026-10-10)

2x CMP 50HX + 2x Xeon E5-2670 v2, Flash-Next IQ3_XXS, `--layer-split auto`, the gate config (`--adapt-swaps 0`,
`--pcie-frac 0.17`), agentbench 34K -> 190K (36 turns, +~4.4K tokens each), a fresh server per arm, inside `perf.sh`.
Base: `build/strata` of `avx1-sm75-opt` (`b44bb09`). This branch built as `build-gcc14` plus `-DSTRATA_ISA_FLOOR=avx`.

| arm | median incremental read | decode, answers of 100+ tokens |
|---|---:|---:|
| base, two runs | 708.4 / 707.5 tok/s | ~60 tok/s |
| this branch, two runs | 692.9 / 693.9 tok/s (-2 %) | ~69 tok/s (+15 %) |
| this branch + `STRATA_BF16_TC=1` | 764.0 tok/s (+8 %) | ~72 tok/s (+19 %) |
| this branch + `--pipeline-windows 2` | 691.0 tok/s (-2.5 %) | ~73 tok/s (+21 %) |

- Two runs of one build differ by under 1 %.
- Decode only over the answers that ran to 100+ tokens (3-4 early turns per arm, 34K-48K): most answers in this
  session stop after 15-40 tokens, so a median over all turns measures answer length more than speed.
- The text differs from the base on every turn (other kernels); the answers were read and are on topic.
- Not tried: `STRATA_GDN_CHUNKED=2` (slower on cards under 128 SMs in upstream's own measurements) and
  `STRATA_FS_SLOTS` (lost on every box upstream measured; the decode cache here hits ~97-99 %).

### Second series: on top of `STRATA_BF16_TC=1`

The same setup; every arm with `STRATA_BF16_TC=1`.

| arm | median incremental read | note |
|---|---:|---|
| `STRATA_BF16_TC=1`, run 2 | 764.5 tok/s | run 1: 764.0; decode the same |
| + `--pipeline-windows 2` | 757.2 tok/s (-1 %) | decode on 100+ token answers 72.0 vs 72.3: no gain on top of BF16_TC |
| + CPU prompt share (`STRATA_PREFILL_CPU_SHARE=auto`, `_MAX=8192`) | 791.8 tok/s (+3.6 %) | cold 34K TTFT 20.53 -> 19.59 s |
| live config (adaptive swaps on) | 759.0 tok/s | the same as swaps off within the noise |
| + `--adapt-async 1` | - | does not switch on without `--resident-experts` (the log says so): stopped |

The CPU prompt share was checked for answer quality with `prefill_probe.py` (24 prompts of 8K-24K tokens, 48 greedy
tokens, a fresh server each): against `STRATA_BF16_TC=1` alone 20/24 answers identical, 23/24 with the same start
(first difference at character 133, median). A rerun of the reference: 24/24 and 24/24. For scale, a different prompt
chunk size that Strata accepts (`--prefill 4096`) gave 17/24 and 23/24 on this box earlier.

Not tried: `STRATA_EXCHANGE_ROTATE=1` (needs `--adapt-async 1`) and `STRATA_ADAPT_LAG=2` (decode only, with the GPU
cache already hitting 97-99 %); agentbench gives too few long answers to see a few percent of decode.

## Settings for this box

Against `avx1-sm75-opt`: prompt reading +8 % (+12 % with the CPU share), decode about +19 % (few samples).
Answers are not word for word the old fork's (these switches round differently); they were read and are correct.

Build: `-DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ISA_FLOOR=avx`, GCC 14, CUDA 13.3. Config
(`strata-<model>.json`), on top of what setup writes (`--layer-split auto`, `--kv int8`, `--spec 4`, `--mtp`):

```json
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_CPU_SHARE": "auto", "STRATA_PREFILL_CPU_SHARE_MAX": "8192"}
```
