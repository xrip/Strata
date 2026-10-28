# xrip/Strata: upstream Strata for a box with 2x CMP 50HX

This fork is [Niko1221/Strata](https://github.com/Niko1221/Strata) with one patch, and the settings measured fastest on
one machine: 2x Xeon E5-2670 v2 (Ivy Bridge-EP, AVX only, 2 NUMA nodes), 80 GB DDR3-1866, 2x CMP 50HX (sm_75, 20 GB) on
a layer split over PCIe 2.0 x16, a third CMP 50HX (10 GB, PCIe x1) for pictures, Qwen3.8-Flash-Next IQ3_XXS.

Base: upstream 0.2.7 (`5e8a4cca`, 2026-10-27). The fork's history before this base (the old patches, every sync and
A/B since 2026-10-10) is the tag `fork-main-2026-10-28`.

## The patch: parked conversations on a disk

`--conversation-disk-dir DIR` (needs the conversation cache): a conversation the RAM cache evicts, or one its budget
refuses, is written to DIR in the background instead of dropped, and read back when a request continues it.
`--conversation-disk-mib N` caps the files (default 32768). `--conversation-disk-persist off|on|exit-only`: `on` keeps
the files for the next start (the engine writes the current and the parked conversations at QUIT and ignores SIGINT,
SIGTERM and SIGHUP; the server waits 300 s for it), `exit-only` writes at QUIT only. A file is read back only when that
is faster than reading its extra tokens again (both speeds measured as the engine runs). The Monitor's cache card shows
what is on the disk. Also: an older, shorter copy of the conversation that shares at least 90 % of its tokens with the
outgoing one is dropped, so the disk is not filled with stale copies.

Measured here: 286,199 tokens back from the HDD in 41.7 s (a cold read: ~136 s); from NVMe 165K tokens in 6.7 s
against 77.6 s, same greedy answers. Upstream PRs: #1937 (the disk tier), #3404 (the older copies). Upstream has no
disk tier of its own (0.2.7 keeps parked conversations in RAM, opt-in in VRAM).

Limits: not with upstream's opt-in `STRATA_PARK_ASYNC` / `STRATA_PARK_VRAM_MIB` (their images are filled after the
park returns; the disk writer would copy them unfinished). A new engine version refuses the saved files once (their
identity holds the version).

## Settings for this box

Build: `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ISA_FLOOR=avx` (the released engines need AVX2:
on this CPU they stop at start). The config (`strata-iq3_xxs.json`), on top of what setup writes:

```json
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_CPU_SHARE": "auto", "STRATA_PREFILL_CPU_SHARE_MAX": "8192",
        "STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_SM75_PERF": "1", "STRATA_EXP_MODE": "8",
        "STRATA_SIDE_MIXER": "2", "STRATA_ATTN_DAILY": "1", "STRATA_B6_MMVQ_ROWS": "1",
        "STRATA_PREFILL_LAYER_MAJOR": "0"},
"args": [..., "--pool-workers", "19", "--max-context", "393216", "--kv", "int8",
         "--conversation-cache-mib", "16384", "--conversation-cache-slots", "16", "--conversation-cache-min-tokens", "32768",
         "--conversation-disk-dir", "/mnt/hdd12tb/conversation-cache", "--conversation-disk-persist", "on",
         "--prompt-cache-spread", "--prompt-cache", "8", "--vision"]
```

- `STRATA_PREFILL_LAYER_MAJOR=0`: 0.2.7 reads a long prompt layer by layer on a CUDA layer split by default; here it
  is off by itself only because CUDA1 has ~450 MiB free. Forced on (2 GB VRAM kept free): a 64K prompt 53.0 s against
  30.5 s (it streams ~74 GB of experts over PCIe 2.0; the chunk path with the CPU share does better), decode 2 % slower.
- `STRATA_ROUTE_TAIL_SKIP=0` keeps the model's exact output (the tail skip gave nothing here).
- Pictures: the CUDA picture reader (`build-vision-cuda/bin/strata-vision`) on the third card (`vision.cuda_device: 2`).
  Layers or a helper cache on that x1 card were slower.
- Memory: no `numactl`. Node 0 has a 32 GB and a 16 GB stick, so its top 16 GiB (physical 34-50 GiB) is on one channel;
  where the arena lands there differs per start (0-38 %, `~/opt-avx1/physmap.py PID arena`), about 2 % of decode.

## Measured 2026-10-28 (64K cold read, decode 6 x 512 tokens at ~8K context, each start with its real settings)

| | 64K read | decode | ms per verify window |
|---|---:|---:|---:|
| A: the fork on 0.2.4.2 + these settings (3 starts) | 30.74-30.90 s | 91.1-94.9 tok/s | 28.1-28.3 |
| C: clean upstream 0.2.7, setup's defaults (`--gpus 0,1`, 256K) | 43.66-43.70 s | 78.7-78.9 tok/s | 34.4-34.8 |
| D: clean upstream 0.2.7 + these settings | 30.48-30.56 s | 91.6-93.0 tok/s | 27.9-28.5 |
| D + layer-major forced on | 52.96-53.13 s | 88.8-89.8 tok/s | 28.8 |

The settings give +42 % prompt reading and +16-20 % decode over upstream's defaults here; 0.2.7 itself is equal to
0.2.4.2 (reads ~1 % faster). Upstream's setup offers all three cards (the x1 card as a helper); only `--gpus 0,1` was
tested. Tools on the box: `~/opt-avx1/quick.sh` (this table), `perf.sh` (the idle governor off for timing).

## Syncing

`git fetch upstream && git merge upstream/main` on `main`; the patch lives in `conversation_disk.*`, a few blocks of
`generate.cpp` (marked `--conversation-disk-dir`), `conversation_cache.hpp` (`set_on_evict`, `shared_tokens`, the 90 %
rule) and the server's cache card. Skip the speed test when nothing on this box's path changed (DeepSeek, AVX2/AVX-512
kernels, one-GPU features); otherwise `quick.sh` against the deployed build.
