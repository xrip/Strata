# xrip/Strata: upstream Strata for a box with 2x CMP 50HX

`main` is upstream's head with three patches on top, one commit each, and this file and the README's fork section.
The history before 2026-11-24 (every sync, A/B and dropped patch since 2026-10-10) is the tag `fork-main-2026-11-24`;
before 2026-10-28, `fork-main-2026-10-28`.

The box: 2x Xeon E5-2670 v2 (Ivy Bridge-EP, AVX only, 2 NUMA nodes), 80 GB DDR3-1866, 2x CMP 50HX (sm_75, 20 GB) on a
layer split over PCIe 2.0 x16, a third CMP 50HX (10 GB, PCIe x1) for pictures; Swift 1.5 Qwen3.8-Flash-Next IQ3_S with
the FP8 per-layer token table. The config and what each setting gave: the README's fork section.

## The patches

| Patch | Upstream PR | What it does | Measured here |
|---|---|---|---|
| The SSD tier on a layer split, saved at QUIT | #6288 | upstream's SSD tier and its `--conversation-spill-persist` take one stage only; this writes a layer split's conversation as one session file per stage (`conv-N.session`, `conv-N.sK.session`; the index is `SPM2` with the file count, a one-stage entry keeps `SPM1`), and with the flag QUIT writes the current and every RAM-parked conversation to the tier (signals ignored until the save ends; the server waits up to 300 s) | 3 conversations (42K / 42K / 166K) saved at SIGTERM in 9.2 s (5.7 GiB), back after a restart in 2.7 / 3.1 / 8.9 s against 20.4 / 20.8 / 64.7 s cold, the same answers (2026-11-24) |
| `STRATA_GDN_TC=2` | #4876 | the prompt's DeltaNet recurrence on the tensor cores on sm_75 too (one 62,480 B plan, m16n8k16 as two m16n8k8, plain copies); `=1` leaves sm_75 as upstream has it. Upstream's rounding, not bitwise | recurrence 11.1 -> 5.1 ms a layer at 8K; prompt reads ~6 % faster (2026-11-01) |
| 5-8 row windows on the interleaved kernels (sm_75) | #5137 | `kIlRows58_75` and the 5-8 column activation copy as two 4-column planes on sm_75; bitwise | IQ4_XS 2560x10240 at 5-6 columns -16..-22 % per kernel; in the engine within noise (5-8 column windows are ~0.6 ms of a 29 ms verify window) |

## Syncing

`git fetch upstream && git merge upstream/main` on `main` (the patches are small; conflicts so far were in the spill
tier's files). Build from `engine/` (`-DSTRATA_ISA_FLOOR=avx -DCMAKE_CUDA_ARCHITECTURES=75`, GCC 14, CUDA 13.3). After
every sync, on GPU 0 with Strata stopped:
- `CUDA_VISIBLE_DEVICES=0 STRATA_GDN_TC=2 gdn_harness --T 5000 --chunk 2500 --cuts 37 --paths auto,pipe --ref-heads
  0,17,47`: the `auto` line must show `tc->tc` and PASS (`h=aea57e39` so far); `tc->pipe` means the sm_75 scan is off;
- `STRATA_SM75_PERF=1 mmvq_il_parity`: OK;
- `conversation_spill_test`, `conversation_cache_test`.
Then the speed gate when our path changed: `~/opt-avx1/chunk.sh A N N A` (64K and 172K cold reads, 6 x 512 tokens at
~8K; compare decode by ms per verify window). Skip it when nothing on our path changed. Every config keeps
`"STRATA_GDN_TC": "2"`.

## Last sync

Upstream 0.2.22 (`c23477403`, engine 0.1.69), 2026-11-26, live; no conflicts. No speed gate: no CUDA kernel or decode change on our
path (sm_75 SASS of the new binary against 0.1.67: 3086 of 3086 kernels identical). New opt-in `STRATA_SPLIT_LEND_RESERVE`
(#7089) acts only on a RAM-budgeted expert copy; ours is the whole-pack arena, so it does nothing here. Server: a request that
names no sampling gets the model card's sampling again under `--mtp` (#7070; greedy stays for Qwen3.6 only); our config sets
`--conversation-cache-mib`, so the RAM-sized cache (#7044) does not apply.

Upstream 0.2.21 (`d46f5f268`, engine 0.1.67), 2026-11-24. Gate `A N N A` against 0.1.66: 64K 28.23 / 28.24 s
against 27.71 / 28.24, 172K 65.24 / 65.53 against 64.00 / 65.31, ms per verify window 30.02 / 29.73 against 30.17 / 30.16:
equal. GDN `tc->tc PASS h=aea57e39`, `mmvq_il_parity` OK, the conversation suites pass.

## Dropped (measured, not carried)

The older-copies rule of the conversation cache (#3404, closed: no copy dropped by any rule in a 2 h multi-agent
session); `STRATA_IQ4_W4A8` on sm_75 (#5498, went with upstream's MMQ path in 0.2.16); the fork's own disk tier (#1937,
replaced by upstream's SSD tier); Volta's `STRATA_GR_V70` widened to sm_75 (bitwise, 0.5-1 % slower); `STRATA_DOWN_SUB8`;
the Monitor's disk row. Settings measured and not used are in the README's table notes and in the tagged history.
