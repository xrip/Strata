# xrip/Strata: upstream Strata for a box with 2x CMP 50HX

This fork is [Niko1221/Strata](https://github.com/Niko1221/Strata) with two patches (the disk tier and the third one, the
GDN scan on sm_75; the second, tool search, is upstream's since 0.2.10), and the settings measured fastest on
one machine: 2x Xeon E5-2670 v2 (Ivy Bridge-EP, AVX only, 2 NUMA nodes), 80 GB DDR3-1866, 2x CMP 50HX (sm_75, 20 GB) on
a layer split over PCIe 2.0 x16, a third CMP 50HX (10 GB, PCIe x1) for pictures, Qwen3.8-Flash-Next IQ3_XXS.

Base: upstream 0.2.10 (`ffa80d73`, engine 0.1.56, merged and built on the box 2026-11-03).
On this box's path 0.2.10 brings: `STRATA_QSA_FUSE_NR` on by default (the QSA layers' norm + rope in one pass; its
no-fold guard is HIP-only, so it was checked here: `qsa_nr_parity` built for sm_75 by hand, 44 of 44 cases bit for
bit on the x1 CMP 50HX, 2.25 -> 1.79 ms a QSA layer at 8,191 tokens, ~0.1 % of a read); a layer split reads prompts
chunk-major by default (the config's `STRATA_PREFILL_LAYER_MAJOR=0` is now the default); the tool search code of
#4663 / PR #4700 is upstream's now. Not used here: the all-cards split engine (sFNX; sm_86 / gfx1201, UD-Q4 models
that fit the cards), `STRATA_QSA_FUSE_GATE` and `STRATA_PA_TRIM` (HIP kernel only). New opt-ins not measured yet:
`STRATA_PF_GROUP_SCRATCH`, `STRATA_PF_HC_REGION` (bitwise; smaller prompt buffers, the prompt path now borrows
4.38 GiB a card here), `STRATA_PREFILL_ROUTED_STREAM`. Base before: upstream 0.2.9 (`c62d5e3a`, 2026-10-31);
0.2.9 brought: the server sets `STRATA_GDN_CONVL2=2` (the prompt's conv, SiLU and q/k norm in one pass; not bitwise,
KL 0.0019 on upstream's card) and `STRATA_GDN_TC=1` (needs sm_80; the third patch below runs it here with `=2`) on
Linux CUDA. Measured with `gdn_harness` on the x1 CMP 50HX: one layer's conv 1.82 -> 1.39 ms at an 8,192-token chunk
(the conv is ~1 % of a prompt read here, so ~0.3 %); the recurrence (`pipe`) is 11.1 ms a layer, ~7 % of a read; calls shown in a code fence are not run (#4702, #4703). 0.2.8 (`870aedd9`, 2026-10-30) brought: `--expert-cache auto` leaves 256 MiB more VRAM free (#4474: ~170 fewer slots per card here, where the cards
now end at 280-380 MiB free and the engine warns at every request; not measured yet), the engine and server wait for
an exiting engine's memory (#4051), and the server's tool-call repairs (`tool_call_recovery` on by default).
`STRATA_ROUTE_PRIOR` is off by default again (it never turned on by itself on these 20 GB cards). The fork's history before this base (the old patches, every sync and
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

## Was the second patch: Anthropic's tool search on /v1/messages (#4663), upstream's since 0.2.10

Upstream 0.2.10 took this commit as it was (`9efac55f`) and added its own follow-up (`6a8752f9`: on the Anthropic and
Responses routes too, a tool the conversation already called counts as declared), so the fork carries no code for it now.
What it does: Claude Code (tool search is on by default behind a gateway) declares `ToolSearch` and a `defer_loading`
`DeferredToolPlaceholder`, and names its MCP tools only in the conversation. Upstream ignored `defer_loading` and dropped
`tool_reference` blocks, and returned a call for an undeclared name as text with `end_turn`: Qwen often calls a
context-mode tool without loading it first, and the session stopped (8 such stops in civone-gbc sessions, 2026-10-29/30).
Now, as on Anthropic's API: a deferred tool is callable but not in the prompt's tool list; a `tool_reference` puts its
definition where it stands (the prompt's start stays the same, so the prompt cache too, where each loaded tool used to
change it); any other reference is left out as before; with a deferred tool in the request, a call for an undeclared
name is a `tool_use`. Checked against Claude Code 2.1.314 with a scripted fake API: it puts a loaded tool in `tools` with
`defer_loading: true` and runs a direct call to a tool it holds back. Code: `serve/frontend.py` (`anthropic_tools`,
`_tool_references`, `OutputParser(deferred=...)`), `serve/server.py` (the `/v1/messages` route), tests
`serve/test_tool_search.py`. Other routes and clients keep upstream's rule (0.2.9 adds its own fix there: on the OpenAI
route a tool the conversation already called counts as declared). Upstream PR #4700. Live since 2026-10-30 14:39:
about 1,700 answers in two Claude Code sessions in 4 hours, no stops; the model loaded each tool with `ToolSearch` first,
so the direct call has not come up live yet.

## The third patch: the tensor-core GDN recurrence on sm_75 (#4876)

`STRATA_GDN_TC=2` runs upstream's tensor-core scan of the prompt's DeltaNet recurrence on Turing too (`=1` leaves sm_75
on the old `pipe` kernel). Upstream's scan needs sm_80 and 98-125 KB of shared memory a block; here each m16n8k16
product is two m16n8k8 steps, `cp.async` is a plain copy, and every buffer is single (62,480 B of the 64 KB opt-in), so
the next chunk's load does not overlap the products. Not bitwise: the same rounding as upstream's `=1` (f16x3 split
products; `gdn_harness` against its FP64 reference PASS, 10 stress cases). One layer's recurrence on the x1 CMP 50HX:
11.19 -> 5.12 ms at 8,192 tokens, 2.77 -> 1.28 at 2,048, 0.39 -> 0.21 at 300. Live since 2026-10-31: prompt reads of
4-46K new tokens after a cached 10K start 6.0 % +- 5.6 % faster than on 0.2.8 (6 reads against 28; 0.2.9 and this
together). Code: `src/prefill/gdn_tc.cuh` (`ONE`), `gdn_rec_tc` in `src/prefill/kernels.cu`. Upstream PR #4876.
On by default here: the fork's server sets `STRATA_GDN_TC=2` where upstream's sets `=1` (`GDN_STACK_CUDA` in
`serve/server.py`, checked by `serve/test_parallel.py`), and the box's configs set it too. The engine says nothing
when it falls back to the old kernel, so check it after every sync (below).

## Settings for this box

Build: `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ISA_FLOOR=avx` (the released engines need AVX2:
on this CPU they stop at start). The config and what each setting gives here: the top of [README.md](../README.md).
More on some of them:

- `STRATA_PREFILL_LAYER_MAJOR=0`: 0.2.7 reads a long prompt layer by layer on a CUDA layer split by default (0.2.10
  no longer does, so the setting only keeps older builds the same); here it is off by itself only because CUDA1 has ~450 MiB free. Forced on (2 GB VRAM kept free): a 64K prompt 53.0 s against
  30.5 s (it streams ~74 GB of experts over PCIe 2.0; the chunk path with the CPU share does better), decode 2 % slower.
- `STRATA_SPLIT_MTP_BATCH=1`: on a layer split the draft layer's prompt K/V was built by its own pass, 8 rows at a
  time, while the next chunk waited (405-473 ms per 8K chunk in a `STRATA_PREFILL_TIMING` run); this builds it in one
  batched pass on CUDA1. Only the drafts can change, not the model's tokens. A 64K cold read 30.70 / 30.61 s against
  29.19 / 29.46 s (-4.3 %), decode at 8K 93.4 against 94.8 tok/s, 28.45 against 28.35 ms per verify window
  (2026-10-28, four starts A / B / A / B).
- `STRATA_ROUTE_TAIL_SKIP=0` keeps the model's exact output (the tail skip gave nothing here).
- Pictures: the CUDA picture reader (`build-vision-cuda/bin/strata-vision`) on the third card (`vision.cuda_device: 2`).
  Layers or a helper cache on that x1 card were slower.
- Memory: no `numactl`. Node 0 has a 32 GB and a 16 GB stick, so its top 16 GiB (physical 34-50 GiB) is on one channel;
  where the arena lands there differs per start (0-38 %, `~/opt-avx1/physmap.py PID arena`), about 2 % of decode.

## Tried on sm_75 and dropped (2026-11-03)

The decode hyper-connection read (96 a verify window: two a layer) is ~66 us at a 5-row window on the x1 CMP 50HX
(`fused_gr_bench 300 1 6 8` with `STRATA_HC_SPLIT=2 STRATA_SM75_PERF=1`, the engine's staged read: norm 12, down 30,
up 31 us; the weights alone need ~34 us), so ~6 ms of a ~30 ms window. Bitwise variants measured there, none kept:
- rows a warp in the staged down kernel (the idea of upstream's `STRATA_GR_V3_PF` bits 7 / 8, whose kernels exist only
  in the v3 read, another summation order): 2 rows 30.4 -> 34.1 us, 4 rows 52.0 us at 5 rows (20 or 10 blocks for 56 SMs);
- `gr_up75` with two-plane `lo` (no bank conflicts) and two tokens a pass: -7 % at 4 and 6 rows, none at 3 and 5;
- `STRATA_GR_NORM_FAST=1` (CUDA opt-in): the same time; the same shared-memory carveout for the norm and up kernels: none.
- Seen, not changed: upstream's Turing rule of token pairs in the down kernel (measured on an RTX 2080 Ti) costs here at
  6 rows (40.4 against 33.5 us) and gives ~1 us at 4 rows; 6-row windows come only from batch slots.

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
rule) and the server's cache card. The third patch: `gdn_tc.cuh`, `gdn_rec_tc` and `GDN_STACK_CUDA` (keep `"2"` when
upstream changes that line). After every sync, on the x1 card: `CUDA_VISIBLE_DEVICES=2 STRATA_GDN_TC=2 gdn_harness
--T 5000 --chunk 2500 --cuts 37 --paths auto,pipe --ref-heads 0,17,47` must show `tc->tc` and PASS on the `auto` line
(`tc->pipe` means the sm_75 scan is off), and `python -m pytest serve/test_parallel.py -k GdnStack` must pass. Skip the
speed test when nothing on this box's path changed (DeepSeek, AVX2/AVX-512 kernels, one-GPU features); otherwise
`quick.sh` against the deployed build.
