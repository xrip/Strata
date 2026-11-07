# xrip/Strata: upstream Strata for a box with 2x CMP 50HX

This fork is [Niko1221/Strata](https://github.com/Niko1221/Strata) with four patches (the disk tier, the third one, the
GDN scan on sm_75, the fourth, 5-8 row decode windows on the interleaved kernels, and the fifth, upstream's int8 IQ4
prompt kernel on sm_75; the second, tool search, is upstream's since 0.2.10), and the settings measured fastest on
one machine: 2x Xeon E5-2670 v2 (Ivy Bridge-EP, AVX only, 2 NUMA nodes), 80 GB DDR3-1866, 2x CMP 50HX (sm_75, 20 GB) on
a layer split over PCIe 2.0 x16, a third CMP 50HX (10 GB, PCIe x1) for pictures, Swift 1.5 Qwen3.8-Flash-Next IQ3_S
with the FP8 per-layer token table (the original IQ3_XXS until 2026-10-31, Swift IQ3_XXS until 2026-11-04).

Base: upstream 0.2.12 (`146df3c4`, engine 0.1.58, merged, measured and deployed on the box 2026-11-07; 95 commits,
no conflicts). On this box's path: fixes only by default (the prompt MMQ row tables get the stream-k fixup's over-read
margin - the Xid 31 "prefill copy_i32: an illegal memory access" on long prompts on a CUDA layer split, #1468; the copy
stream's flag B stays behind queued copies; the PLE reader folds a decode step's rows without a hash table) and three
opt-ins. `STRATA_IQ4_W4A8` (an int8 kernel for the prompt's IQ4 expert products, sm_80+) is ported to sm_75 as the
fifth patch and on here. Not taken: `STRATA_SELECT_TWO_PASS` (sm_80+: a single-pass TF32 QSA block scorer with an
error bound, 3xTF32 again only near the top-k cut-off; sm_75 has no TF32, so a port needs an FP16 first pass and a new
bound against the exact FP32 thread scorer; that scorer was 5.6 % of GPU 0's kernel time in a 2.6 s nsys sample of a
140K read on 2026-10-08, so likely 2 % or less of a 172K read), `STRATA_ADAPT_FETCH` (the PCIe share's fetched experts become resident; upstream +5-10 %
decode on one card, but the engine turns it off on a layer split, and it changes which experts the CPU computes, so
answers can change). 0.2.12 against 0.2.11 with the live config: 64K read +0.7 %, 172K +0.4 %, decode -1.2 %, all
within noise (the table "Measured 2026-11-07"). Server: 1,494 tests pass on the box; tool-call parser fixes (#5409,
#5276, #5281), Anthropic tool results in call order (#4832). Base before: upstream 0.2.11 (`9f7d977d`, engine 0.1.57,
2026-11-05). On this box's path 0.2.11 brings: a CUDA layer split refines its CPU/PCIe miss share from the CPU pool's speed
(`STRATA_PCIE_AUTO_SPLIT`, on by default on a discrete card; here the share went 0.17 -> 0.25-0.30, the pool measured
15.8-16.2 GB/s of expert bytes); a later stage keeps 256 MiB free for what it allocates after its cache (CUDA1: 6,389 ->
6,267 expert slots, 346 -> 602 MiB free, no LOW warning); and ~15 server fixes (an engine is ended only where the
request really ran, a client that stops reading no longer holds the turn, console output through a bounded queue).
The new share rule costs decode here, so the configs pin `STRATA_PCIE_AUTO_SPLIT=0` (the table "The 0.2.11 sync"
below). Adapted from it: the 32K prompt-chunk ceiling of cc 8.0+ cards, as `--prefill auto:16384` (the table "Prompt
chunk size" below). Not on this box's path: `STRATA_MMQ_KQ` (no K-quant experts here), the gfx12 ring, the Q5_0 expert
dot (no Q5_0 here), the all-cards split engine, DFlash2. The FP16 weight cache is now on with batch slots on Linux CUDA
when 3 GiB stay free (not on these cards). Base before: upstream 0.2.10 (`ffa80d73`, engine 0.1.56, 2026-11-03).
0.2.10 brought: `STRATA_QSA_FUSE_NR` on by default (the QSA layers' norm + rope in one pass; its
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

## The fourth patch: 5-8 row windows on the interleaved kernels (decode, bitwise; upstream PR #5137)

Two parts, as in the PR. `kIlRows58_75`: on Turing the interleaved path (`STRATA_SM75_PERF=1` here) had rows for 2-4
columns only, so 5-8 row windows ran `native_mmvq`'s plain kernels, 1.5-2.4x the interleaved ones at those widths (IQ4_XS
2560x10240 at 5 columns 113 against 49 us); the table was read off `mmvq_il_parity --bench --emit-table` three times on
GPU 0 (they agreed except one cell, set to 0). And in sm_75 code the position-major activation copy of 5-8 columns is
two 4-column planes: IQ4_XS read it with each 16-byte load touching every other 16 bytes (ncu: 17.2 against 9.5 sectors
a request, `lg_throttle` 5.8 against 0.7, 40 -> 70 us from 4 to 5 columns); now IQ4_XS -16..-22 % at 5-6 columns, the
K-quants unchanged (their block-major copy and the scales keep their layout). `mmvq_il_parity` (every case, T 1-8, rows
1/2/4, the table, every `--pf` variant): bitwise. Other architectures compile the old instructions (SASS compared).

End to end here: nothing measurable. A first version (fork `6d3d4853`) had the planes without the table, so the engine
never ran them on this card; its "~+0.3 %" was wrong. With both parts, arms alternated 3 x (8K, `--spec 4`, 6 x 512
tokens): planes only 29.05, both 29.17, the table only 29.33 ms per window, within noise. nsys: the 5-8 column
projections are ~0.6 ms of a ~29 ms window (0.6 s of GPU time at 6 columns, no 5-column launch, against 3.9 s
at 4 columns). `--batch 2` on the layer split runs pipelined groups of one row, so it has no 5-8 column windows
either. Kept because it is bitwise
and faster per launch; it matters where windows are wider. Code: `Q81Il` (`PLANES`) in `src/kernels/cuda/q8_1_il.cuh`,
`il_pm_at` and `kIlRows58_75` in `src/kernels/cuda/native_mmvq.cu`. Tried and dropped: planes for the block-major copy
(Q6_K 6144x2560 5 columns 52 -> 74 us), Q6_K 5 columns as 6 (11-62 % slower).

## The fifth patch: upstream's IQ4 W4A8 prompt kernel on sm_75 (`STRATA_IQ4_W4A8=2`)

0.2.12's opt-in `moe_iq4_w4a8.cu` (from upstream PR #4871) computes the prompt's IQ4_XS / IQ4_NL expert products with
MMQ's sums in another schedule: 128 weight rows a CTA, the 4-bit weights turned into int8 in registers, 128-value K
steps through a cp.async pipeline. It needs cp.async and the m16n8k32 s8 MMA (sm_80); sm_75 has ldmatrix and the
m8n8k16 s8 MMA. The patch: on sm_75 the next K step's pieces are loaded into registers before this step's MMAs and
stored to the other shared-memory stage after them, each m16n8k32 is four m8n8k16 from the same C (exact int32 sums),
and `arch_ok()` takes 7.5. The model here has IQ4_NL down products in 39 of 48 layers (2560 x 640, K = 640), so `=2`
(the IQ4_NL products only; one layer's IQ4_XS gate/up is left on MMQ).
- `prefill_iq4_w4a8_test` on GPU 0: all identical against MMQ (default, `--full`, `--tokens 97 / 2049 / 8191`,
  `--wide`; three routings each). `--full --bench`, MMQ time over the kernel's: IQ4_NL down +46..+59 %, IQ4_XS gate/up
  +23..+31 % (config 0; configs 1-3 slower). 255 registers, 12-36 bytes of spill a thread.
- Engine: prompt reads ~2 % faster, decode not on this path (the table "Measured 2026-11-07"); 20,288 products ran,
  128 stayed on MMQ (launches MMQ would split along K). The exactness gate: the same answers.
Code: the `STRATA_W4A8_SM75` blocks in `src/prefill/moe_iq4_w4a8.cu`; the test's arch check. The sm_80+ source is
upstream's word for word: sm_86 and sm_120 SASS of all 8 kernels identical to 0.2.12's. Upstream PR #5498 (branch
`pr/iq4-w4a8-sm75`); drop the patch here when it is merged.

## Settings for this box

Build: `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ISA_FLOOR=avx` (the released engines need AVX2:
on this CPU they stop at start). The config and what each setting gives here: the top of [README.md](../README.md).
More on some of them:

- `STRATA_PREFILL_LAYER_MAJOR=0`: 0.2.7 reads a long prompt layer by layer on a CUDA layer split by default (0.2.10
  no longer does, so the setting only keeps older builds the same); here it is off by itself only because CUDA1 has ~450 MiB free. Forced on (2 GB VRAM kept free): a 64K prompt 53.0 s against
  30.5 s (it streams ~74 GB of experts over PCIe 2.0; the chunk path with the CPU share does better), decode 2 % slower.
- `--prefill auto:16384` with `STRATA_PREFILL_CPU_SHARE_MAX=16384` (adapted from 0.2.11's 32K ceiling for cc 8.0+
  single cards; a bare `auto` takes 8,192 on any layer split, measured on 2x R9700): over PCIe 2.0 every chunk streams
  the experts the cards do not hold, so half as many chunks stream half as much. Measured: the table "Prompt chunk
  size" below. The share limit must follow the chunk, or the 16K chunks lose the CPU share. Not bit-exact against 8K
  chunks (no chunk size change is: upstream's 0.2.11 notes, near-tie block choices in the attention); kept as
  upstream keeps its own 32K default.
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

## Tried on sm_75 and dropped (2026-11-05)

The i-quant sign expansion of 0.2.11's HIP split engine (`750cbf0d`): a multiply and a mask in place of `__vcmpne4` /
`__vsub4`, which CUDA cards since Maxwell emulate. Ported to the decode dots (`iq_kernels.cu`, 16 places) and ggml's MMQ
tile loads (IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S). The same integers: a CPU check of every sign byte and of every
grid word against all 16 sign patterns (70,656 cases, 0 differ), and `mmq_expert_bench` output hashes equal for every
case. Not kept: nothing measurable in the engine, and the patch would have to rewrite the pinned llama.cpp.

| `mmq_expert_bench` on GPU 0, gate/up, medians of 3 alternated runs | 160 rows an expert (8K chunk) | 626 rows (32K) |
|---|---:|---:|
| IQ3_S | -2.3 % | -5.0 % |
| IQ3_XXS | -2.8 % | -5.1 % |
| IQ2_S | -0.3 % | -1.4 % |
| IQ2_XS | +1.0 % | -9.9 % |
| unchanged kernels (NVFP4, Q8_0, IQ4_NL), for the noise | -0.5 .. +0.3 % | -0.1 .. +7.6 % |

| The engine at 16K chunks (`chunk.sh F T H H T F`, its F and T starts) | 64K read | 172K read | decode at 8K |
|---|---:|---:|---:|
| F: live | 30.20 / 30.18 s | 70.21 / 70.30 s | 84.7 / 84.0 tok/s |
| T: + the sign patch | 30.17 / 30.43 s | 70.37 / 70.49 s | 84.1 / 84.8 tok/s |

## Measured 2026-11-07 (the 0.2.12 sync and the fifth patch; `~/opt-avx1/w4a8/win_0212.sh`)

`chunk.sh L N W W N L` as below (two starts an arm, interleaved; the live config, vision on GPU 2): L = the live
build (0.2.11), N = 0.2.12, W = 0.2.12 + the fifth patch with `STRATA_IQ4_W4A8=2`. The share of the arena in node 0's
one-channel part: 26-32 % in every start.

| Arm | 64K read | 172K read | decode at 8K |
|---|---:|---:|---:|
| L: 0.2.11 (live) | 29.27 / 29.93 s | 67.71 / 69.53 s | 82.0 / 82.2 tok/s |
| N: 0.2.12 | 29.62 / 29.99 s | 68.56 / 69.25 s | 80.9 / 81.3 tok/s |
| W: 0.2.12 + `STRATA_IQ4_W4A8=2` | 29.13 / 29.25 s (-2.1 %) | 67.41 / 67.81 s (-1.9 %) | 83.2 / 83.2 tok/s |

Both W starts read faster than both N starts at both lengths. The kernel does not run in decode; W's decode against N
(the same build options) is counted as noise between two binaries. The exactness gate (`gate_req.py`, CPU share 0,
`--adapt-swaps 0`, `--pcie-frac 0.17`, 16K chunks): N against N (the control) and W against N give the same three
200-token greedy answers after 2K / 20K / 70K prompts. The GDN check: `tc->tc PASS h=aea57e39` (as on 0.2.11).

## Measured 2026-11-05 (engine 0.2.11, Swift IQ3_S + FP8 PLE, vision on GPU 2 as live)

Every start with the page cache dropped (`dropwrap.sh`), inside `perf.sh` (the idle governor off); `chunk.sh`: a
warm-up, a 64K and a 172K prompt read cold (another first line, so nothing is reused), then 6 x 512 greedy tokens at
~8K context. Two starts an arm, interleaved; each cell gives both. Tools in `~/opt-avx1` on the box.

### Stock upstream against this fork + config (the README's table; `chunk.sh U L L U`, `mkcfg-report.py`)

U keeps the live config's model and capacity settings (393K context with YaRN, MTP, int8 KV, the conversation cache,
pictures) and takes upstream's defaults for every speed setting the fork changes: no fork env except
`STRATA_GDN_TC=1` (upstream's server value, which leaves sm_75 on the old scan), no `--pool-workers` (the engine took
18), `--prefill auto` (8K chunks).

| | 64K read | 172K read | decode at 8K | ms per verify window |
|---|---:|---:|---:|---:|
| U: stock upstream | 45.55 / 45.84 s (1,441 tok/s) | 127.52 / 128.23 s (1,349 tok/s) | 75.0 / 74.3 tok/s | 37.7 / 38.1 |
| L: this fork + the live config | 30.26 / 30.33 s (2,173 tok/s) | 70.47 / 70.41 s (2,449 tok/s) | 84.1 / 85.0 tok/s | 32.1 / 32.9 |
| **L faster than U** | **+51 %** (34 % less time) | **+82 %** (45 % less time) | **+13 %** | 14 % less |

### The 0.2.11 sync (`quick.sh A B E E B A`, 64K read and decode only, `mkcfg-0211.py`, 8K chunks)

| | 64K read | decode at 8K | ms per verify window |
|---|---:|---:|---:|
| A: the 0.2.10 build (live before) | 34.00 / 34.18 s | 85.1 / 84.6 tok/s | 31.6 / 31.9 |
| B: 0.2.11 as shipped (PCIe share refined to 0.25-0.30) | 34.59 / 34.49 s (+1.3 %) | 83.5 / 83.2 tok/s (-1.9 %) | 32.1 / 32.2 |
| E: 0.2.11 + `STRATA_PCIE_AUTO_SPLIT=0` (the probe's 0.17) | 34.58 / 34.49 s (+1.3 %) | 85.0 / 83.9 tok/s (-0.6 %, noise) | 31.6 / 32.0 |

The reads' +1.3 % comes with 0.2.11 under either rule (likely the later stage's 256 MiB reserve: CUDA1 6,389 -> 6,267
expert slots; not split out).

### Prompt chunk size (`chunk.sh A F G G F A`, then `F T H H T F`; `mkcfg-chunk.py`)

| `--prefill` (the engine's pick, the staging ring) | 64K read | 172K read | decode at 8K |
|---|---:|---:|---:|
| `auto` (8,192, 384 slots; live before) | 34.58 / 34.60 s | 87.85 / 88.00 s | 85.2 / 85.2 tok/s |
| **`auto:16384`** (16,384, 384 slots; live now), first round | 30.11 / 30.31 s (**-12.7 %**) | 70.23 / 70.57 s (**-19.9 %**) | 84.8 / 84.2 tok/s |
| `auto:16384`, second round | 30.20 / 30.18 s | 70.21 / 70.30 s | 84.7 / 84.0 tok/s |
| `auto:20480` (16,384, 8 slots) | 34.91 / 34.92 s | 80.06 / 80.54 s | 84.9 / 85.1 tok/s |
| `auto:32768` (26,624, 8 slots) | 38.88 / 38.91 s | 76.18 / 76.17 s | 82.8 / 81.9 tok/s |

Each arm with `STRATA_PREFILL_CPU_SHARE_MAX` equal to its ceiling. Above 16K the cards' VRAM leaves the prompt path a
ring of 8 slots, and the reads slow down again.

### Upstream opt-ins the Speedy fork runs (`chunk.sh L P Q E E Q P L`, `mkcfg-knobs.py`; live since 2026-11-05 20:15)

[DevDock-AI/Speedy](https://github.com/DevDock-AI/Speedy) (a Strata fork on 0.2.7.1, tuned on an RTX 3080) turns on
upstream opt-ins that are documented as the same bits. P = the live config + `STRATA_PF_GROUP_SCRATCH=1`
`STRATA_PF_HC_REGION=1` (smaller prompt buffers), `STRATA_KV_PREFETCH=1` (the next layer's KV upload beside the
compute), `STRATA_PREFILL_ROUTED_STREAM=1` (no copy of an expert no token of the chunk routes to),
`STRATA_MTP_ROOM_MIN=16384` (the draft layer's 675 MiB of experts lent to the prompt ring on reads of 16K+); Q = P at
`auto:32768`; E = the live config + `STRATA_EXPDEC=1 STRATA_EXPDEC_DOWN=1` (the decode experts with the window's
activations in shared memory, measured upstream on an RTX 2080 Ti).

| | 64K read | 172K read | decode at 8K | ms per verify window |
|---|---:|---:|---:|---:|
| L: live before | 29.91 / 30.17 s | 69.89 / 70.30 s | 84.3 / 85.4 tok/s | 32.0 / 32.1 |
| **P: + the prompt switches** (live now) | 29.08 / 29.33 s (**-2.8 %**) | 67.23 / 67.47 s (**-3.9 %**) | 84.7 / 85.2 tok/s | 32.9 / 32.9 |
| Q: P with 32K chunks (32,768, now with a 384-slot ring) | 34.51 / 34.67 s (+15 %) | 71.74 / 71.66 s (+2.3 %) | 83.8 / 84.3 tok/s | 32.6 / 32.4 |
| E: + EXPDEC | 30.20 / 30.22 s | 70.74 / 70.68 s | 82.2 / 82.9 tok/s (-2.8 %) | 32.1 / 31.7 |

Exactness (`gate2.sh`; every arm with `STRATA_PREFILL_CPU_SHARE=0`, `--adapt-swaps 0`, `--pcie-frac 0.17`, 16K chunks:
the live CPU share follows run-time timings, so two runs of one config can round apart with it; a first gate with it
on failed for that reason): three greedy 200-token answers after 1,847-, 19,830- and 69,820-token prompts.

| | 2K | 20K | 70K |
|---|---|---|---|
| base against base (the control) | same | same | same |
| P against base | same | same | same |
| E against base | same | same | same |

Not taken from Speedy: `STRATA_VERIFY_DEFER_CPU` (not in upstream, changes the answers), `STRATA_PROMPT_ATTN_IMMA` (no
sm_75 kernel), `STRATA_EXCHANGE_ROTATE` (changes the bits), `STRATA_PREFILL_CPU_SHARE=0` (the share is faster here),
the two-pass QSA select (#4119: the TF32 scorer, sm_80+), the hyper-connection read's two rows a warp (measured here
on 2026-11-03: slower).

Tried and dropped: `warp_sum8` (Speedy's 5-8 token sum, already in upstream's v3 read) in the staged down kernel that
sm_75 runs, for 2-8 tokens (9 shuffles instead of 5 a token, the same additions). Bitwise (`fused_gr_bench 300 1 6 8`,
`STRATA_HC_SPLIT=2 STRATA_SM75_PERF=1`, GPU 0, 3 alternated rounds: all 36 output hashes equal), but no faster: the
down stamp 30.8 -> 30.8 us at 5 tokens, 16.0 -> 16.1 at 1, 40.8 -> 40.7 at 6 (the kernel waits on its weight reads,
not on the shuffles). Engine `chunk.sh L N N L`: decode 83.7 against 83.2 tok/s, reads the same (noise).

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
rule) and the server's cache card. The fourth: `Q81Il` (`PLANES`) and `il_pm_at` (re-run `mmvq_il_parity` after a
sync that touches `native_mmvq.cu`). The fifth: the `STRATA_W4A8_SM75` blocks of `moe_iq4_w4a8.cu` (after a sync that
touches that file, `CUDA_VISIBLE_DEVICES=0 prefill_iq4_w4a8_test --full` must say "all identical" on sm_75, not
"skipped"; drop the patch if upstream takes the PR). The third patch: `gdn_tc.cuh`, `gdn_rec_tc` and `GDN_STACK_CUDA` (keep `"2"` when
upstream changes that line). After every sync, on GPU 0 with Strata stopped (the x1 card runs other work now):
`CUDA_VISIBLE_DEVICES=0 STRATA_GDN_TC=2 gdn_harness
--T 5000 --chunk 2500 --cuts 37 --paths auto,pipe --ref-heads 0,17,47` must show `tc->tc` and PASS on the `auto` line
(`tc->pipe` means the sm_75 scan is off), and `python -m pytest serve/test_parallel.py -k GdnStack` must pass. Skip the
speed test when nothing on this box's path changed (DeepSeek, AVX2/AVX-512 kernels, one-GPU features); otherwise
`quick.sh` against the deployed build.
