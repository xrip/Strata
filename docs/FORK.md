# This fork: Strata on 2x CMP 50HX (sm_75) + 2x Xeon E5-2670 v2 (AVX1)

Upstream is [Niko1221/Strata](https://github.com/Niko1221/Strata); this fork is
[xrip/Strata](https://github.com/xrip/Strata), a direct GitHub fork of it:

- `main`: upstream `main` (0.1.42, `61b3fb5d`) plus a short list of patches for this box;
- `pr/*`: each generic patch on top of upstream `main`, sent upstream as a PR;
- tags `cmp50hx-*`: measured snapshots (`cmp50hx-2026-10-10`: the settings below).

The numbers for each patch are in its commit message; the older notes in [AVX1_SM75_PLAN.md](AVX1_SM75_PLAN.md) can
be out of date. The earlier fork (xrip/strata-dual-cmp50hx, branch `avx1-sm75-opt`) is retired.

## Syncing with upstream

```sh
git fetch upstream
git merge upstream/main          # on main; the history is shared, so a plain merge works
```

Rules that keep the next merge small:

- A patch here is small and on its own: new files where possible, a few lines in upstream files.
- A patch that upstream takes (or replaces with its own) is dropped here.
- Each generic patch is also kept as a branch on top of upstream `main` (`pr/*`), ready to send upstream.

## Patches on top of upstream

| commit | what | switch |
|---|---|---|
| `7b07ac7` | tokenizer: a long message's ids remembered whole, on top of upstream's piece cache | always |
| `11a4f15`, `18ec242` | QSA prompt-path block scores: a thread per query head, four lanes per query (Turing) | `STRATA_SCORES_THREAD=0` = old |
| `1eb1010` | prompt attention, int8 K/V: next chunk in registers, K and V in one buffer (Turing) | none (same bits) |
| `94292d6` | gate/up MMQ reads the experts in place (a patched copy of llama.cpp's `mmq.cuh` in the build tree) | `STRATA_MMQ_INPLACE=0` = old |
| `e8f68e0`, `cc3fd60`, `3a5e69e`, `8f93be3` | `dual3090/verify/`: agentbench (gate and compare), flashbench corpus, prefill_probe; `prefill_mmq_bench` | tools |
| `3fcb526`..`f72cce9` (upstream PR #1937) | the disk tier of the parked conversations (`conversation_disk.hpp`), below | `--conversation-disk-dir DIR` (off by default) |

### The disk tier of the parked conversations

With `--conversation-cache-mib` the engine keeps a few conversations in RAM; one it evicts (slots or RAM budget
full) was dropped, and an agent that came back to it read its whole prompt again. With `--conversation-disk-dir DIR`
the evicted conversation is written to `DIR` instead, and a request that continues it reads it back (the same
validation and restore as a RAM-parked one) when the disk holds a longer prefix than the session, a `--batch` slot
or the RAM cache:

- each image of a layer split (the first stage's and every later stage's) is one session file, written and read
  with the session files' code (`conversation_file.hpp`: atomic write, size and hash checks, the free-space preflight
  `--session-min-free-mib`); their identity is a random number of this run, so only this process reads them back;
- `--conversation-disk-ttl-min N` (default 30): a conversation stays on the disk N minutes after it was written
  (removed at the next request or write after that); `--conversation-disk-mib N` (default 32768): oldest first out;
- read back = removed from the disk (the session holds it again, and it is parked again on the next switch);
- RAM for a read: the RAM preflight with the parking floor (`--conversation-cache-min-free-mib`); when RAM is short
  the RAM cache's oldest conversations go to the disk first, as when parking;
- the folder's files of this tier (`strata-conv-*`) are removed at start and when the engine ends.

Latency, by design:

- **No disk wait on a switch.** An evicted conversation only joins a write queue; the tier's own thread (CPU nice 10
  on Linux, below the serving threads) writes it (O_DIRECT where the filesystem takes it, so the expert arena's page
  cache is not pushed out). A conversation requested again before its write started comes back from RAM, with no disk
  at all (a "RAM hit"); one being written is waited for, then read.
- **RAM:** a queued image holds its RAM until its write ends. When RAM is short (parking a conversation, or reading
  one back), the engine first waits for the queued writes (their RAM comes back) and only then evicts more.
- **Smaller files:** only the live state and the deepest checkpoint are written (the session files' rule: the
  earlier checkpoints serve edits further back and cost ~118 MB each here).
- mmap would not help: the restore validates the whole image before the first device write (fail-closed), so every
  byte is read first anyway, and a host-to-device copy from mapped file pages is slower than from ordinary RAM.

Diagnostics, all on stderr as `strata serve: conversation disk: ...` (with the upstream `conversation cache:` lines
for parking and the GPU restore):

```
on: DIR, budget 32768 MiB, 30 min per conversation; N stale files removed
queued 155000 tokens (4012 MiB in RAM until written; queue 1, 4012 MiB)
wrote 155000 tokens: 3480 MiB in 2 files, 1900 ms (1920 MB/s); on disk 1, 3480 MiB; 0 removed for the budget, 0 expired
RAM hit: 26000 tokens back from the write queue (700 MiB, no disk) in 0.1 ms
disk hit: 155000 tokens, 3480 MiB in 2 files, 1700 ms (2146 MB/s)[, after N ms waiting for its write]
waited 850 ms for 1 queued write (their RAM was needed)
removed 1 conversation older than 30 min
totals: queued, written (MiB, MB/s), RAM hits, disk hits (MiB, MB/s), prompt tokens given back, waits for RAM (ms),
        expired, removed for the budget, failed writes/reads; on disk (MiB), queue
```

(the numbers above show the format, they are not measurements). The impact of a hit: the tokens given back against
the engine's prompt speed (~2,000 tok/s cold here), against the read plus the upstream `conversation cache: restored
... in N ms` (GPU restore).

The parked RAM image is ~26 KB per token here, so a 155K-token conversation is ~4 GB in RAM and somewhat less on the
disk (deepest checkpoint only). Unit test: `conversation_disk_test` (CPU only, with
`-DSTRATA_BUILD_CONVERSATION_TESTS=ON`; 55 checks; also clean under ThreadSanitizer).

On the box (2026-10-11; gate config + this box's switches, one RAM slot so that a third conversation goes to the
disk, greedy, 48 tokens): the follow-up answers of two 42K-token conversations restored from the disk (and from RAM)
were the same text as the live conversations' answers. A 165,557-token conversation: cold read 77.8 s, back from the
disk 6.4 s (read 3.6 s: 2,639 MiB at 761 MB/s; GPU restore 0.87 s; in RAM it was 3,364 MB). The background writes ran
at 80-370 MB/s while the next prompt was read; a request waited for a write once (0.9 s). No file was left after the
engine ended. Sent upstream as PR #1937.

## Dropped in the sync (from the `avx1-sm75-opt` branch)

| commit | why |
|---|---|
| `87fa125` no-AVX2 support | upstream's older-CPU build (`STRATA_ISA_FLOOR=avx`) does the same |
| `8aa74eb` Q2_0 rows on AVX-only CPUs (dropped at the 0.1.42 sync) | upstream's own `q2_avx1.cpp` (`7b14a443`) and `iq_avx1.cpp` (`dd7b0b5f`): an AVX-only CPU now runs this pack's IQ3_XXS gate/up rows and its Q2_0 / IQ4_NL down rows on 128-bit kernels (`iq128_gu_rows`, `iq128_down_rows`); our patch only had the Q2_0 rows |
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

## Syncs

- 0.1.42 (`61b3fb5d`, 235 upstream commits): conflicts only in our AVX1 Q2_0 patch, which upstream replaced (see
  above); the tokenizer cache follows upstream's new cache style (set up in `__init__`). Box A/B 2026-10-11 (agentbench
  34K -> 130K, gate config, `STRATA_BF16_TC=1` + the CPU prompt share in every arm):

  | arm | median incremental read | decode, answers of 100+ tokens |
  |---|---:|---:|
  | 0.1.41 snapshot (`cmp50hx-2026-10-10`) | 808.2 tok/s | 69.3 tok/s |
  | 0.1.42, `STRATA_ROUTE_TAIL_SKIP` default (7) | 808.7 tok/s | 70.4 tok/s |
  | 0.1.42, `STRATA_ROUTE_TAIL_SKIP=0` | 810.5 tok/s | 70.2 tok/s |

  No regression. The tail skip was on (all 22 answers differ from `=0`) but gives nothing here (97-99 % of the routed
  experts are in VRAM, and it only saves work on the missed ones), so this box runs `STRATA_ROUTE_TAIL_SKIP=0`, which
  keeps the model's exact output.

## Settings for this box

Measured on 0.1.41 (tag `cmp50hx-2026-10-10`). Against `avx1-sm75-opt`: prompt reading +8 % (+12 % with the CPU share), decode about +19 % (few samples).
Answers are not word for word the old fork's (these switches round differently); they were read and are correct.

Build: `-DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ISA_FLOOR=avx`, GCC 14, CUDA 13.3. Config
(`strata-<model>.json`), on top of what setup writes (`--layer-split auto`, `--kv int8`, `--spec 4`, `--mtp`):

```json
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_CPU_SHARE": "auto", "STRATA_PREFILL_CPU_SHARE_MAX": "8192",
        "STRATA_ROUTE_TAIL_SKIP": "0"}
```

and in `"args"` the conversation cache, so an agent's main conversation and its sub-agents do not evict each other:

```json
"--conversation-cache-mib", "8192", "--conversation-cache-slots", "3", "--conversation-cache-min-tokens", "32768"
```

Without it the engine keeps one conversation: in an agent session every switch between the main conversation
(~155K tokens) and a sub-agent re-read the whole prompt (~77 s). Checked on the box: conversation A (41,854 tokens)
23.0 s, B 24.0 s, A again 2.9 s with 41,849 tokens reused (parked in 1.3 s, restored in 0.66 s). A parked conversation
takes ~26 KB per token (1.06 GiB at 42K tokens), so the 8 GiB budget holds a ~155K main conversation plus a long
sub-agent; conversations under 32K tokens are not parked (re-reading them takes 8-10 s). The engine skips parking
when RAM is short (`--conversation-cache-min-free-mib`, default 2560) and runs as without the cache.
