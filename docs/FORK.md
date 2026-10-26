# This fork: Strata on 2x CMP 50HX (sm_75) + 2x Xeon E5-2670 v2 (AVX1)

Upstream is [Niko1221/Strata](https://github.com/Niko1221/Strata); this fork is
[xrip/Strata](https://github.com/xrip/Strata), a direct GitHub fork of it:

- `main`: upstream `main` (0.2.0-beta, `f57c0adc`) plus a short list of patches for this box;
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
| `11a4f15`, `18ec242`, `1eb1010` | Turing prompt scorer and prompt attention: upstream since 0.1.45 (#1825, #1821), in its Turing series | `STRATA_SM75_PERF=1` |
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
- `--conversation-disk-mib N` (default 32768): the oldest conversation goes first when a write needs room (the
  time limit `--conversation-disk-ttl-min` was removed on 2026-10-14: the budget does the same job);
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
- **No useless disk work (2026-10-20).** An older copy of the same conversation, rewritten turns back (at most a tenth
  of its tokens not in the outgoing conversation's), is dropped when parking instead of evicted to the disk: on the
  HDD such writes held the next park 30, 32 and 41 s. A file is read only when that is faster than reading the
  tokens it adds, by this run's measured disk and prompt speeds: a new conversation that shares only the 16K-token
  system prompt with a saved one no longer reads its whole 6 GB file.
- **Smaller files:** only the live state and the deepest checkpoint are written (the session files' rule: the
  earlier checkpoints serve edits further back and cost ~118 MB each here).
- mmap would not help: the restore validates the whole image before the first device write (fail-closed), so every
  byte is read first anyway, and a host-to-device copy from mapped file pages is slower than from ordinary RAM.

Diagnostics, all on stderr as `strata serve: conversation disk: ...` (with the upstream `conversation cache:` lines
for parking and the GPU restore):

```
on: DIR, budget 32768 MiB, removed at the end; N stale files removed
queued 155000 tokens (4012 MiB in RAM until written; queue 1, 4012 MiB)
wrote 155000 tokens: 3480 MiB in 2 files, 1900 ms (1920 MB/s); on disk 1, 3480 MiB; 0 removed for the budget
RAM hit: 26000 tokens back from the write queue (700 MiB, no disk) in 0.1 ms
disk hit: 155000 tokens, 3480 MiB in 2 files, 1700 ms (2146 MB/s)[, after N ms waiting for its write]
waited 850 ms for 1 queued write (their RAM was needed)
totals: queued, written (MiB, MB/s), RAM hits, disk hits (MiB, MB/s), prompt tokens given back, waits for RAM (ms),
        removed for the budget, failed writes/reads; on disk (MiB), queue
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

`--conversation-disk-persist on` (PR #1937: `--conversation-disk-persist` without a value): the conversations outlive
the engine. The files are bound to the
session files' identity (model, engine version, settings); an index per conversation lets the next start take them in;
no time limit (the budget alone); on QUIT the current and every parked conversation go to the disk first (the server
waits up to 300 s; the engine ignores SIGINT, SIGTERM and SIGHUP and ends only through its stdin, so neither Ctrl+C nor
a poweroff's SIGTERM to every process cuts the save short - until 2026-10-18 a poweroff ended the engine first and lost
the run's conversations). On the box (2026-10-11): the
save at the stop 6.8 s (a 165K-token current conversation and two parked 42K ones), the next start took in 3
conversations (4,311 MiB), and all three came back from the disk with the same answers - the 165K one in 6.7 s instead
of 77.6 s.

`--conversation-disk-persist exit-only`: the same files for the next start, but nothing is written while the engine runs (an
eviction drops the conversation; QUIT writes the current and the parked ones). On the box on 2026-10-14 the evictions'
writes were never read back within a run: 40 writes (197 GiB) and 34 waits for a write (4.5 min) in five runs, and the
only two disk hits were the first requests after a start, from the QUIT save.

### Checkpoints spread back from the end (`--prompt-cache-spread`, opt-in)

The 6 checkpoints rotate by least recent use and every turn adds one at its end, so after a few short turns they all
sit in the last ~1-2K tokens. The pi coding agent (with pi-blackhole) rewrites its history a few thousand tokens back
now and then; the engine then resumed from the system prompt. In the live log (2026-10-11), with the new "read again"
line: a 210,528-token prompt the same as a held conversation for 204,349 tokens reused 9,525 and read 201,003 again
(100 s); another, 42K back, read all 205,253 (101 s). With the flag the kept points thin out geometrically (after 200
turns of 1,000 tokens: 0, 1K, 3K, 8K, 43K, 200K before the end; an edit 6K back reads 8K again, 42K back 43K). Same
number of checkpoints, no more RAM.

On the box (an 82,593-token conversation grown in 2-turn steps, a turn ~6 steps back rewritten): resumed from 73,610
tokens in 10.2 s instead of 40.0 s cold, and with the expert placement fixed (`--adapt-swaps 0 --pcie-frac 0.17`) the
same greedy answer as the cold read, word for word. Without fixing it two cold reads differ too (the placement moves
with timing): that run-to-run variation is the engine's, not the cache's.

### Turing decode: the interleaved matvec table and expert mode 8 (2026-10-12)

Upstream measures per-architecture tables and kernel modes on the cards it has; Turing had none. On a CMP 50HX:
`mmvq_il_parity --bench --emit-table` (parity OK for every case, bitwise) gave `kIlRows_75` / `kIl1Rows_75` (IQ4_XS
2560x10240 at 4 columns 94.6 -> 35.6 us, the Q5_K head at 3 columns 1781 -> 1097 us), opt-in `STRATA_SM75_TABLE=1`
(0.1.45: replaced by upstream's own sm_75 table, measured on an RTX 2080 Ti, under `STRATA_SM75_PERF=1`).
`native_expert_bench` over all 48 layers, mode 0 against `STRATA_EXP_MODE=8` (Volta's and the P100's layout, already
in the code): bitwise equal everywhere; gate/up IQ3_S / IQ2_S / IQ2_XXS / IQ3_XXS -27..-42 % at 4 tokens, down Q2_0
-9..-12 %, but IQ2_XS gate/up +19 % at 1 token and IQ4_NL down +31 % at 4 tokens (a per-type rule would keep those on
mode 0: about -26 % instead of -22 % of the expert time at 4 tokens).

Decode A/B (one binary, the arms differ only by env; `--adapt-swaps 0 --pcie-frac 0.17`, CPU prompt share off; a
fixed 35.7K-token prompt, 6 questions x 512 greedy tokens, two runs per arm):

| arm | decode | |
|---|---:|---:|
| base | 72.62 / 72.57 tok/s | |
| `STRATA_EXP_MODE=8` | 74.74 / 74.67 | +2.9 % |
| `STRATA_SM75_TABLE=1` | 77.85 / 77.46 | +7.0 % |
| both | 80.04 / 80.04 | **+10.3 %** |

A per-type rule on top (mode 8 keeping the CUDA layout for IQ4_NL down on sm_75, branch `sm75-exp-types`): the bench's
IQ4_NL down at 4 tokens 129 -> 103 us and the whole expert call over 48 layers -17.5 % instead of -14.2 %, bitwise
equal; in decode nothing (old 80.52 / 80.20 / 80.41, new 80.36 / 80.09 / 80.23 tok/s), so not merged. Its IQ2_XS
gate/up half did not even move the bench (+17.6 % at 1 token with the same kernel as mode 0: the cost is elsewhere in
mode 8's call).

Two runs of the same arm already differ in 1-2 of 6 answers late in the text (the engine is not fully repeatable
here even with these settings), so the texts cannot judge bit-exactness; the parity tests do. Both are on in the box's
config.

### From the RTX 2080 Ti fork (imanu86/Strata-2080Ti), 2026-10-12

- Upstream #2072 (rafatxf, open; cherry-picked): the decode QSA selection at long context on sm_75 - a halving scorer
  and a split top-k, both on by default on Turing, bitwise (`qsa_topk_parity --selftest` PASS, `qsa_parity` 0 failures,
  `qsa_select_bench` score hashes equal with `STRATA_SCORES_HALVE` 0/1 at 131K-262K on a CMP 50HX). The scorer at 172K,
  4 queries 141.1 -> 57.8 us. Decode A/B at 172K (deterministic, live env): 72.38 / 72.38 -> 74.64 / 74.29 tok/s
  (**+2.9 %** over `STRATA_QSA_ROWS`, ~+4 % over neither). It replaced `STRATA_QSA_ROWS` (the 2080 Ti fork's
  `pr/qsa-rows-scores`, +1.5 %, cherry-picked and reverted: #2072's scorer is faster and covers it).
- Upstream #2091 + #2093 (rafatxf, open; cherry-picked), on by default with `STRATA_BF16_TC=1` on sm_75: the
  hyper-connection's xn16 written as FP16 for its tensor-core GEMMs (`STRATA_XN_F16`) and the prompt path's top-k with one
  read of the scores (`STRATA_TOPK_SAMPLE`). Bitwise: `qsa_topk_parity --selftest` PASS; the first logits after a 172K
  prompt byte-identical on vs off (and between two "off" runs, CPU share 0 - the control). Prompt read, the live env:
  172K cold 80.06 / 79.92 -> 78.02 / 77.90 s, appends of 4.8 / 9.3 / 14.5K at 172K depth -3.1 / -3.0 / -3.1 % (**~3 %**).
- `STRATA_SELECT_SIMT=1` (upstream opt-in, #1091; on by default in that fork): the 172K prompt read 78.6 / 79.3 s ->
  78.5 / 78.6 s, and appends of 4.8-14.5K tokens at 172K depth -0.8 % (noise): nothing here - our own Turing
  thread scorer (`11a4f15a`) already speeds up that part - and not bitwise. Not used.
- Not taken: its sm_75 MMVQ table (upstream #1437 is the same idea as our table), EXPDEC (the same idea as mode 8).

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
- 0.1.43 (`7695859a`, 72 upstream commits, 2026-10-11): one conflict, upstream's `ConversationCache::clear()` (#879)
  beside our `set_on_evict` - both kept. #879 drops every parked conversation after non-finite logits; the disk tier
  holds conversations that were parked, so `ConversationDiskTier::clear()` now drops them too (also in PR #1937). Not
  yet A/B'd on the box.
- 0.1.43.1 (`a0116b27`, 33 upstream commits, 2026-10-12, merged with GitHub's sync): no conflicts. For this box: parking
  no longer trips the watchdog (#2049: a 352K-token park took 15.5 s here), tool calls in a Markdown fence and a quoted
  `</tool_call>` (#2037 #2026), pinned bounce slots for the prompt path's pageable copies (#1996). The version change makes
  the disk tier refuse the saved conversations once. Builds; conv_cache_test, conversation_disk_test 77,
  conversation_cache_test 4219, serve 309 pass.
- 0.1.44 (`ed4773eb`, 271 upstream commits, 2026-10-13): Mac Metal, NVFP4 W4A4, Qwen3.6, sm_61 tables, the #2228 picture
  hash in the conversation cache. Upstream rewrote the prompt path's expert group gather for NVFP4, so our in-place
  gate/up MMQ (`94292d66`, 0.7 % of a 140K turn, a patched copy of llama.cpp's `mmq.cuh`) was reverted before the
  merge. Conflicts kept both sides: the sm_75 and sm_61 one-column tables, the disk-persist QUIT timeout, our read-again
  diagnostic after #2228's. The new multi-GPU defaults (drafter on the first card, pipelined decode from two stages,
  16K prompt chunks) are HIP/gfx12 only: CUDA is unchanged. The version change makes the disk tier refuse the saved
  conversations once. Compile check of the conflicted files, conversation_disk_test 77, serve 764 pass.
- 0.1.44.1 (`296b59d2`, 11 more, 2026-10-14, GitHub's sync `ddcf3260`): `0b6fc1ce` is needed here - 0.1.44's NVFP4 CPU
  files had AVX2/AVX-512 static initialisers, an illegal instruction before `main` on these AVX1 Xeons (#2324). Box A/B
  against the deployed `4b850db8` (0.1.43.1 + ours), gate config, two runs each:

  | | A `4b850db8` | B `ddcf3260` |
  |---|---:|---:|
  | first logits after a 172K prompt | - | differ: KL 0.0021, same top 5 (upstream's own change: the new reference) |
  | decode, 6x512 tokens at 172K | 74.9 / 74.5 tok/s | 74.0 / 74.5 tok/s |
  | 172K cold read (CPU share on) | 78.5 / 78.2 s | 75.5 / 75.5 s (-3.6 %) |
  | first +4.8K append after it | 6.25-6.42 s | 6.51-6.60 s (+4 %) |
  | next appends (strict extension) | 5.92 / 6.59 s | 6.03 / 6.55 s |

  With the CPU share off A and B append alike (6.42 / 6.44 s): the gap is the share's - STRATA_PREFILL_TIMING shows
  B's GPU waiting longer for streamed experts ("wait copy" 311 -> 757 ms) on small chunks, while its expert GEMMs got
  faster. Only the first append after a cold read pays it, so B was deployed.
- 0.1.45 (`477ac13c`, 264 upstream commits, 2026-10-17, merge `0040d8f1`): upstream took the Turing kernels this fork
  had (#1821 #1825 #2069-#2072 #2081 #2091 #2093 #2220-#2222) as one opt-in, `STRATA_SM75_PERF=1`; their versions were
  kept in the conflicts, and upstream's sm_75 table replaces ours (`STRATA_SM75_TABLE` is gone). Our `STRATA_EXP_MODE=8`
  still wins over the series' mode 4. The disk tier's eviction hook sits on upstream's new `evict_oldest`, and a disk
  read-back never borrows (#1164). Box A/B against the deployed `44f7bb6e`, gate config, two runs each:

  | | A `44f7bb6e` | B `0040d8f1` + `STRATA_SM75_PERF=1` |
  |---|---:|---:|
  | first logits after a 172K prompt | - | differ: KL 0.0057, same top 5 (upstream: the new reference) |
  | decode, 6x512 tokens at 172K | 74.54 / 74.63 tok/s | 74.09 / 74.13 tok/s (fewer drafts accepted: 1,882 vs 1,930) |
  | decode, ms per verify window | 36.0 | 34.8 (-3 %) |
  | 172K cold read (CPU share on) | 75.16 / 75.18 s | 74.28 / 74.23 s (-1.2 %) |
  | appends of 4.8 / 9.3 / 14.5K at 172K depth | 6.52 / 10.79 / 12.58 s | 6.47 / 10.53 / 12.47 s (-0.8 / -2.4 / -0.8 %) |

  `STRATA_SPLIT_OVERLAP=1` (the overlapped layer-split hand-off, +32 % decode on two RX 7900 XTX): 74.12 / 74.07 tok/s,
  35.0 ms per window - nothing here, not used.
- 0.2.0-beta (`f57c0adc`, 786 upstream commits, 96 of them in the engine, 2026-10-19, merge `098f475e`): mostly the
  desktop app, Gyro's trellis packs, Volta / Pascal / sm_120 / gfx12 / Arc kernels and the Linux file tier (a GGUF read
  in place only). Conflicts kept both sides: `nv_defaults.cpp` beside `conversation_disk.cpp`, the adaptive tier's wait
  line beside our disk note, `end_leftover_engines` beside the persist QUIT wait (`ENGINE_QUIT_WAIT_S` when off). The
  version change (0.1.45 -> 0.1.45.3) makes the disk tier refuse the saved conversations once. For this box: upstream's
  new CPU pool default stays on the main thread's socket - 9 workers here instead of 19 - and decode at 172K fell 12.6 %;
  `--pool-workers 19` gives the old count back (the first logits do not depend on the count). Box A/B against the
  deployed `7a922860`, gate config (reads with the live CPU share), interleaved, after a reboot:

  | | A `7a922860` | B `098f475e` (9 workers) | B + `--pool-workers 19` |
  |---|---:|---:|---:|
  | first logits (CPU share 0) | - | differ: KL 0.0057 after 172K, 0.0019 after 245 tokens, same top 5 | as B |
  | 42K cold read | 21.87 / 21.89 s | 21.95 / 22.11 s | 21.86 / 22.08 s |
  | 172K cold read | 74.41 / 74.34 s | 74.23 / 74.08 s | 74.10 / 74.12 s |
  | decode, 6x512 tokens at 172K | 74.96 tok/s | 65.51 tok/s (-12.6 %) | 75.08 tok/s |

  The logits differ with our switches off too (`STRATA_SM75_PERF=0`, `STRATA_EXP_MODE=0`, `STRATA_BF16_TC=0`): upstream's
  own change, the new reference. At 42K, `STRATA_PREFILL_TIMING` (2 runs each): GPU timeline +0.4 %, the expert GEMMs,
  attention, GDN and dequant within 0.3 %; the rest is upstream's new accounting (`hc write` on its own line, the first
  chunk's PLE wait on the `ple` line).

  Upstream's opt-in exact decode switches on B + 19 workers, one run each, ms per verify window against the base's 34.91:
  `STRATA_ATTN_DAILY=1` -1.4 %, `STRATA_SIDE_MIXER=2` -1.3 %, `STRATA_B6_MMVQ_ROWS=1` -0.7 %, `STRATA_ARGMAX_ROWS_SM70=1`
  -0.3 %, `STRATA_GR_NORM_FAST=1` -0.1 %, `STRATA_QSA_ROWS=1` +0.5 %, `STRATA_GU_SWIGLU=1` +1.2 % (`STRATA_FUSE_DECODE` is
  in the experimental Pascal / Volta build only, `STRATA_CPU_IQ_ALL_EXACT1` needs AVX2). Exactness on the CMP 50HX:
  `qsa_merge_daily_parity` 1,172 cases bit-identical on both cards, `b6_mmvq_bench` every output bit-identical, the side
  mixer runs the same kernels on branches of the window's graph. The three together, 3 interleaved rounds:

  | | B + 19 workers | + `SIDE_MIXER=2` `ATTN_DAILY=1` `B6_MMVQ_ROWS=1` |
  |---|---:|---:|
  | decode, 6x512 tokens at 172K | 75.13 / 74.87 / 75.11 tok/s | 77.03 / 77.15 / 77.17 tok/s (+2.8 %) |
  | ms per verify window | 34.86 / 34.71 / 34.99 | 34.17 / 34.27 / 34.29 (-1.7 %) |
  | 172K cold read (CPU share 0) | 73.33 / 74.07 / 73.60 s | 73.83 / 73.47 / 73.64 s |
- 0.2.1 (`90767252`, 98 upstream commits, 2026-10-20): app, AMD, integrated GPUs, Windows and Docker. The engine
  changes (18 files) are for AMD, integrated GPUs, hybrid Intel CPUs, the file tier and `--batch`, none on this box's
  path, so there was no speed A/B. The smaller VRAM reserve on a card that drives no display (512 MiB, 700 with pictures):
  our engine already ran with 700 MiB (`--vision`), and CUDA1 has only ~386 MiB free after loading. I/O prefetch and
  the drive probe are for models read in place; ours sit in the pinned arena. The two-card batch default needs
  `--batch`. One conflict: upstream split `serve/test_parallel.py`, so our usage assertion (PR #2290) moved to
  `serve/test_parallel_yield.py`. The engine version is 0.1.46: the conversations saved on the disk are refused once,
  at the first start of the new engine.
- 0.2.1.3 (`5b47f889`, 49 upstream commits, 2026-10-20): Terminal, setup, Control and the server. Upstream merged our
  #2290 (Anthropic `cache_creation_input_tokens`) and #2140 (`thinking_budget_tokens`); the one conflict takes upstream's
  form, so the fork no longer differs there. Engine: only the version (0.1.46.2) on this box; the other changes are HIP
  / Windows only, so no speed A/B. The version change makes the disk tier refuse the saved conversations once.
- 0.2.2 (`011c9e60`, 242 upstream commits, 2026-10-22): engine 0.1.47. For this box the one engine change is the CPU
  pool: the NUMA split (#3201) now turns on by itself with 2 NUMA nodes. Here it makes 2 slices of 10 threads and puts
  `--pool-workers 19` on the sockets in turn (1,10,2,11,...); the experts come from the arena, so their memory is not
  placed per node (the engine says so). The rest is Gyro / trellis, Metal, SYCL, AMD, setup and docs. One conflict:
  upstream's `engine_nice` beside our persist QUIT wait in `server.py`. Box A/B (`~/opt-avx1/ab_022.sh`, gate config:
  swaps off, `--pcie-frac 0.17`), A = the deployed 0.1.46 (`49b4bafb`), N = this merge, N0 = N + `STRATA_NUMA_POOL=0`:

  | | A | N (split on) | N0 (split off) |
  |---|---:|---:|---:|
  | first logits (CPU share 0, warm-up and 172K) | - | bit-identical to A | bit-identical to A |
  | 42K cold read, 2 rounds | 21.91 / 21.98 s | 21.97 / 22.05 s | 21.93 / 22.04 s |
  | 172K cold read, 2 rounds | 74.54 / 74.25 s | 74.16 / 74.12 s | 74.47 / 74.18 s |
  | decode, 6x512 tokens at 172K, 2 rounds | 75.64 / 76.85 tok/s | 76.58 / 77.75 tok/s | 76.90 / 76.94 tok/s |
  | experts loaded at start | 43-60 s | 43-61 s | 44-52 s |

  No change outside the noise (A's own rounds differ by 1.6 % in decode); the split stays at upstream's default. The
  decode texts are not a check here: the same binary gave different answers 3 and 4 in its two rounds (late, after
  ~1,000 characters). Upstream's `pool_tasks_test` and `pool_cores_test` fail on this box: they expect a one-node
  machine (the real pool here has the split on); not an engine fault.
- 0.2.3 (`5c8dd322`, 127 upstream commits, 2026-10-23): engine 0.1.48. No conflicts. For this box: the hyper-connection
  read (`gr_read`, decode and the MTP layer) starts 4 kernels instead of 6, bitwise the same. The DFlash2 drafter
  (`--draft dflash2`) is for one card and the CLI in this build (not `--serve` or a layer split), so not for this box.
  `STRATA_SPEC_COST=1` (opt-in): the MTP window by expected tokens per ms at the measured row costs. The rest is Strix
  Halo, Arc, Metal, the app and the Terminal. Box A/B (`~/opt-avx1/ab_023.sh`, gate config: swaps off,
  `--pcie-frac 0.17`, `numactl --interleave=all` as the live start), A = the deployed 0.2.2 (`aeb84ae8`), N = this
  merge, NC = N + `STRATA_SPEC_COST=1`:

  | | A | N | NC |
  |---|---:|---:|---:|
  | 42K cold read, 2 rounds | 22.00 / 22.01 s | 22.12 / 22.12 s (+0.5 %) | - |
  | 172K cold read, 2 rounds | 74.15 / 74.38 s | 74.30 / 74.61 s (+0.3 %) | - |
  | decode, 6x512 tokens at 172K, 2 rounds | 77.75 / 77.77 tok/s | 76.63 / 76.98 tok/s | 77.19 / 77.86 tok/s |
  | the same, answers 1-5 (answer 6 is another text in N and NC) | 76.68 / 76.74 tok/s | 76.67 / 77.09 tok/s | 77.10 / 78.19 tok/s |
  | drafts accepted / drafted | 1,895 / 2,711 | 1,880 / 2,757 | 1,747 / 2,262 |

  No regression in decode; the reads are 0.3-0.5 % slower in both rounds (the prompt path's code is the same with no
  DFlash2 taps asked for). `STRATA_SPEC_COST=1` drafts shorter windows (30 against 34 ms per window): +1 % decode, inside
  the noise, and its answers differ between its own two runs (the window follows the measured times), so it is not
  used. Upstream's `conversation_snapshot_test` fails while the live engine shares the card and passes on a free one;
  `pool_tasks_test` / `pool_cores_test` fail as in 0.2.2.
- 0.2.4.2 (`7dd5cf18`, 287 upstream commits, 2026-10-25): engine 0.1.49.2. Almost all of it is the DeepSeek-V4.1-Flash
  engine (`strata-dsv41`, a separate build, `-DSTRATA_BUILD_DSV41=ON`, off by default). For the Qwen path only CPU rows
  for AVX2 / AVX-512 (not run on these AVX1 Xeons), the trellis encoder and `STRATA_ACT_DUMP` (research, off). No
  conflicts. Box A/B (`~/opt-avx1/ab_024.sh`, gate config, no `numactl`), A = the deployed 0.2.3 (`e55109f7`), N = this
  merge:

  | | A | N |
  |---|---:|---:|
  | 42K cold read, 2 rounds | 22.16 / 22.55 s | 22.17 / 22.59 s |
  | 172K cold read, 2 rounds | 74.02 / 74.76 s | 74.29 / 74.24 s |
  | decode, 6x512 tokens at 172K, 1 round (stopped: no difference) | 71.82 tok/s | 71.85 tok/s |

  No change. Decode is lower than in the 0.2.3 A/B (77 tok/s) for both builds: since 2026-10-23 node 0 has a 32 GB and a
  16 GB stick, its top 16 GiB (physical 34-50 GiB) is on one channel, and without `numactl` this start put 28 % of the
  arena there (`~/opt-avx1/physmap.py`; the live start of 2026-10-23 had none there). `conversation_snapshot_test`
  fails by timing: after `f.fill(177)` (a `cudaMemcpy` from pageable memory, whose DMA can land after the call returns)
  the test saves and restores on the snapshot's non-blocking stream without the device synchronization the snapshot
  functions ask for. With `cudaDeviceSynchronize()` after that fill it passed 10 of 10 runs (3 of 3 failed without).
- 0.2.6 (`4b9e05a2`, 163 upstream commits, 0.2.5 and 0.2.6, 2026-10-26): engine 0.1.51. No conflicts. Again mostly the
  DeepSeek engine. For the Qwen path: the adaptive tier picks its asynchronous mode by measurement, but only on one GPU
  (it says "a layer split" and stays off here); 512-bit CPU expert rows (AVX-512 only); two small server fixes (a chat
  request with no message is a 400; no tok/s for a one-token reply). No speed A/B (nothing on this box's path changed);
  builds on the box, the CPU unit tests (24, with `adapt_async_default_test`) and serve 379 pass.

## Settings for this box

Measured on 0.1.41 (tag `cmp50hx-2026-10-10`). Against `avx1-sm75-opt`: prompt reading +8 % (+12 % with the CPU share), decode about +19 % (few samples).
Answers are not word for word the old fork's (these switches round differently); they were read and are correct.

Build: `-DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ISA_FLOOR=avx`, GCC 14, CUDA 13.3. Config
(`strata-<model>.json`), on top of what setup writes (`--layer-split auto`, `--kv int8`, `--spec 4`, `--mtp`):

```json
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_CPU_SHARE": "auto", "STRATA_PREFILL_CPU_SHARE_MAX": "8192",
        "STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_SM75_PERF": "1", "STRATA_EXP_MODE": "8",
        "STRATA_SIDE_MIXER": "2", "STRATA_ATTN_DAILY": "1", "STRATA_B6_MMVQ_ROWS": "1"}
```

and `"--pool-workers", "19"` in `"args"` (since 0.2.0-beta the default is the main thread's socket only: 9 workers, decode
-12.6 % here; see Syncs).

and in `"args"` the conversation cache, so an agent's main conversation and its sub-agents do not evict each other:

```json
"--conversation-cache-mib", "8192", "--conversation-cache-slots", "3", "--conversation-cache-min-tokens", "32768",
"--conversation-disk-dir", "/mnt/hdd12tb/conversation-cache", "--conversation-disk-persist", "on", "--prompt-cache-spread"
```

The disk tier is on a 12 TB 7200 rpm HDD since 2026-10-20 (`exit-only` before): a main conversation of ~265-340K
tokens (5.2-6.3 GB parked) and two parallel sub-agents (~2.6 GB each) do not fit the 8 GiB, and `exit-only` drops what
it evicts (each next request of the main conversation read ~265K tokens again, 122 s; 8 times in 1.5 h). Back from the
HDD: 286,199 tokens, 5,352 MiB read in 32.8 s plus 8.9 s GPU restore. Not the NVMe: ~60 GB/h of writes in such a session.

Without it the engine keeps one conversation: in an agent session every switch between the main conversation
(~155K tokens) and a sub-agent re-read the whole prompt (~77 s). Checked on the box: conversation A (41,854 tokens)
23.0 s, B 24.0 s, A again 2.9 s with 41,849 tokens reused (parked in 1.3 s, restored in 0.66 s). A parked conversation
takes ~26 KB per token (1.06 GiB at 42K tokens), so the 8 GiB budget holds a ~155K main conversation plus a long
sub-agent; conversations under 32K tokens are not parked (re-reading them takes 8-10 s). The engine skips parking
when RAM is short (`--conversation-cache-min-free-mib`, default 2560) and runs as without the cache.
