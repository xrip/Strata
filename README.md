<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app/host/public/brand/strata-lockup-dark.svg">
    <img src="app/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

## This fork: Strata on 2x CMP 50HX

[xrip/Strata](https://github.com/xrip/Strata) is [Niko1221/Strata](https://github.com/Niko1221/Strata) with four patches
(upstream's SSD tier of the conversation cache on a layer split and across restarts, `--conversation-spill-persist on`,
which replaced the fork's own disk tier #1937; the older-copies rule, #3404; the tensor-core GDN scan on sm_75, #4876; 5-8
row decode windows on the interleaved kernels, #5137; the int8 IQ4 prompt kernel on sm_75, #5498, went with upstream's
MMQ in 0.2.16; the tool search patch is upstream's since 0.2.10) and the settings measured fastest on one box: 2x Xeon E5-2670 v2 (AVX only, 2 NUMA
nodes), 80 GB DDR3-1866, 2x CMP 50HX (Turing sm_75, 20 GB) on a layer split over PCIe 2.0 x16, a third CMP 50HX
(10 GB, PCIe x1) for pictures. The model: [Swift 1.5 Qwen3.8-Flash-Next GSQ-RCO IQ3_S](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
with an FP8 per-layer token table made from Qwen's FP8 checkpoint (`tools/ple_fp8_pack.py`) in place of the GGUF's
IQ4_NL one. The rest of this README is upstream's; the patches, the build and the syncs are in [docs/FORK.md](docs/FORK.md).

**Gain over upstream** (upstream 0.2.11, engine 0.1.57, 2026-11-05; the same model files; each start with the page cache dropped, a 64K
and a 172K prompt read cold, then 6 x 512 tokens of decode at ~8K context; ranges are two starts, interleaved U L L U):

| | 64K prompt | 172K prompt | Decode | ms per verify window |
|---|---:|---:|---:|---:|
| Stock upstream: no fork env, the engine's own pool size (18 here), `--prefill auto` | 45.55-45.84 s (1,441 tok/s) | 127.5-128.2 s (1,349 tok/s) | 74.3-75.0 tok/s | 37.7-38.1 |
| This fork + this config without the prompt switches (see the settings below) | 30.26-30.33 s (2,173 tok/s) | 70.41-70.47 s (2,449 tok/s) | 84.1-85.0 tok/s | 32.1-32.9 |
| This fork + the live config (with the prompt switches; a later run, whose arm without them read 30.04 s / 70.10 s) | 29.08-29.33 s (2,255 tok/s) | 67.23-67.47 s (2,561 tok/s) | 84.7-85.2 tok/s | 32.9 |
| **Live config faster than stock** | **+56 %** (36 % less time) | **+90 %** (47 % less time) | **+14 %** | 13 % less |

The longer the prompt, the bigger the gain: a 172K prompt that takes stock Strata 2 min 8 s is read here in 1 min 7 s.
Since 2026-11-07 the fork is on 0.2.12 with `STRATA_IQ4_W4A8=2` on top: prompt reads another ~2 % faster than 0.2.12
without it, the same answers (its row below). **Live now** (since 2026-11-22): upstream 0.2.18 (engine 0.1.64 - the
hy4flash release; it changes no engine source this build compiles, so 0.1.64 is 0.1.63 with a new version line) with
`STRATA_PREFILL_CPU_SHARE_MAX=4096`, `STRATA_PREFILL_RING=256` and `STRATA_P2P_HANDOFF=1`: 64K read 28.3-28.5 s,
172K 64.7-65.5 s, decode at 8K 5.52-5.61 s per 512 tokens - reads and decode equal to the 0.2.17 live build (every
delta inside the run's own spread), the loan unchanged at 2,847 slots ("Measured 2026-11-22" in docs/FORK.md; the
0.2.17 numbers, "Measured 2026-11-20"; 0.2.16's own, "Measured 2026-11-19"). The 172K read and stock 0.2.16 against the 0.2.11 table above were not measured. Both arms keep the same model and capacity
settings (393K context with YaRN, MTP drafts, int8 KV, the conversation cache, pictures on the third card); stock
upstream runs `STRATA_GDN_TC=1`, which leaves sm_75 on the old scan. Decode of one config moves ~2 % from start to start
here (where the expert memory lands in RAM). The table of 2026-10-28 (engine 0.2.7, the original IQ3_XXS) is in
[docs/FORK.md](docs/FORK.md).

The engine is built from `engine/` (`-DSTRATA_ISA_FLOOR=avx`, `-DCMAKE_CUDA_ARCHITECTURES=75`); the server is `terminal/serve/server.py`. The config (`strata-swift-iq3_s-fp8ple.json`):

```json
"gpu": [0, 1], "layer_split": "auto",
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_CPU_SHARE": "auto", "STRATA_PREFILL_CPU_SHARE_MAX": "4096",
        "STRATA_SM75_PERF": "1", "STRATA_EXP_MODE": "8",
        "STRATA_SIDE_MIXER": "2", "STRATA_ATTN_DAILY": "1", "STRATA_B6_MMVQ_ROWS": "1",
        "STRATA_SPLIT_MTP_BATCH": "1", "STRATA_PREFILL_LAYER_MAJOR": "0", "STRATA_GDN_TC": "2",
        "STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_ROUTE_PRIOR": "0", "STRATA_PCIE_AUTO_SPLIT": "0",
        "STRATA_P2P_HANDOFF": "1",
        "STRATA_PF_GROUP_SCRATCH": "1", "STRATA_PF_HC_REGION": "1", "STRATA_KV_PREFETCH": "1",
        "STRATA_PREFILL_ROUTED_STREAM": "1", "STRATA_MTP_ROOM_MIN": "16384", "STRATA_PREFILL_RING": "256"},
"args": ["--pack", ".../packs/swift-iq3_s", "--native", ".../Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
         "--ple-gguf", ".../ple/ple-fp8.gguf", "--expert-profile", ".../terminal/data/expert-profile.bin",
         "--pool-workers", "19", "--expert-cache", "auto", "--prefill", "auto:16384",
         "--spec", "4", "--spec-min-p", "0.5", "--mtp", ".../mtp/rt",
         "--max-context", "393216", "--rope-scaling", "yarn", "--rope-scale", "1.5", "--yarn-orig-ctx", "262144",
         "--kv", "int8", "--kv-resident", "32768",
         "--conversation-cache-mib", "10240", "--conversation-cache-slots", "16", "--conversation-cache-min-tokens", "32768",
         "--conversation-cache-min-free-mib", "1536",
         "--conversation-spill-dir", "/mnt/hdd12tb/conversation-spill", "--conversation-spill-persist", "on",
         "--conversation-cache-rewind-keep", "8192",
         "--prompt-cache-spread", "--prompt-cache", "8", "--vision"],
"vision": {"exe": ".../build-vision-cuda/bin/strata-vision", "gpu": true, "cuda_device": 2, "threads": 20, ...}
```

What each setting does and what it gave here (each measured on the version of its date; the table above is all of
them together):

| Setting | What it does | Measured on this box |
|---|---|---|
| Swift 1.5 IQ3_S (the model) | the fine-tune's IQ3_S quant (78 GiB, experts 46.8 GiB) | against its IQ3_XXS (40.0 GiB of experts): 64K read -15 %, decode -11 %, because 17 % fewer experts fit the cards (2026-11-04) |
| `--ple-gguf .../ple-fp8.gguf` | the per-layer token table in FP8 from Qwen's FP8 checkpoint, not the GGUF's IQ4_NL one | speed the same (64K read 1,927 against 1,925 tok/s, 2026-11-04); per-row error 7.60 % -> 2.64 % against the BF16 table (upstream's measurement, docs/DETAILS.md). The GGUFs' own tables are cut out on this box, so the flag is required here |
| `STRATA_BF16_TC=1`, `STRATA_PREFILL_CPU_SHARE=auto` | BF16 projections on the FP16 tensor cores (opt-in on Turing); the CPU computes the experts few prompt tokens route to while the GPUs do the rest (in chunks up to `STRATA_PREFILL_CPU_SHARE_MAX`, default 1,024) | prompt 708 -> 792 tok/s, +12 % (2026-10-10) |
| `STRATA_SM75_PERF=1`, `STRATA_EXP_MODE=8` | the bitwise Turing decode kernels and the grouped expert kernel (opt-in upstream) | decode 72.6 -> 80.0 tok/s, +10.3 % (2026-10-12) |
| `STRATA_SIDE_MIXER=2`, `STRATA_ATTN_DAILY=1`, `STRATA_B6_MMVQ_ROWS=1` | upstream's exact sm_86 decode switches, on sm_75 | decode 75.0 -> 77.1 tok/s, +2.8 % (2026-10-19) |
| `--pool-workers 19` | CPU expert workers on both sockets, one per physical core except the host thread's | against 0.2.0's 9 (the host's socket only): decode 65.5 -> 75.1 tok/s, +14.6 % (2026-10-19); 0.2.11 picks 18 by itself here, 19 against 18 not measured |
| `STRATA_SPLIT_MTP_BATCH=1` | on a layer split, the draft layer's prompt K/V in one batched pass on CUDA1, not 8 rows at a time | 64K prompt -4.3 %, decode same (2026-10-28) |
| `STRATA_PREFILL_LAYER_MAJOR=0` | keeps the chunked prompt path; 0.2.7's layer-by-layer path streams the experts over PCIe 2.0 (upstream's default since 0.2.10; kept as a pin) | forced on: 64K prompt 53.0 s against 30.5 s (2026-10-28) |
| `STRATA_GDN_TC=2` (the third patch) | the prompt's DeltaNet recurrence on the tensor cores, also on sm_75 (not bitwise, upstream's rounding) | recurrence 11.2 -> 5.1 ms a layer at 8K; prompt reads of 4-46K tokens 6 % faster than 0.2.8 (2026-11-01) |
| `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_ROUTE_PRIOR=0` | the model's own routing, so the answers stay exact (`ROUTE_PRIOR` is off by default since 0.2.8; `TAIL_SKIP` also defaults off upstream since 0.1.63) | the tail skip gave no speed here |
| `STRATA_P2P_HANDOFF=1` (since 0.2.17, upstream #6023 opt-in) | the layer split's hand-off buffers sit in the next card's VRAM, written over P2P (a PHB path here, PCIe Gen2 x16), not through pinned host memory | 64K read -0.6 %, decode -0.9 %, 172K unchanged (2026-11-20, N H H N); `STRATA_L2_PF=1` and `STRATA_PREFILL_CPU_SHARE_NT=1` of the same release measured neutral-to-worse here, `STRATA_POOL_DEP` (on by default) neutral |
| `--prefill auto:16384` | 16,384-token prompt chunks (a bare `auto` keeps 8,192 on a layer split; upstream raises it to 32,768 only on cc 8.0+ single cards) | 64K cold read 34.59 -> 30.21 s (-12.7 %), 172K 87.93 -> 70.40 s (-19.9 %), decode same within noise (2026-11-05); `auto:32768` and `auto:20480` leave an 8-slot ring here and are slower. Like any chunk size change, not bit-exact against 8K (upstream's 0.2.11 notes: near-tie block choices) |
| `STRATA_PREFILL_CPU_SHARE_MAX=4096`, `STRATA_PREFILL_RING=256` (since 0.2.16; the ring 384 -> 256 later the same day) | the no-MMQ prompt path sizes its staged-chunk buffers up to this limit and takes a 512-slot ring by itself; these keep its loan of cache slots at 0.2.14's | 64K read 29.78 -> 28.58 s, 172K 68.0 -> 65.4 s, ~8K prompt 11.1 -> 9.3 s (2026-11-19, N S S N): 0.2.16 then reads as 0.2.14.1 did, ~8K prompts 12.5 % faster; ring 256 against 384: 64K 28.31 against 28.54 s (-0.8 %), the rest the same; a share limit of 8192 made ~8K prompts 18 % slower |
| `--conversation-cache-rewind-keep 8192` (upstream opt-in) | an agent sends the history back without the model's reasoning, so each next turn resumes a little before the end of the live conversation; without it the engine first parks (copies) the whole conversation, with it the reasoning past the resume point is dropped and nothing is copied. The output does not change (a park only reads) | at 49K context the next turn's prompt part 1.1-1.4 s -> 0.28-0.34 s (2 starts each); the live log had 451 such turns within 8,192 tokens, 1,718 s of parks (median 3.6 s at 100K+) |
| `STRATA_PF_GROUP_SCRATCH=1`, `STRATA_PF_HC_REGION=1`, `STRATA_KV_PREFETCH=1`, `STRATA_PREFILL_ROUTED_STREAM=1`, `STRATA_MTP_ROOM_MIN=16384` (the prompt switches) | upstream opt-ins, all on in the [Speedy](https://github.com/DevDock-AI/Speedy) fork: prompt buffers that hold one expert group, the hyper-connection scratch in the shared region, the next layer's KV upload beside the compute, no copy of an expert no token routes to, the draft layer's experts lent to the prompt ring on reads of 16K+ | 64K read 30.04 -> 29.20 s (-2.8 %), 172K 70.10 -> 67.35 s (-3.9 %), decode the same (2026-11-05); the same greedy answers (2K / 20K / 70K prompts, CPU share fixed, swaps off) |
| `STRATA_PCIE_AUTO_SPLIT=0` | keeps the CPU/PCIe miss share of a layer split at the probe's 0.17; 0.2.11 refines it from the CPU pool (0.25-0.30 here) | 0.2.11's rule: decode 84.4 -> 83.3 tok/s, -1.3 % (2026-11-05, Swift IQ3_S) |
| `"gpu": [0, 1]`, `--layer-split auto` | the two x16 cards share the layers; the x1 card is left out (upstream's setup offers it as a helper) | a helper cache on the x1 card: decode 34-108 % slower (2026-10-18) |
| `--max-context 393216`, `--rope-scaling yarn --rope-scale 1.5 --yarn-orig-ctx 262144` | a 393,216-token context: YaRN stretches the model's trained 262,144 positions 1.5x (setup writes the same for this length) | not measured here; upstream's run (RTX 5080, IQ3_S, YaRN factor 2): at 293K tokens the loss 2.22 -> 2.04 nats a token against no scaling (docs/DETAILS.md) |
| conversation cache (10 GiB, 16 slots, from 32K tokens) + `--conversation-spill-dir` + `--conversation-spill-persist on` (the patch) | parked agent conversations stay in RAM, then on the HDD, and survive a restart | 165,558 tokens back after a restart in 8.4 s against a 66.1 s cold read, the same answers (2026-11-19); the same e2e PASSes on 0.2.17 (QUIT save 8.8 s, back in 3.0 / 3.5 / 9.8 s, 2026-11-20) and on 0.2.18 (kept 5.7 GiB, back in 2.7 / 3.2 / 8.4 s, 2026-11-22); earlier tier: 286,199 tokens from the HDD in 41.7 s, cold ~136 s |
| `--prompt-cache-spread`, `--prompt-cache 8` | after an agent edits its history, the read restarts near the edit | a 196K conversation reused 187,696 tokens, not a ~100 s re-read |
| picture reader on CUDA 2 | the x1 card reads pictures; layers on it were slower | a picture 2.8 s -> 0.18 s (2026-10-18) |

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Sponsor Strata on GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="docs/lang/README.zh-CN.md">简体中文</a> · <a href="docs/lang/README.ja.md">日本語</a> · <a href="docs/lang/README.de.md">Deutsch</a> · <a href="docs/lang/README.fr.md">Français</a> · <a href="docs/lang/README.es.md">Español</a> · <a href="docs/lang/README.pt-BR.md">Português</a> · <a href="docs/lang/README.pl.md">Polski</a></sub></p>

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.18-windows-x64-setup.exe` · `Strata-0.2.18-linux-x86_64.AppImage` · `Strata-0.2.18-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |
| **NVIDIA DGX Spark · Linux** | [DGX Spark setup](docs/DGX_SPARK_LINUX.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](.github/CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A GitHub sponsorship or a coffee helps fund development and testing on more hardware.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: sponsor Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

### Loyal supporters

Thank you to everyone who supports Strata.

<!-- supporters:start (generated from the public GitHub sponsors: one row per level, the biggest supporters first and largest; private sponsors are not listed) -->
<a href="https://github.com/codeconduit-official"><img src="https://images.weserv.nl/?url=github.com/codeconduit-official.png&w=240&h=240&fit=cover&mask=circle&output=png" width="120" height="120" alt="codeconduit-official" title="codeconduit-official"></a>

<a href="https://github.com/kod88vn"><img src="https://images.weserv.nl/?url=github.com/kod88vn.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="kod88vn" title="kod88vn"></a>
<a href="https://github.com/router0mail"><img src="https://images.weserv.nl/?url=github.com/router0mail.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="router0mail" title="router0mail"></a>
<a href="https://github.com/kingdonb"><img src="https://images.weserv.nl/?url=github.com/kingdonb.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="kingdonb" title="kingdonb"></a>
<a href="https://github.com/Daisame"><img src="https://images.weserv.nl/?url=github.com/Daisame.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="Daisame" title="Daisame"></a>

<a href="https://github.com/ryan4yin"><img src="https://images.weserv.nl/?url=github.com/ryan4yin.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="ryan4yin" title="ryan4yin"></a>
<a href="https://github.com/themactep"><img src="https://images.weserv.nl/?url=github.com/themactep.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="themactep" title="themactep"></a>
<a href="https://github.com/macintog"><img src="https://images.weserv.nl/?url=github.com/macintog.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="macintog" title="macintog"></a>
<a href="https://github.com/bisonbet"><img src="https://images.weserv.nl/?url=github.com/bisonbet.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="bisonbet" title="bisonbet"></a>
<a href="https://github.com/vizv"><img src="https://images.weserv.nl/?url=github.com/vizv.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="vizv" title="vizv"></a>
<a href="https://github.com/seoulworker"><img src="https://images.weserv.nl/?url=github.com/seoulworker.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="seoulworker" title="seoulworker"></a>
<a href="https://github.com/jaceknoga"><img src="https://images.weserv.nl/?url=github.com/jaceknoga.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="jaceknoga" title="jaceknoga"></a>
<a href="https://github.com/jackyini"><img src="https://images.weserv.nl/?url=github.com/jackyini.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="jackyini" title="jackyini"></a>
<a href="https://github.com/moruku36"><img src="https://images.weserv.nl/?url=github.com/moruku36.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="moruku36" title="moruku36"></a>
<a href="https://github.com/hrkgk"><img src="https://images.weserv.nl/?url=github.com/hrkgk.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="hrkgk" title="hrkgk"></a>
<!-- supporters:end -->

All supporters and how to support Strata: [SUPPORTERS.md](SUPPORTERS.md)

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

