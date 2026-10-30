<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

## This fork: Strata on 2x CMP 50HX

[xrip/Strata](https://github.com/xrip/Strata) is [Niko1221/Strata](https://github.com/Niko1221/Strata) with one patch
(parked conversations on a disk) and the settings measured fastest on one box: 2x Xeon E5-2670 v2 (AVX only, 2 NUMA
nodes), 80 GB DDR3-1866, 2x CMP 50HX (Turing sm_75, 20 GB) on a layer split over PCIe 2.0 x16, a third CMP 50HX
(10 GB, PCIe x1) for pictures, Qwen3.8-Flash-Next IQ3_XXS. The rest of this README is upstream's; the patch, the build
and the syncs are in [docs/FORK.md](docs/FORK.md).

**Gain over upstream** (engine 0.2.7, 2026-10-28; a 64K-token prompt read cold, then 6 x 512 tokens of decode at ~8K
context; ranges are separate starts):

| | 64K prompt | Decode | ms per verify window |
|---|---:|---:|---:|
| Upstream, setup's settings (`--gpus 0,1`) | 43.66-43.70 s | 78.7-78.9 tok/s | 34.4-34.8 |
| + this config without `STRATA_SPLIT_MTP_BATCH` | 30.48-30.56 s | 91.6-93.0 tok/s | 27.9-28.5 |
| This config (the live one) | 29.19-29.46 s | 94.8 tok/s | 28.35 |

The prompt reads 1.48-1.50x faster and decode is ~1.20x. Decode of one config moves ~2 % from start to start here
(where the expert memory lands in RAM).

The config (`strata-iq3_xxs.json`), on top of what setup writes:

```json
"gpu": [0, 1], "layer_split": "auto",
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_CPU_SHARE": "auto", "STRATA_PREFILL_CPU_SHARE_MAX": "8192",
        "STRATA_SM75_PERF": "1", "STRATA_EXP_MODE": "8",
        "STRATA_SIDE_MIXER": "2", "STRATA_ATTN_DAILY": "1", "STRATA_B6_MMVQ_ROWS": "1",
        "STRATA_SPLIT_MTP_BATCH": "1", "STRATA_PREFILL_LAYER_MAJOR": "0",
        "STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_ROUTE_PRIOR": "0"},
"args": [..., "--pool-workers", "19", "--max-context", "393216", "--kv", "int8",
         "--conversation-cache-mib", "16384", "--conversation-cache-slots", "16", "--conversation-cache-min-tokens", "32768",
         "--conversation-disk-dir", "/mnt/hdd12tb/conversation-cache", "--conversation-disk-persist", "on",
         "--prompt-cache-spread", "--prompt-cache", "8", "--vision"],
"vision": {"exe": ".../build-vision-cuda/bin/strata-vision", "gpu": true, "cuda_device": 2, ...}
```

What each setting does and what it gave here (each measured on the version of its date; the table above is all of
them together):

| Setting | What it does | Measured on this box |
|---|---|---|
| `STRATA_BF16_TC=1`, `STRATA_PREFILL_CPU_SHARE=auto`, `STRATA_PREFILL_CPU_SHARE_MAX=8192` | BF16 projections on the FP16 tensor cores (opt-in on Turing); the CPU computes the experts few prompt tokens route to while the GPUs do the rest, in chunks up to 8,192 tokens (default 1,024) | prompt 708 -> 792 tok/s, +12 % (2026-10-10) |
| `STRATA_SM75_PERF=1`, `STRATA_EXP_MODE=8` | the bitwise Turing decode kernels and the grouped expert kernel (opt-in upstream) | decode 72.6 -> 80.0 tok/s, +10.3 % (2026-10-12) |
| `STRATA_SIDE_MIXER=2`, `STRATA_ATTN_DAILY=1`, `STRATA_B6_MMVQ_ROWS=1` | upstream's exact sm_86 decode switches, on sm_75 | decode 75.0 -> 77.1 tok/s, +2.8 % (2026-10-19) |
| `--pool-workers 19` | CPU expert workers on both sockets; since 0.2.0 upstream picks 9 here (the host's socket only) | decode 65.5 -> 75.1 tok/s, +14.6 % (2026-10-19) |
| `STRATA_SPLIT_MTP_BATCH=1` | on a layer split, the draft layer's prompt K/V in one batched pass on CUDA1, not 8 rows at a time | 64K prompt -4.3 %, decode same (2026-10-28) |
| `STRATA_PREFILL_LAYER_MAJOR=0` | keeps the chunked prompt path; 0.2.7's layer-by-layer path streams the experts over PCIe 2.0 | forced on: 64K prompt 53.0 s against 30.5 s (2026-10-28) |
| `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_ROUTE_PRIOR=0` | the model's own routing, so the answers stay exact | the tail skip gave no speed here |
| `"gpu": [0, 1]`, `--layer-split auto` | the two x16 cards share the layers; the x1 card is left out (upstream's setup offers it as a helper) | a helper cache on the x1 card: decode 34-108 % slower (2026-10-18) |
| conversation cache (16 GiB, 16 slots, from 32K tokens) + `--conversation-disk-dir` (the patch) | parked agent conversations stay in RAM, then on the HDD, and survive a restart | 286,199 tokens back from the HDD in 41.7 s; a cold read ~136 s |
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
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

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

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

In the **desktop app** and the **Terminal**. What does not fit in memory is read from your SSD, so it needs a lot of free
disk space.

| Set (IST-DAS) | Download | Free disk needed |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | about 245 GB |
| 2.00 bpw | 257 GB | about 265 GB |
| 2.25 bpw | 274 GB | about 280 GB |

Runs on NVIDIA RTX 30, 40 and 50 series (Windows and Linux), AMD Radeon AI PRO R9700 (Linux), AMD Ryzen AI Max — Strix Halo
(Windows), and Apple Silicon Macs with 64 GB or more, text only ([Mac](docs/DEEPSEEK_MAC.md)). Setup downloads the files and the DeepSeek engine, then starts the model.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Measured with the 2.00 bpw set on Strata 0.2.5 (Strix Halo: 0.2.7). Output is the speed of the answer, prompt is how fast Strata reads your input.

| Hardware | Output | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4 to 6 tokens/s | about 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9 to 11 tokens/s | about 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21 to 25 tokens/s | 320 to 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 to 77 tokens/s | 400 to 450 tokens/s (8K) |
| Ryzen AI Max+ 395 (Strix Halo), 128 GB unified | 19 to 21 tokens/s (code), 11 (chat) | 87 tokens/s (4K), 125 tokens/s (8K) |

[Memory, disk and details](docs/DEEPSEEK41.md) · [Release notes](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.8-windows-x64-setup.exe` · `Strata-0.2.8-linux-x86_64.AppImage` · `Strata-0.2.8-macos-arm64.dmg`</sub>

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
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
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
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A coffee helps fund development and testing on more hardware.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

