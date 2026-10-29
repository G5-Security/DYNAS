# DYNAS: A Dynamic Network Penetration Testing Framework Using Deep Hierarchical Reinforcement Learning

**Md. Rakibul Hassan, Majedul Islam, Md. Ashraful Alam, Muhammad Iqbal Hossain, Swakkhar Shatabda**
School of Data and Computational Sciences, BRAC University, Dhaka, Bangladesh

DYNAS (**DY**namic **N**etwork **A**ttack **S**imulator) is a defense-aware reinforcement-learning environment for automated penetration testing, built on top of [NASimEmu](https://github.com/jaromiru/NASimEmu). This repository also contains **Pen-DHRL**, a recurrent manager–worker (feudal) policy that selects persistent subgoals from IDS-conditioned global context.

---

## Table of Contents

- [Abstract](#abstract)
- [Highlights](#highlights)
- [What DYNAS Adds to NASimEmu](#what-dynas-adds-to-nasimemu)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Framework Overview](#framework-overview)
  - [Part 1: Dynamic Scenario Generation](#part-1-dynamic-scenario-generation)
  - [Defense-Dynamics Layer](#defense-dynamics-layer)
  - [Curriculum Schedule](#curriculum-schedule)
  - [Part 2: Pen-DHRL Agent](#part-2-pen-dhrl-agent)
- [Scenarios](#scenarios)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducing the Paper's Training Regimes](#reproducing-the-papers-training-regimes)
- [Results](#results)
- [Command-Line Reference](#command-line-reference)
- [Emulation (inherited from NASimEmu)](#emulation-inherited-from-nasimemu)
- [Experimental: LLM-Teacher Extensions](#experimental-llm-teacher-extensions)
- [Tests](#tests)
- [Known Limitations](#known-limitations)
- [Citation](#citation)
- [License and Acknowledgements](#license-and-acknowledgements)

---

## Abstract

As network scale and uncertainty make automated penetration testing harder, reinforcement learning (RL) environments need to support controlled evaluation under defensive and non-stationary conditions. DYNAS combines parameterized large-scale scenario generation with:

- stateful intrusion detection (IDS) responses,
- explicit scan false-positive and false-negative noise,
- temporary service unavailability,
- duration-based action timeouts.

**Pen-DHRL** is a recurrent manager–worker policy that selects persistent subgoals using IDS-conditioned global context. We compare it with GNN, Invariant, and Attention baselines across four scenario variants and several training regimes.

## Highlights

| Training regime | Finding |
| --- | --- |
| IDS **disabled** in training, **enabled** in evaluation | Pen-DHRL gets **~64%** (Bridge) and **~24%** (Dynamic) higher reward/step, and **~26%** (VarA) and **~30%** (VarB) more captured hosts, than the next-best baseline. |
| All defenses active from the start (no curriculum) | Pen-DHRL has **positive** mean episode return in all 4 scenarios. Flat baselines are **negative** in all 12 architecture–scenario combinations. |
| Staged curriculum training | All **16** architecture–scenario combinations reach positive mean returns. |

---

## What DYNAS Adds to NASimEmu

NASimEmu provides efficient, partially observable attack planning with variable-size graph observations and scenario randomization. DYNAS keeps NASimEmu's agent interface and action vocabulary and extends its **simulation layer**:

| Feature | NASim | IAPTF | WCE | DynPen | NASimEmu | PenGym | **DYNAS** |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Simulation-based training | ✓ | ✓ | Conceptual | ✓ | ✓ | ✓ | **✓** |
| Structural scenario variation | Limited | – | Conceptual | ✓ | Limited | Limited | **✓** |
| Firewall-aware reachability | ✓ | Limited | Conceptual | – | ✓ | ✓ | **✓** |
| IDS representation / response | – | Limited | – | – | – | – | **✓** |
| Curriculum scheduling | – | – | – | – | – | – | **✓** |
| Explicit scan FP/FN noise | – | – | – | – | – | – | **✓** |
| Service-state dynamics | – | – | – | Limited | – | – | **✓** |
| Duration-based action timeouts | – | – | – | – | – | – | **✓** |

*(Table 1 of the paper. ✓ = explicit support; Limited = related support at a different granularity; Conceptual = proposed framework or case study; – = not documented in the cited protocol.)*

**Design requirements** (paper §2):

- **R1. Stateful, configurable defense.** Reaction has memory across steps; it is not a per-action penalty or a scripted opponent.
- **R2. Observable footprint.** The agent sees each host's accumulated detection level, alert threshold, and monitoring multiplier. This models an assumed-breach engagement, so stealth can be learned.
- **R3. Staged exposure.** Defenses are introduced through a curriculum, because a randomly initialized policy faced with full defenses learns to terminate early.

DYNAS addresses R1 and R3. Pen-DHRL is designed for R2.

---

## Repository Layout

```
DYNAS/
├── src/nasimemu/                  # Simulator / emulator (NASimEmu + DYNAS extensions)
│   ├── env.py                     # NASimEmuEnv: gym env, auto-generation, domain randomization, curriculum hookup
│   ├── env_utils.py               # Graph conversion, action helpers
│   ├── env_emu.py, msf_interface.py, vagrant_gen.py   # Emulation (Vagrant + Metasploit)
│   └── nasim/
│       ├── envs/
│       │   ├── curriculum.py      # CurriculumManager: stage selection by training progress
│       │   ├── host_vector.py     # Host state incl. IDS detection level / threshold / multiplier
│       │   ├── action.py          # Actions, scan noise, timeouts
│       │   ├── environment.py, state.py, network.py, observation.py
│       └── scenarios/
│           ├── loader_v2.py       # v2 YAML loader (parses intrusion_detection, curriculum, ...)
│           └── generator.py       # Scenario instance generation
│
├── NASimEmu-agents/               # RL agents and training loop
│   ├── main.py                    # Entry point: training / --eval / --trace / --debug
│   ├── config.py                  # PPO and optimizer hyperparameters
│   ├── rl.py                      # Recurrent PPO update
│   ├── net.py, graph_nns.py       # Base network + GNN building blocks
│   ├── nasim_problem/
│   │   ├── nasim_config.py        # Problem-specific CLI args (scenarios, auto-generation, DR)
│   │   ├── nasim_net_base_hrl.py  # ★ Pen-DHRL (NASimNetDHRL)
│   │   ├── nasim_net_gnn_mact.py  # GNN baseline (NASimNetGNN_MAct)
│   │   ├── nasim_net_inv_mact.py  # Invariant baseline (NASimNetInvMAct)
│   │   ├── nasim_net_xatt_mact.py # Attention baseline (NASimNetXAttMAct)
│   │   └── ...                    # Other NASimEmu architectures (MLP, LSTM variants, Feudal)
│   ├── llm_teacher/               # Experimental LLM-teacher extensions (not in the paper)
│   ├── compare_runs.py, compare_eval.py   # Run comparison utilities
│   ├── trained_models/            # Original NASimEmu pretrained baselines (mlp, gnn-mact, inv-mact, att-mact)
│   └── tests/
│
├── scenarios/                     # v2 scenario YAMLs
│   ├── corp_100hosts_dynamic*.v2.yaml   # The four DYNAS evaluation scenarios
│   ├── variants/                  # Single-stage ablations (no_ids, no_churn, all_off, ...)
│   ├── variants_sensitivity/      # Parameter sweeps (IDS threshold/decay, churn, noise, timeouts, ...)
│   └── corp.v2.yaml, uni.v2.yaml, ...   # Original NASimEmu reference scenarios
│
├── vagrant/                       # Emulation provisioning scripts
├── setup_env.sh                   # Creates the Python 3.10 venv
└── samplepaper.pdf                # The DYNAS paper
```

---

## Installation

**Requirements:** Python **3.10** (exactly) and Linux. CUDA is optional; the paper's workloads are CPU-bound and were run with `-device cpu`.

```bash
git clone https://github.com/G5-Security/DYNAS.git
cd DYNAS
./setup_env.sh                       # uses python3.10 on PATH, creates ./venv
# or: ./setup_env.sh /path/to/python3.10 venv
source venv/bin/activate
```

`setup_env.sh` handles two packaging quirks you would otherwise hit:

- `gym==0.21.0` has invalid metadata, so pip is pinned below 24.1.
- gym is built with old `setuptools<66` / `wheel<0.40`.

The script then installs `requirements.txt` (torch 2.7.1, torch-geometric 2.6.1, torch-scatter from the PyG wheel index) and installs `nasimemu` in editable mode. It finishes with an import check:

```
OK: gym 0.21.0 | torch 2.7.1 | cuda False
```

**Recommended shell settings** for training (avoids BLAS oversubscription, since training already parallelizes with worker processes):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline          # runs are logged locally under NASimEmu-agents/wandb/
```

---

## Quick Start

### Use the environment directly

```python
import gym, random
import nasimemu, nasimemu.env_utils as env_utils

env = gym.make('NASimEmu-v0', emulate=False,
               scenario_name='scenarios/corp_100hosts_dynamic.v2.yaml')
s = env.reset()

for _ in range(5):
    actions = env_utils.get_possible_actions(env, s)
    action = random.choice(actions)
    s, r, done, info = env.step(action)
    (subnet, host), a_id = action
    print(f"{env.action_list[a_id]} on ({subnet},{host}) -> r={r:.2f} done={done}")
```

`env_utils.convert_to_graph(s)` converts an observation to a graph for PyTorch Geometric.

### Smoke-train Pen-DHRL (a few minutes on CPU)

```bash
cd NASimEmu-agents
python main.py ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 16 -epoch 10 -max_epochs 10 --no_debug \
  -net_class NASimNetDHRL -use_a_t -episode_step_limit 200 \
  -observation_format graph_v2 -lr 0.0007 -alpha_h 0.02
```

---

## Framework Overview

The methodology has two connected parts (paper Fig. 1):

```
 Part 1: Dynamic scenario generation                Part 2: Pen-DHRL agent
 ───────────────────────────────────                ───────────────────────
 1a Template YAML + cache                           2a Graph observation (host/subnet features, pos. enc.)
 1b Auto-generator (subnets, topology, hosts, FW)   2b Recurrent manager GNN  → global context (+ IDS fields)
 1c Curriculum + defense dynamics                   2c Persistent subgoals from an 8-slot goal bank
 1d NASimEmu runtime (transitions, rewards)   ───►  2d Goal-conditioned GAT worker → (host, action)
                                              ◄───  2e Value + termination heads
                                                    2f Recurrent PPO replay update
```

### Part 1: Dynamic Scenario Generation

- **Template layer (1a).** A YAML template fixes the primitives: OS types, service/process probability maps, exploits, and privilege-escalation rules. It is cached, so every generated instance has an **identical action signature and observation dimension**.
- **Auto-generator (1b).** It samples subnet counts and sizes, a topology style (`mesh`, `chain`, `random`), per-host OS/services/processes, sensitive-host placement, and firewall rules derived from the connectivity style.
- **Domain randomization.** Exploit/privesc success probabilities, action costs, and scan costs can be jittered per episode. Feature dropout zeros observed service/process bits to simulate partial reconnaissance.

Relevant CLI flags (from `nasim_problem/nasim_config.py`):

| Flag | Meaning |
| --- | --- |
| `--auto_mode {off,per_episode,per_epoch}` | Regenerate a fresh scenario per episode or epoch |
| `--auto_template PATH` | Template YAML providing OS/services/exploits/costs |
| `--auto_host_range 72-96` | Total host-count range |
| `--auto_subnet_count 12` | Number of subnets (internet subnet is added automatically) |
| `--auto_topology {mesh,chain,random}` | Topology style |
| `--auto_sensitive_policy {deep,uniform}` | Sensitive-host placement |
| `--auto_sensitive_jitter F` | Per-episode jitter of sensitive-host probabilities |
| `--feature_dropout_p P` | Dropout on observed service/process bits |
| `--dr_prob_jitter`, `--dr_cost_jitter`, `--dr_scan_cost_jitter` | Domain randomization of probabilities and costs |
| `--auto_template_test`, `--auto_mode_test` | Separate generator settings for the test split |

### Defense-Dynamics Layer

All four mechanisms are configured per curriculum stage in the scenario YAML (see `curriculum.stages[*]` in [`scenarios/corp_100hosts_dynamic.v2.yaml`](scenarios/corp_100hosts_dynamic.v2.yaml)). The values below are the **final** stage, which is also used for evaluation. The **medium** stage is in parentheses.

| Mechanism | YAML key | Behaviour |
| --- | --- | --- |
| **(i) IDS** | `ids` | At reset, `(d, m) = (0, 1)` and threshold `θ ~ U(0.65, 0.88)` (U(0.75, 0.92)). Detection accumulates as `d ← λ(d + m·Δ_a)` with `λ = 0.98` (0.99), and increments grow for failed or repeated actions. Crossing `θ` triggers a **persistent** response: quarantine / patch / monitor with probabilities 0.10 / 0.35 / 0.55 (0.05 / 0.25 / 0.70). The agent **observes `d`, `m`, and `θ`**. |
| **(ii) Scan errors** | `scan_noise` | Per-scan-type FP 0.015–0.040 / FN 0.030–0.070 (0.010–0.030 / 0.020–0.050). Errors affect **observations only**, not true state. |
| **(iii) Service outages** | `service_dynamics` | Selected services fail with probability 0.01 (0.005) for 5–30 (4–20) steps and become unobservable and unavailable. |
| **(iv) Action timeouts** | `network_reliability` | Remote exploits time out with probability 0.015 (0.01), blocking that (action, subnet) pair for 1–7 (1–3) steps. Blocked actions report failure, cost `c`, and leave `d` unchanged. |

The IDS state lives in `src/nasimemu/nasim/envs/host_vector.py` (`detection_level`, `detection_threshold`, multiplier), and the stage parameters are applied through `src/nasimemu/nasim/envs/curriculum.py`.

### Curriculum Schedule

Training proceeds through three stages, with bounds given as fractions of `-max_epochs`:

| Stage | Fraction of run | Mechanisms |
| --- | --- | --- |
| `baseline` | 0 – 30% | All off |
| `medium_difficulty` | 30 – 60% | All on, low intensity |
| `full_difficulty` | 60 – 100% | All on, evaluation intensity |

```yaml
curriculum:
  enabled: true
  stages:
    - name: "baseline"
      start_frac: 0.0
      end_frac: 0.30
      start_epoch: 0      # absolute fallback when -max_epochs is unset
      end_epoch: 60
      ids: { enabled: false }
      ...
    - name: "medium_difficulty"
      start_frac: 0.30
      end_frac: 0.60
      ids:
        enabled: true
        detection_decay: 0.99
        base_thresholds: [0.75, 0.92]
        response_types: { quarantine: 0.05, patch: 0.25, monitor: 0.70 }
      ...
    - name: "full_difficulty"
      start_frac: 0.60
      end_frac: 1.0
      ...
```

The stage changes once per reporting interval (epoch). `--eval` always uses the **final** stage. A scenario with a single `full_difficulty` stage spanning `0.0–1.0` (as in `scenarios/variants/`) gives **no-curriculum** training.

### Part 2: Pen-DHRL Agent

Implemented as `NASimNetDHRL` in [`NASimEmu-agents/nasim_problem/nasim_net_base_hrl.py`](NASimEmu-agents/nasim_problem/nasim_net_base_hrl.py) (paper Fig. 2).

**Forward path for one decision:**

1. **Embedding.** Node features plus an 8-dim positional encoding pass through `Linear → LeakyReLU`.
2. **Recurrent manager.** A message-passing GNN with a global node and a GRU over the global state produces per-node features `U_t` and a global context `u_t`.
3. **IDS conditioning.** The graph-wise **mean of the observable host IDS fields** (detection level, threshold, multiplier) is projected and added to `u_t`.
4. **Persistent subgoals.** A goal bank `Q ∈ R^{8×128}` holds `K = 8` learned latent slots. A learned **switch** and a counter (horizon `H = 4`) decide whether to keep the active slot or draw a new one. The selected goal `q_t` is refined with `u_t`.
5. **Goal-conditioned worker.** A GAT layer scores (host, action) pairs conditioned on `q_t`. Subnet nodes are masked out as action targets.
6. **Value and termination heads.** These read `[u_t ; q_t ; ℓ_t]` (context, goal, normalized episode progress) and output the return estimate `V_t` and the stop decision `z_t`.

**Persistent state** across steps: active goal index `k_t`, goal vector `q_t`, remaining steps `c_t`, manager hidden state `h_t`, episode step `e_t`, and batch map. The network supports `reset`, masked reset, and `clone` for recurrent training.

**Recurrent PPO.** The update snapshots the initial hidden state and replays the recorded worker action, stop decision, and active goal index with done-masked resets. It then forms bootstrapped returns and jointly updates manager and worker with clipped policy, value, and entropy losses. A separate entropy bonus on the manager's subgoal distribution (`--alpha_h_manager`, default 0.02) counters goal collapse.

**Baselines** (from NASimEmu, sharing the same IDS-aware host features, actions, and budgets):

| Paper name | `-net_class` |
| --- | --- |
| GNN | `NASimNetGNN_MAct` |
| Invariant | `NASimNetInvMAct` |
| Attention | `NASimNetXAttMAct` |
| **Pen-DHRL** | **`NASimNetDHRL`** |

---

## Scenarios

### Reference vs. dynamic scenarios

| Scenario | File | Subnets | Hosts | Services | Exploits | Sensitive hosts | Density |
| --- | --- | :-: | :-: | :-: | :-: | :-: | :-: |
| Corp (NASimEmu) | `corp.v2.yaml` | 8 | 21 | 6 | 5 | 7 | 0.25 |
| Uni (NASimEmu) | `uni.v2.yaml` | 9 | 27 | 6 | 5 | 3 | 0.27 |
| Corp-Dyn | `corp_100hosts_dynamic.v2.yaml` | 12 | 83 | 32 | 8 | 39 | 0.19 |
| VarA | `corp_100hosts_dynamic_varA.v2.yaml` | 12 | 86 | 32 | 8 | 41 | 0.19 |

Corp-Dyn has 12 subnets of 6–8 hosts each (72–96 hosts per instance), 13 processes, and Linux/Windows service stacks.

### The four defended evaluation scenarios

All four share one action vocabulary and form a progression from interpolation to joint structural–parametric transfer:

| Paper name | File | Description |
| --- | --- | --- |
| **Dynamic** | `corp_100hosts_dynamic.v2.yaml` | Reference large-network setting |
| **Bridge** | `corp_100hosts_dynamic_bridge.v2.yaml` | Interpolates host, service, process, exploit, and cost parameters |
| **VarA** | `corp_100hosts_dynamic_varA.v2.yaml` | Lower exploit probabilities, higher costs, redistributed sensitive hosts |
| **VarB** | `corp_100hosts_dynamic_varB.v2.yaml` | Topology change plus a further parameter shift |

`corp_100hosts_dynamic_test.v2.yaml` is an additional held-out test variant.

### Ablation and sensitivity variants

- **`scenarios/variants/`** (single-stage, no curriculum): `full_dynamics`, `all_off`, `no_ids`, `no_scan_noise`, `no_churn`, `no_timeouts`, `intensity_off`, `intensity_low`, `intensity_final`, `intensity_harder`.
- **`scenarios/variants_sensitivity/`**: sweeps over IDS threshold (`ids_thr_x0.6`–`x1.5`), IDS decay (`0.90`–`0.995`), IDS increment (`x0.5`, `x2`), response type (`resp_all_{monitor,patch,quarantine}`), scan noise (`x0`–`x4`), churn (`x2`, `x3`), timeouts (`x2`, `x4`), and failed-exploit multiplier (`fem_0`, `fem_x0.5`).

Several scenarios can be combined with `:`; one is sampled uniformly at each reset. They must share `address_space_bounds`.

### Interaction-budget reference

For a scenario with `S` subnets, `H` hosts, and `N` sensitive hosts, a complete sweep needs roughly

```
A_heuristic = S + 2H + 3.5N
```

(one subnet scan per subnet, two recon actions per host, and expected exploit/privesc retries at ~0.8 success). Corp-Dyn (N=39, H=83, S=12) gives 314.5, and VarA (41, 86, 12) gives 327.5. This motivates the 400-step episode limit.

---

## Training

All commands are run from `NASimEmu-agents/`.

### Paper protocol (paper §4.3)

| Setting | Value |
| --- | --- |
| Parallel environments (`-batch`) | 128 |
| Rollout length (`ppo_t`) | 8 steps → 1,024 transitions/iteration |
| PPO passes (`ppo_k`) / clip ε | 3 / 0.2 |
| Reporting interval (`-epoch`) × intervals (`-max_epochs`) | 100 × 200 = 20,000 PPO iterations |
| Discount γ | 0.99 |
| Optimizer | AdamW, lr `7e-4`, weight decay `1e-4` |
| Initial entropy coefficient (`-alpha_h`) | 0.02 |
| Gradient-norm clip (`-max_norm`) | 3.0 |
| Target network soft update ρ | 0.1 |
| Episode step limit | 400 |
| Evaluation | 64 fresh envs, 100 episodes, final curriculum stage |

`-batch` must be divisible by `-cpus`, and `-cpus` must divide the fixed evaluation batch of 64 (use 1, 2, 4, 8, 16, 32, or 64).

### Pen-DHRL (with curriculum)

```bash
python main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 128 \
  -epoch 100 -max_epochs 200 --no_debug \
  -net_class NASimNetDHRL \
  -force_continue_epochs 0 -use_a_t \
  -episode_step_limit 400 -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  --sched_lr_rate 10000 --sched_lr_factor 0.8 --sched_lr_min 0.0003 \
  --sched_alpha_h_rate 15000 --sched_alpha_h_factor 0.5 --sched_alpha_h_min 0.005 \
  -seed 1
```

### Baselines

Swap `-net_class` and add `-mp_iterations 2 -augment_with_action`:

```bash
for NET in NASimNetGNN_MAct NASimNetInvMAct NASimNetXAttMAct; do
  python main.py ../scenarios/corp_100hosts_dynamic.v2.yaml \
    --test_scenario ../scenarios/corp_100hosts_dynamic_test.v2.yaml \
    -device cpu -cpus 8 -batch 128 \
    -epoch 100 -max_epochs 200 --no_debug \
    -net_class $NET -use_a_t \
    -episode_step_limit 400 -observation_format graph_v2 \
    -mp_iterations 2 -augment_with_action \
    -lr 0.0007 -alpha_h 0.02 -seed 1
done
```

### Outputs and checkpoints

- **Checkpoints.** `wandb/latest-run/files/model.pt` (latest) and `model_best.pt` (best by `-save_best_metric` on `-save_best_split`, default `captured_avg` on `eval_tst`). Capture the path right after a run finishes, before starting another one:
  ```bash
  CKPT="$(readlink -f wandb/latest-run)/files/model_best.pt"
  ```
- **Metrics.** Each run appends JSON lines to `training_data/runs/<run_id>.json` and `training_data/latest/latest.json`.
- **Resume.** `-load_model CKPT --resume` restores the weights, global step, schedule, curriculum position, and best value.
- **Run lock.** Only one training process may run at a time (`training_data/.training.lock`). `--eval` is not blocked.

---

## Evaluation

```bash
python main.py ../scenarios/corp_100hosts_dynamic_bridge.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic_bridge.v2.yaml \
  --eval -load_model "$CKPT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetDHRL -use_a_t \
  -episode_step_limit 400 -observation_format graph_v2
```

The evaluation must use the same `-net_class`, observation format, `-use_a_t`, and (for baselines) `-augment_with_action` / `-mp_iterations` as training. `--eval_split {both,train,test}` selects which split to run.

**Metrics** (reward `r = (v − c) / 10`, where `v` is non-zero only on first root of a sensitive host or when an IDS response fires, and `c` is the action cost):

| Metric | Key | Meaning |
| --- | --- | --- |
| Reward / step | `reward_avg` | Mean utility per interaction |
| Episode return | `reward_avg_episodes` | Cumulative utility per episode |
| Episode length | `eplen_avg` | Interaction efficiency |
| Captured hosts | `captured_avg` | Distinct sensitive hosts rooted (coverage) |

Other modes: `--trace` prints a step-by-step trace of one episode, and `--debug` opens the interactive visual debugger.

---

## Reproducing the Paper's Training Regimes

| Regime (paper) | Train on | Evaluate on |
| --- | --- | --- |
| **Defense-disabled** (Table 5) | `variants/corp_100hosts_dynamic__all_off.v2.yaml` | Corp-Dyn (train) / VarA (test) with defenses off. `--eval` uses each YAML's final stage, so the VarA test file needs a single all-off stage like the `all_off` variant. |
| **No curriculum**: all defenses from step 1 (Tables 6–7, "No") | `variants/corp_100hosts_dynamic__full_dynamics.v2.yaml` | Dynamic, Bridge, VarA, VarB with `--eval` (final stage) |
| **Curriculum** (Tables 6–7, "Curr") | `corp_100hosts_dynamic.v2.yaml` (3-stage curriculum) | Dynamic, Bridge, VarA, VarB with `--eval` |
| **IDS-disabled → IDS-enabled transfer** (Table 8) | `variants/corp_100hosts_dynamic__no_ids.v2.yaml` | Dynamic, Bridge, VarA, VarB with `--eval` (IDS on) |

---

## Results

### Defense-disabled (Corp-Dyn train / VarA test)

| Architecture | Reward/step (trn / tst) | Episode return | Episode length | Captured hosts |
| --- | --- | --- | --- | --- |
| GNN | 0.27 / 0.27 | 29.90 / 30.02 | 111.50 / 112.63 | **36.91 / 37.10** |
| Invariant | 0.34 / 0.33 | 27.21 / 27.06 | **81.10 / 80.96** | 32.23 / 32.13 |
| Attention | 0.26 / 0.26 | 28.21 / 28.09 | 109.90 / 109.42 | 35.63 / 35.46 |
| **Pen-DHRL** | **0.49 / 0.48** | **49.41 / 48.43** | 103.40 / 106.40 | 36.13 / 36.68 |

### IDS-enabled evaluation: no curriculum → curriculum (reward/step)

| Scenario | Pen-DHRL | GNN | Invariant | Attention |
| --- | --- | --- | --- | --- |
| Bridge | 0.40 → 1.51 | −0.08 → 1.37 | −1.30 → 1.22 | −1.23 → 1.24 |
| Dynamic | 0.54 → 2.25 | −0.08 → 2.17 | −1.41 → 1.89 | −1.31 → 2.01 |
| VarA | 0.17 → 0.61 | −0.10 → 0.64 | −1.29 → 0.63 | −1.20 → 0.67 |
| VarB | 0.08 → 0.38 | −0.06 → 0.40 | −1.18 → 0.40 | −0.56 → 0.19 |

Without a curriculum, every flat baseline is negative and most hit the 400-step limit while capturing ~0–2 hosts. With a curriculum, all become positive and gain ~9–28 captured hosts. Pen-DHRL is already positive without a curriculum and gains less from it (< 6 hosts). See paper Tables 6–7 for episode return, length, and capture.

### IDS disabled in training, enabled in evaluation

| Scenario | Architecture | Reward/step | Episode return | Ep. length | Captured |
| --- | --- | --- | --- | --- | --- |
| Bridge | **Pen-DHRL** | **1.15** | **161.96** | 141.20 | 23.36 |
| | GNN | 0.18 | 66.20 | 368.86 | 26.17 |
| | Invariant | 0.01 | 5.20 | 367.77 | 25.69 |
| | Attention | 0.70 | 120.04 | 170.39 | 17.58 |
| Dynamic | **Pen-DHRL** | **1.49** | 200.17 | 134.56 | 27.80 |
| | GNN | 0.26 | 91.32 | 356.31 | 30.59 |
| | Invariant | 0.19 | 66.58 | 348.37 | 30.88 |
| | Attention | 1.20 | 204.54 | 170.28 | 26.87 |
| VarA | **Pen-DHRL** | 0.14 | 46.42 | 323.19 | **16.53** |
| | GNN | 0.38 | 62.45 | 162.44 | 10.53 |
| | Invariant | 0.14 | 46.08 | 327.06 | 13.17 |
| | Attention | −0.11 | −8.50 | 74.20 | 0.34 |
| VarB | **Pen-DHRL** | 0.11 | 28.66 | 266.58 | **11.42** |
| | GNN | 0.22 | 36.52 | 169.74 | 8.26 |
| | Invariant | 0.08 | 19.89 | 264.46 | 8.79 |
| | Attention | −0.13 | −5.24 | 39.64 | 0.10 |

**Takeaway.** The training regime changes the model ranking. Pen-DHRL's clearest advantage is in defended transfer. Curriculum scheduling improves every architecture and narrows the gap.

---

## Command-Line Reference

Selected flags of `main.py`. Run `python main.py -h` for the full list.

| Flag | Default | Description |
| --- | --- | --- |
| `scenario` (positional) | – | Training scenario(s), `:`-separated |
| `--test_scenario` | – | Held-out evaluation scenario(s) |
| `-net_class` | `NASimNetMLP` | Architecture (`NASimNetDHRL`, `NASimNetGNN_MAct`, `NASimNetInvMAct`, `NASimNetXAttMAct`, ...) |
| `-observation_format` | `list` | `graph_v2` for graph nets (forced automatically for GNN/DHRL) |
| `-device` / `-cpus` / `-batch` | `cpu` / 2 / 128 | Device, worker processes, parallel envs |
| `-epoch` / `-max_epochs` | 1000 / ∞ | PPO steps per interval / number of intervals (`0` = unbounded) |
| `-lr` / `-alpha_h` / `-max_norm` | 3e-3 / 0.3 / 3.0 | Learning rate / entropy coef / grad clip |
| `--alpha_h_manager` | 0.02 | Pen-DHRL manager subgoal-entropy bonus |
| `--sched_lr_*`, `--sched_alpha_h_*` | – | Exponential decay schedules |
| `-mp_iterations` / `-emb_dim` | 3 / 64 | Message passes / embedding size |
| `-episode_step_limit` | 200 | Max steps per episode (paper: 400) |
| `-use_a_t` | off | Allow the agent to terminate the episode |
| `-augment_with_action` | off | Append last action to the observation |
| `-force_continue_epochs` | 0 | Disable termination for the first N epochs |
| `-seed` | – | Seed (env *i* gets `seed + i`) |
| `-load_model` / `--resume` | – | Load checkpoint / resume training state |
| `-save_best_split` / `-save_best_metric` | `eval_tst` / `captured_avg` | Best-checkpoint criterion |
| `--eval` / `--trace` / `--debug` | – | Evaluation / trace / interactive debug modes |

---

## Emulation (inherited from NASimEmu)

DYNAS keeps NASimEmu's emulator: a scenario can be converted into a Vagrant + VirtualBox deployment with a RouterOS router, Ubuntu/Windows targets, and a Kali attacker running Metasploit:

```bash
./setup_vagrant.sh scenarios/<scenario>.v2.yaml   # writes vagrant/Vagrantfile and vagrant/firewall.rsc
```

The DYNAS defense dynamics (IDS, scan noise, churn, timeouts) are **simulation-only**. Emulation does not produce rewards. See the [NASimEmu repository](https://github.com/jaromiru/NASimEmu) for emulation details.

---

## Experimental: LLM-Teacher Extensions

> **Not part of the paper.** These components are research-in-progress and are not used in any reported result.

`NASimEmu-agents/llm_teacher/` adds optional ways to guide Pen-DHRL's manager with an offline teacher (a heuristic teacher or a local LLM via Ollama) over an 8-goal semantic ontology (`goal_ontology.py`):

- `--llm_distill`: confidence-weighted distillation of the subgoal head against a precomputed dataset (`python -m llm_teacher.label_states`).
- `--llm_shaping`: potential-based reward shaping over the active subgoal.

Both require `-net_class NASimNetDHRL`.

---

## Tests

```bash
cd NASimEmu-agents
python -m pytest -q tests
```

> **Note:** The `experiments/` package (fixed-seed evaluation harness, ablation/intervention scripts, report generators) was removed from this branch. The following tests still import it and currently fail at collection: `test_eval_harness`, `test_analyze`, `test_analyze_goals`, `test_analyze_training_log`, `test_component_interventions`, `test_dagger_target_records`, `test_dynamics_recorder`, `test_evaluate_llm_selector`, `test_ids_interventions`, `test_make_followup_report`, `test_make_latex_tables`, `test_make_report`, `test_make_variants`, `test_net_reset_state_ep_t`, `test_scan_noise_observed`. Skip them with, for example, `--ignore=tests/test_eval_harness.py`, or restore `experiments/` from git history.

---

## Known Limitations

- Defense dynamics are simulated abstractions. There is no active defender policy, and firewalls operate at subnet granularity (no host-level rules).
- Inherited from NASimEmu: a firewall either blocks or allows **all** traffic between two subnets; maximum subnet and host counts must be fixed per scenario (`address_space_bounds`) so vector sizes stay constant.
- `trained_models/` contains only the original NASimEmu baseline checkpoints. Pen-DHRL checkpoints from the paper are not included.
- Future work (paper §7): more topology families, active defender policies, and host-level firewall rules.

---

## Citation

If you use DYNAS or Pen-DHRL, please cite:

```bibtex
@inproceedings{hassan2026dynas,
  title     = {{DYNAS}: A Dynamic Network Penetration Testing Framework Using Deep Hierarchical Reinforcement Learning},
  author    = {Hassan, Md. Rakibul and Islam, Majedul and Alam, Md. Ashraful and Hossain, Muhammad Iqbal and Shatabda, Swakkhar},
  booktitle = {TODO: venue},
  year      = {TODO},
}
```

and the NASimEmu work it builds on:

```bibtex
@inproceedings{janisch2024nasimemu,
  title     = {{NASimEmu}: Network Attack Simulator \& Emulator for Training Agents Generalizing to Novel Scenarios},
  author    = {Janisch, Jarom{\'i}r and Pevn{\'y}, Tom{\'a}{\v{s}} and Lis{\'y}, Viliam},
  booktitle = {Computer Security. ESORICS 2023 International Workshops},
  series    = {LNCS},
  volume    = {14399},
  pages     = {589--608},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-031-54129-2_35}
}
```

---

## License and Acknowledgements

Released under the [MIT License](LICENSE).

DYNAS builds on [NASimEmu](https://github.com/jaromiru/NASimEmu) and [NASimEmu-agents](https://github.com/jaromiru/NASimEmu-agents) by Jaromír Janisch et al., which in turn build on the [Network Attack Simulator](https://github.com/Jjschwartz/NetworkAttackSimulator) by Jonathon Schwartz. We thank the original authors for releasing their code.
