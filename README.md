# Deep‑Q‑Learning‑Stock‑Trading (PyTorch)

End‑to‑end DQN baseline for **single‑asset stock trading** with **Gymnasium** environment, a modular PyTorch agent (MLP/CNN/LSTM backbones, **Double/Dueling DQN**), and training utilities. It’s intentionally minimal so you can fork fast, iterate faster, and ship learnings.

<p align="center">
  <img src="./assets/reward_plot.png" alt="Total rewards per episode chart" width="720"><br>
  <em>Total reward per episode (10-episode moving average in orange).</em>
</p>

---

## 🔎 Scope
This repo is a research sandbox that demonstrates: (1) how to build a small, reproducible RL stack for markets; (2) how reward shaping and state design drive outcomes. It can be extended for real constraints (costs, sizing, risk).

---

## ✨ Features
- **Custom Gymnasium env**: `StockTrading-v0` with `Buy/Hold/Sell` discrete actions, FIFO inventory, realized‑P&L rewards.
- **State**: `window_size` most recent **price diffs** (1‑D float32), compatible with `MlpPolicy` style nets.
- **Agent**: DQN w/ toggles for **Double DQN**, **Dueling**, soft (Polyak) or hard target updates.
- **Backbones**: MLP, 1D‑CNN, LSTM (dueling variants included)..
- **Data**: `yfinance` adjusted closes; train window set in config.
- **Artifacts**: per‑run folder under `results/` with `hyperparameters.txt`, `reward_plot.png`, and optional checkpoint.

---

## 📦 Environment setup
Use Conda (recommended):

```bash
git clone https://github.com/Tahernezhad/Deep-Q-Learning-Stock-Trading.git
cd Deep-Q-Learning-Stock-Trading
conda env create -f environment.yml
conda activate rl
```

If you’re CPU‑only, remove CUDA lines in the `environment.yml` or let Conda resolve a CPU build.

---

## 🗂️ Project structure
```
Deep-Q-Learning-Stock-Trading/
├── config.py            # All switches: data window, algo toggles, HParams
├── stock_env.py         # Gymnasium env: Buy/Hold/Sell, FIFO inventory, rewards
├── dqn_agent.py         # DQN w/ Double & Dueling options + soft/hard target updates
├── networks.py          # MLP, 1D‑CNN, LSTM (+ dueling variants)
├── replay_buffer.py     # Uniform experience replay
├── utils.py             # Seeding, plotting, checkpoint & config save
├── main.py              # Training entry point
├── environment.yml      # Conda spec
└── results/             # Auto‑created per‑run folders
```
Each run creates `results/StockTrading-v0_YYYYmmdd_HHMMSS/` with:
```
- hyperparameters.txt
- reward_plot.png
- best_model.pth        # if SAVE_MODEL=True
- total_rewards.txt
```

---

## ⚙️ Configure
Edit **`config.py`**.

**Data & env**
- `ENV_NAME = 'StockTrading-v0'`
- `TICKER = 'AAPL'`   # pick any supported by `yfinance`
- `START_DATE`, `END_DATE`  # train window
- `WINDOW_SIZE = 5`   # length of price‑diff window (state)

**Agent & algorithm**
- `MODEL_TYPE = 'MLP' | 'CNN1D' | 'LSTM'`
- `double_dqn = True|False`
- `dueling_network = True|False`
- `SOFT_UPDATE = True|False`, `TAU = 0.005`  # Polyak
- `TARGET_UPDATE_FREQ` (used when `SOFT_UPDATE=False`)
- `LOSS = 'huber' | 'mse'`

**Optimization & exploration**
- `LEARNING_RATE`, `BATCH_SIZE`, `REPLAY_BUFFER_SIZE`, `WARMUP_STEPS`
- `GAMMA`
- `EPSILON_START`, `EPSILON_END`, `EPSILON_DECAY`

**Run control**
- `NUM_EPISODES`, `MOVING_AVG_WINDOW`, `REPORT_INTERVAL`, `SEED`
- `SAVE_MODEL = True|False`

---

## 🚀 Train
```bash
python main.py
```
Artifacts are written to `results/StockTrading-v0_<timestamp>/`.

### Visualizing rewards
Open `reward_plot.png` in the run folder. The blue line is per‑episode reward; the orange line is the moving average you define via `MOVING_AVG_WINDOW`.

---

## 🧠 Environment design (TL;DR)
- **Actions**: `0=Hold`, `1=Buy`, `2=Sell`.
- **Inventory**: unlimited long FIFO queue (first‑in sells first). No shorting by default.
- **Reward**: realized P&L only (profit appears **on sell**); holding has zero reward.
- **State**: last `window_size` price **differences** (left‑padded at start) — a simple stationary-ish signal.
- **Termination**: end of historical series.

> This is deliberately spartan. It’s great for method testing, not capital allocation.

---

## 📉 Known limitations / To‑dos
- No transaction costs, slippage, or latency.
- No position sizing; buys always add one unit.
- No risk metrics (Sharpe/Sortino), drawdown, or live PnL tracking beyond realized profit.
- No walk‑forward or train/validation split baked in.

**Roadmap**
- Commission/slippage, borrow fees; capped inventory; optional shorting.
- Position‑sizing actions (discrete or continuous) and cash accounting.
- Train/val/test split, walk‑forward evaluation, and early stopping.
- Metric suite: Sharpe, max drawdown, hit rate; TensorBoard logging.
- Portfolio env for multi‑asset allocation.

---

## 🔧 Troubleshooting
- **`yfinance` empty frame**: check ticker/date range; ensure internet; try a different asset.
- **Divergent rewards**: widen `EPSILON_DECAY`, increase `REPLAY_BUFFER_SIZE`, or reduce `LEARNING_RATE`.
- **Non‑determinism**: seeds help but RL is stochastic. Expect run‑to‑run variance.

---

## 🙌 Acknowledgements
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [PyTorch](https://pytorch.org/)
- [yfinance](https://github.com/ranaroussi/yfinance)

---

## 📣 Citation
If this repo helps your work, star it and cite it in your README/paper notes. Then go build the next experiment — this is a launchpad, not a landing zone.

