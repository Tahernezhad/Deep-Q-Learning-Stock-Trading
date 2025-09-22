import torch

# --- Model Selection ---
MODEL_TYPE = 'MLP' # Options: 'MLP', 'CNN1D', 'LSTM'

# --- Training Settings ---
ENV_NAME = 'StockTrading-v0'
TICKER = 'AAPL'  # S&P 500. Try 'AAPL', 'GOOG', 'MSFT', etc.
START_DATE = '2015-01-01'
END_DATE = '2020-12-31'
WINDOW_SIZE = 5

# --- Agent Hyperparameters ---
LEARNING_RATE = 0.0005
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 200000
TARGET_UPDATE_FREQ = 1000
HIDDEN_LAYER_SIZE = 256

WARMUP_STEPS = 500

# --- Epsilon-Greedy Strategy ---
EPSILON_START = 1.0
EPSILON_END = 0.01

EPSILON_DECAY = 100000

# --- DQN Variants ---
double_dqn = True
dueling_network = True

# --- Training Loop Settings ---
NUM_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 1000
REPORT_INTERVAL = 100
MOVING_AVG_WINDOW = 100
SEED = 42

LOSS = "huber"  # options: "huber", "mse"

SOFT_UPDATE = True
TAU = 0.005  # Polyak coefficient

# --- Device Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Save/Load Settings ---
SAVE_MODEL = True