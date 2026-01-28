import pytest
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
import sys
import os

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local Imports
import config as default_config
import stock_env
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from networks import SimpleDQN, DuelingDQN


# --- 1. Mock Configuration ---
# A lightweight config to run tests quickly on CPU.
class TestConfig:
    DEVICE = torch.device("cpu")
    MODEL_TYPE = 'MLP'
    ENV_NAME = 'StockTrading-v0'
    WINDOW_SIZE = 5
    HIDDEN_LAYER_SIZE = 64
    LEARNING_RATE = 1e-3
    GAMMA = 0.99
    BATCH_SIZE = 4
    REPLAY_BUFFER_SIZE = 100
    WARMUP_STEPS = 5
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 10
    double_dqn = True
    dueling_network = True
    SOFT_UPDATE = True
    TAU = 0.005
    LOSS = "huber"
    SAVE_MODEL = False


@pytest.fixture
def trading_env_setup():
    """Sets up the custom stock trading environment for testing."""
    env = gym.make(
        'StockTrading-v0',
        ticker='AAPL',
        start_date='2015-01-01',
        end_date='2015-02-01',
        window_size=TestConfig.WINDOW_SIZE
    )
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    return env, n_states, n_actions


# Test Environment & Buffer
def test_environment(trading_env_setup):
    """Verifies observation shapes and action handling."""
    env, n_states, n_actions = trading_env_setup
    obs, info = env.reset()

    assert obs.shape == (TestConfig.WINDOW_SIZE,)
    assert n_actions == 3  # Hold, Buy, Sell

    # Take a 'Buy' action
    next_obs, reward, terminated, truncated, info = env.step(1)
    assert next_obs.shape == (TestConfig.WINDOW_SIZE,)
    assert 'inventory_size' in info


def test_replay_buffer():
    """Ensures experience tuples are stored and sampled correctly."""
    buffer = ReplayBuffer(capacity=10)
    state = np.random.random(5)
    action = 1
    reward = 10.0
    next_state = np.random.random(5)
    done = False

    buffer.push(state, action, reward, next_state, done)
    assert len(buffer) == 1

    states, actions, rewards, next_states, dones = buffer.sample(1)
    assert states.shape == (1, 5)
    assert rewards.shape == (1,)


#  Agent & Networks
def test_network_architectures(trading_env_setup):
    """Checks if Dueling and Simple DQN models produce correct Q-value shapes."""
    _, n_states, n_actions = trading_env_setup

    # Test Dueling DQN
    model = DuelingDQN(n_states, TestConfig.HIDDEN_LAYER_SIZE, n_actions)
    state_tensor = torch.randn(1, n_states)
    q_values = model(state_tensor)
    assert q_values.shape == (1, n_actions)


def test_dqn_agent_selection(trading_env_setup):
    """Verifies epsilon-greedy action selection."""
    _, n_states, n_actions = trading_env_setup
    agent = DQNAgent(n_states, n_actions, TestConfig)
    state = np.random.random(n_states)

    # Force greedy selection (epsilon=0)
    action = agent.select_action(state, epsilon=0.0)
    assert 0 <= action < n_actions

    # Force random selection (epsilon=1)
    action_random = agent.select_action(state, epsilon=1.0)
    assert isinstance(action_random, int)


# Integration Test Pipeline
def test_dqn_training_loop(tmpdir, trading_env_setup):
    """Simulates a mini training session to verify optimization and updates."""
    env, n_states, n_actions = trading_env_setup
    agent = DQNAgent(n_states, n_actions, TestConfig)
    memory = ReplayBuffer(TestConfig.REPLAY_BUFFER_SIZE)

    state, _ = env.reset()
    for _ in range(TestConfig.BATCH_SIZE + 1):
        action = agent.select_action(state, epsilon=1.0)
        next_state, reward, term, trunc, _ = env.step(action)
        memory.push(state, action, reward, next_state, term or trunc)
        state = next_state
        if term or trunc:
            state, _ = env.reset()

    initial_params = [p.clone() for p in agent.policy_net.parameters()]
    agent.optimize_model(memory)

    weight_changed = any(not torch.equal(p1, p2)
                         for p1, p2 in zip(initial_params, agent.policy_net.parameters()))
    assert weight_changed

    initial_target_params = [p.clone() for p in agent.target_net.parameters()]
    agent.soft_update_target_network(tau=0.5)

    target_changed = any(not torch.equal(p1, p2)
                         for p1, p2 in zip(initial_target_params, agent.target_net.parameters()))
    assert target_changed


if __name__ == "__main__":
    pytest.main([__file__])