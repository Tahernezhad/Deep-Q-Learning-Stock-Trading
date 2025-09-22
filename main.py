import math
import gymnasium as gym
from pathlib import Path
from collections import deque
from datetime import datetime

import config
import stock_env
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from utils import plot_rewards, save_checkpoint, set_seed, save_config, save_rewards


def main():
    """Main function to run the training loop."""
    # --- Setup ---
    set_seed(config.SEED)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path("results") / f"{config.ENV_NAME}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, results_dir / "hyperparameters.txt")

    # --- Environment ---
    env = gym.make(
        config.ENV_NAME,
        ticker=config.TICKER,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        window_size=config.WINDOW_SIZE,
        render_mode="None"  # human or None
    )

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # --- Agent and Memory ---
    agent = DQNAgent(n_states, n_actions, config)
    memory = ReplayBuffer(config.REPLAY_BUFFER_SIZE)

    # --- Training Loop ---
    print(f"Starting training on {config.DEVICE}...")
    total_rewards = []
    total_steps = 0
    recent_scores = deque(maxlen=config.MOVING_AVG_WINDOW)

    best_avg_score = -float('inf')

    for i_episode in range(config.NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0.0

        while True:
            epsilon = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * \
                      math.exp(-1. * total_steps / config.EPSILON_DECAY)
            total_steps += 1

            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            memory.push(state, action, reward, next_state, done)
            state = next_state

            agent.optimize_model(memory)

            if config.SOFT_UPDATE:
                agent.soft_update_target_network(config.TAU)
            elif total_steps % config.TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            if done:
                break

        total_rewards.append(episode_reward)
        recent_scores.append(episode_reward)
        avg_score = sum(recent_scores) / len(recent_scores)

        if (i_episode + 1) % config.REPORT_INTERVAL == 0:
            print(f"Episode {i_episode + 1}/{config.NUM_EPISODES} | "
                  f"Avg Score (last {config.MOVING_AVG_WINDOW}): {avg_score:.2f} | "
                  f"Epsilon: {epsilon:.4f}")

        if len(recent_scores) == config.MOVING_AVG_WINDOW and avg_score > best_avg_score:
            best_avg_score = avg_score
            if config.SAVE_MODEL:
                save_path = results_dir / "best_model.pth"
                save_checkpoint(agent, save_path)

    print("Training complete.")

    # --- Save Final Plot ---
    plot_path = results_dir / "reward_plot.png"
    plot_rewards(total_rewards, plot_path, config.MOVING_AVG_WINDOW)
    rewards_path = results_dir / "total_rewards.txt"
    save_rewards(total_rewards, rewards_path)
    env.close()


if __name__ == "__main__":
    main()


