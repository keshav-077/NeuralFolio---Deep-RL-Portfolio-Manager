import argparse
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3
from environment import PortfolioEnv

def train_agent(agent_name="td3", timesteps=100000):
    """
    Main function to train a specified RL agent.

    Args:
        agent_name (str): The RL algorithm to use ('ppo', 'sac', or 'td3').
        timesteps (int): The total number of timesteps for training.
    """
    # 1. Map agent names to their corresponding classes
    AGENT_CLASSES = {
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3
    }
    agent_class = AGENT_CLASSES.get(agent_name.lower())
    if agent_class is None:
        raise ValueError(f"Unknown agent: {agent_name}. Choose from {list(AGENT_CLASSES.keys())}")

    model_name = agent_name.lower()

    # 2. Load data and create the environment
    print("--- Loading Data and Creating Environment ---")
    try:
        df = pd.read_csv('data/train.csv', index_col='Date', parse_dates=True)
        env = PortfolioEnv(df)
        print("Environment created successfully.")
    except FileNotFoundError:
        print("❌ Error: 'data/train.csv' not found. Make sure to run a data fetching script first.")
        return

    # 3. Create the RL Agent
    print(f"--- Creating {agent_name.upper()} Agent ---")
    model = agent_class(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    # 4. Train the Agent
    print(f"--- Starting Agent Training for {timesteps} timesteps ---")
    model.learn(total_timesteps=timesteps)
    print("--- Agent Training Complete ---")

    # 5. Save the Trained Model
    save_path = f"checkpoints/{model_name}_portfolio_model"
    model.save(save_path)
    print(f"✅ Model saved to checkpoints/{save_path}.zip")


if __name__ == "__main__":
    # 6. Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Train a Reinforcement Learning agent for portfolio management.")
    
    parser.add_argument(
        "--agent",
        type=str,
        default="td3",
        choices=["ppo", "sac", "td3"],
        help="The RL algorithm to use for training (default: td3)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="The total number of timesteps for training (default: 100000)."
    )
    
    args = parser.parse_args()

    # Call the main training function with the parsed arguments
    train_agent(agent_name=args.agent, timesteps=args.timesteps)