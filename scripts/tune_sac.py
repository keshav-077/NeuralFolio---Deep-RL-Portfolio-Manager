# scripts/tune_sac.py 

import os
import sys
import pandas as pd
import numpy as np
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv  # Use DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from environment import PortfolioEnv

# ==============================================================================
# 1. Configuration & Data Loading
# ==============================================================================

TRAIN_DATA_PATH = 'data/train.csv'
EVAL_DATA_PATH = 'data/eval.csv'
OPTUNA_LOG_DIR = 'optuna_logs'
CHECKPOINT_DIR = 'checkpoints/optuna_sac_trials'

# Create directories if they don't exist
os.makedirs(OPTUNA_LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load data once
df_full_train = pd.read_csv(TRAIN_DATA_PATH, index_col='Date', parse_dates=True)
df_eval = pd.read_csv(EVAL_DATA_PATH, index_col='Date', parse_dates=True)

# Split df_full_train for tuning
train_split_point = int(len(df_full_train) * 0.8)
df_train_tune = df_full_train.iloc[:train_split_point]
df_validation_tune = df_full_train.iloc[train_split_point:]

print(f"Total training data points: {len(df_full_train)}")
print(f"Optuna training data points: {len(df_train_tune)}")
print(f"Optuna validation data points: {len(df_validation_tune)}")


# ==============================================================================
# 2. Environment Creation Helper
# ==============================================================================

def make_env(df, window_size=30, initial_balance=10000, transaction_cost_pct=0.001):
    """
    Helper function to create a PortfolioEnv instance.
    """
    def _init():
        env = PortfolioEnv(
            df=df,
            initial_balance=initial_balance,
            window_size=window_size,
            transaction_cost_pct=transaction_cost_pct
        )
        return env
    return _init

# ==============================================================================
# 3. Optuna Objective Function
# ==============================================================================

def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna to optimize hyperparameters for SAC.
    """
    # Hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    tau = trial.suggest_float('tau', 0.005, 0.02)
    buffer_size = trial.suggest_int('buffer_size', 50000, 1000000, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    ent_coef = trial.suggest_float('ent_coef', 0.001, 0.1, log=True) # Use log scale for ent_coef

    # Network architecture
    n_layers = trial.suggest_int('n_layers', 1, 3)
    net_arch = []
    for i in range(n_layers):
        layer_size = trial.suggest_categorical(f'layer_size_{i}', [64, 128, 256])
        net_arch.append(layer_size)

    policy_kwargs = dict(net_arch=net_arch) # SAC uses shared network or separate [pi, qf]

    # Create environments for this trial
    train_env = DummyVecEnv([make_env(df_train_tune)])
    eval_env = DummyVecEnv([make_env(df_validation_tune)])

    # Set up logger for the trial
    trial_log_path = os.path.join(OPTUNA_LOG_DIR, f"trial_{trial.number}")
    new_logger = configure(trial_log_path, ["stdout", "csv", "tensorboard"])

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        buffer_size=buffer_size,
        batch_size=batch_size,
        ent_coef=ent_coef, # Pass the sampled value
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=42, # Use a fixed seed for reproducibility within a trial
        tensorboard_log=OPTUNA_LOG_DIR
    )
    model.set_logger(new_logger)

    # Callback for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(CHECKPOINT_DIR, f"best_sac_trial_{trial.number}"),
        log_path=trial_log_path,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=1
    )

    try:
        # Train for a set number of steps per trial
        total_timesteps_per_trial = 50000
        model.learn(total_timesteps=total_timesteps_per_trial, callback=eval_callback, progress_bar=False)

        # Load the best model found during this trial's training
        best_model_path = os.path.join(CHECKPOINT_DIR, f"best_sac_trial_{trial.number}", "best_model.zip")
        if os.path.exists(best_model_path):
            model = SAC.load(best_model_path, env=eval_env)
        else:
            print(f"Warning: No best model saved for trial {trial.number}, using last model.")

        # --- Final evaluation on the validation set ---
        obs = eval_env.reset()
        portfolio_values = [eval_env.envs[0].initial_balance]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            portfolio_values.append(info[0]['portfolio_value'])

        final_portfolio_value = portfolio_values[-1]
        initial_portfolio_value = portfolio_values[0]
        total_return = (final_portfolio_value / initial_portfolio_value) - 1

        print(f"Trial {trial.number} finished. Total Return on validation: {total_return:.4f}")

    except Exception as e:
        print(f"Trial {trial.number} failed due to: {e}")
        return float('nan') # Optuna handles NaN as a failure

    finally:
        train_env.close()
        eval_env.close()

    return total_return # Optuna aims to maximize this metric


# ==============================================================================
# 4. Run Optuna Study
# ==============================================================================

if __name__ == '__main__':
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    n_trials_to_run = 50
    study.optimize(objective, n_trials=n_trials_to_run, n_jobs=1) # n_jobs=1 is safer for Colab

    print("\n--- Optimization finished. ---")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the best parameters to a file
    best_params = trial.params
    with open('checkpoints/best_sac_params.txt', 'w') as f:
        f.write(str(best_params))
    print(f"\n✅ Best parameters saved to checkpoints/best_sac_params.txt")

    # Plotting results
    try:
        import plotly
        from optuna.visualization import plot_optimization_history, plot_param_importances

        fig1 = plot_optimization_history(study)
        fig1.show()

        fig2 = plot_param_importances(study)
        fig2.show()
    except ImportError:
        print("\nInstall plotly and kaleido to visualize Optuna results: !pip install plotly kaleido")