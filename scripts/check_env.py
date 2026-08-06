import pandas as pd
from stable_baselines3.common.env_checker import check_env
from environment import PortfolioEnv

def main():
    """
    Main function to create and check the custom portfolio environment.
    """
    print("--- Loading Data and Creating Environment ---")
    try:
        # Load your training data
        df = pd.read_csv('data/train.csv', index_col='Date', parse_dates=True)
        # Create an instance of your environment
        env = PortfolioEnv(df)
        print("Environment created successfully.")
    except FileNotFoundError:
        print("❌ Error: 'data/train.csv' not found. Make sure you've run the data fetching script.")
        return

    print("\n--- Checking Environment Compatibility ---")
    try:
        # The check_env function will raise an error if the environment is not compatible.
        check_env(env)
        print("✅ Environment check passed!")
    except Exception as e:
        print("❌ Environment check failed:")
        # It's helpful to print the full traceback for debugging complex errors.
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()