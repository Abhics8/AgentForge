import yaml
import numpy as np
import time
from src.train import train

def run_tuning():
    print("="*50)
    print("  Day 9: Hyperparameter Tuning Run")
    print("="*50)

    # We will test 3 different Target Update Frequencies and Learning Rates
    # to see which one provides the most consistent and fastest convergence.
    
    configs_to_test = [
        {"target_update_freq": 1000, "learning_rate": 0.001, "name": "Baseline"},
        {"target_update_freq": 500,  "learning_rate": 0.001, "name": "Fast Target Update"},
        {"target_update_freq": 1000, "learning_rate": 0.0005, "name": "Conservative LR"},
    ]

    best_config = None
    best_convergence = 10000

    for cfg in configs_to_test:
        print(f"\n--- Testing Config: {cfg['name']} ---")
        print(f"Target Update Freq: {cfg['target_update_freq']} | LR: {cfg['learning_rate']}")
        
        # Load and dynamically patch the default yaml
        with open("configs/default.yaml", "r") as f:
            full_cfg = yaml.safe_load(f)
            
        full_cfg["agent"]["target_update_freq"] = cfg["target_update_freq"]
        full_cfg["agent"]["learning_rate"] = cfg["learning_rate"]
        
        with open("configs/temp_tune.yaml", "w") as f:
            yaml.dump(full_cfg, f)
            
        # Run training
        try:
            results = train(config_path="configs/temp_tune.yaml")
            conv_ep = results.get("convergence_episode")
            
            if conv_ep is not None and conv_ep < best_convergence:
                best_convergence = conv_ep
                best_config = cfg
        except Exception as e:
            print(f"Failed: {e}")

    print("\n" + "="*50)
    if best_config:
        print(f"🥇 BEST CONFIGURATON: {best_config['name']}")
        print(f"Converged at Episode: {best_convergence}")
        
        # Update default config permanently
        with open("configs/default.yaml", "r") as f:
            final_cfg = yaml.safe_load(f)
        final_cfg["agent"]["target_update_freq"] = best_config["target_update_freq"]
        final_cfg["agent"]["learning_rate"] = best_config["learning_rate"]
        with open("configs/default.yaml", "w") as f:
            yaml.dump(final_cfg, f, default_flow_style=False)
        print("Updated configs/default.yaml with best hyperparameters.")
    else:
        print("None of the configurations converged! Reverting to baseline.")
    print("="*50)

if __name__ == "__main__":
    run_tuning()
