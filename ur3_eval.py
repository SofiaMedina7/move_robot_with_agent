import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env

from ur3_env import ReachingUR3

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        # Exact structure from trained model
        self.net_container = nn.Sequential(
            nn.Linear(25, 64),  # First layer: 25 -> 64
            nn.ELU(),
            nn.Linear(64, 64),  # Second layer: 64 -> 64
            nn.ELU(),
        )
        
        self.policy_layer = nn.Linear(64, 6)  # Output layer: 64 -> 6
        self.value_layer = nn.Linear(64, 1)   # Value layer: 64 -> 1
        
        self.log_std_parameter = nn.Parameter(torch.zeros(6))  # 6 dimensions

    def compute(self, inputs, role):
        features = self.net_container(inputs["states"])
        actions = self.policy_layer(features)
        return actions, self.log_std_parameter, {}

# Load the environment
env = ReachingUR3()
env = wrap_env(env)
device = env.device

# Setup policy and agent
models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)

# Configure agent
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space.shape, "device": device}
# Optional: Configure experiment logging if needed
cfg_ppo["experiment"]["write_interval"] = 32
cfg_ppo["experiment"]["checkpoint_interval"] = 0

# Create agent
agent = PPO(
    models=models_ppo,
    memory=None,
    cfg=cfg_ppo,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device
)

# Load the agent using the agent.load() method
print("Loading agent...")
agent.load("best_agent.pt")

# Configure trainer
cfg_trainer = {"timesteps": 1000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# Start evaluation
print("Starting evaluation...")
trainer.eval()