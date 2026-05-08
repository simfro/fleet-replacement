import gymnasium as gym

# Import to trigger environment registration
import fleet_replacement  # noqa: F401

env = gym.make("FleetReplacement-v0")

# %%
obs, info = env.reset()

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
