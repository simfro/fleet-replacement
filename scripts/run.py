import gymnasium as gym

# Import to trigger environment registration
import fleet_replacement  # noqa: F401


env = gym.make("FleetReplacement-v0")

env

# obs, info = env.reset(seed=42)
# env.render()

# episode_reward = 0.0
# for step in range(10):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     episode_reward += reward
#     env.render()

#     if terminated or truncated:
#         print(f"\nEpisode ended at step {step + 1}")
#         break

# print(f"Episode reward: {episode_reward}")
# env.close()
