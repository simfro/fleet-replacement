from gymnasium.envs.registration import register

register(
    id="FleetReplacement-v0",
    entry_point="fleet_replacement.envs.fleet_replacement:FleetReplacementEnv",
    max_episode_steps=None,
)
