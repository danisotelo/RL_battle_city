from gymnasium.envs.registration import register

register(
     id="gym_tanks/tanks-v0",
     entry_point="gym_tanks.envs:TanksEnv",
     max_episode_steps=300,
)