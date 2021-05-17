from PIL import Image
import gym
import custom_envs

env_id = 'AntWall-v0'
env = gym.make(env_id)
_ = env.reset()
for i in range(5):
    _ = env.step(env.action_space.sample())
image = env.render(mode='rgb_array', camera_id=0)
#image = env.render(mode='rgb_array')
im = Image.fromarray(image)
im.save('./icrl/plots/env_plots/antWall3.png')
