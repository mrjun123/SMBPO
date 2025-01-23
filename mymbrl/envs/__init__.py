# from .cartpole import CartpoleEnv
# from .half_cheetah import HalfCheetahEnv
# from .pusher import PusherEnv
# from .reacher import Reacher3DEnv
import importlib
import numpy as np

import torch
import os
import inspect
import random
import gymnasium

class handReachWrap:

    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.init()

    def init(self):
        env = self.env
        name = self.name
        self.action_space = env.action_space

        original_space = env.observation_space['observation']
        original_low = original_space.low
        original_high = original_space.high

        desired_goal_space = env.observation_space['desired_goal']
        self.desired_goal_space = desired_goal_space
        self.desired_goal_dim = desired_goal_space.shape[-1]
        # new_low = np.insert(original_low, 0, -np.inf)
        # new_high = np.insert(original_high, 0, np.inf)

        new_low = np.concatenate((original_low, desired_goal_space.low))
        new_high = np.concatenate((original_high, desired_goal_space.high))

        
        new_space = gymnasium.spaces.Box(low=new_low, high=new_high, dtype=original_space.dtype)

        self.observation_space = new_space
        self.MODEL_IN, self.MODEL_OUT = original_low.shape[-1]+self.action_space.low.shape[-1], original_low.shape[-1]
        self.POLICY_IN = original_low.shape[-1]+self.action_space.low.shape[-1] + self.desired_goal_dim

    def seed(self, seed):
        # self.env.seed(seed)
        self.rseed = seed
        self.action_space.seed(seed)
    
    def step(self, a):
        # self.t += 1
        s, reward, done, truncated, info = self.env.step(a)
        observation = s['observation']
        desired_goal = s['desired_goal']
        ob = np.concatenate([
            observation,
            desired_goal
        ])
        reward = reward*10
        # if info['is_success']:
        #     done = True
        
        return ob, reward, done, info
    
    def reset(self, *args, **kwargs):

        s, info = self.env.reset(*args, **kwargs)
        observation = s['observation']
        desired_goal = s['desired_goal']
        self.cur_desired_goal = desired_goal
        ob = np.concatenate([
            observation,
            desired_goal
        ])
        return ob
    
    @staticmethod
    def ac_cost_fn_cost(acs):
        return 0

    def obs_cost_fn_cost(self, obs):

        goal_a = obs[..., -self.desired_goal_dim:]
        goal_b = obs[..., -self.desired_goal_dim*2:-self.desired_goal_dim]

        if isinstance(obs, np.ndarray):
            cost = np.linalg.norm(goal_a - goal_b, axis=-1)
        else:
            cost = torch.norm(goal_a - goal_b, dim=-1)

        # if 'Reach' in self.name:
        cost = cost*10

        return cost

    def obs_model_preproc(self, obs):
        return obs[..., :-self.desired_goal_dim]
    
    def obs_model_postproc(self, obs, model_obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([model_obs, obs[..., -self.desired_goal_dim:]], axis=-1)
        else:
            return torch.cat([model_obs, obs[..., -self.desired_goal_dim:]], dim=-1)
    
    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, es, pnes):
        return es + pnes
    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

def get_item(name):

    def create_env():
        
        env = gymnasium.make(name, render_mode='rgb_array')
        env = handReachWrap(env, name)
        return env
    return create_env