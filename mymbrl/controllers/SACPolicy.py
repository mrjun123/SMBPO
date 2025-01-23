import mymbrl.optimizers as optimizers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mymbrl.utils import swish
from .controller import Controller
from mymbrl.mfrl.sac.replay_memory import ReplayMemory
from mymbrl.mfrl.sac.sac import SAC
from mymbrl.envs.utils import termination_fn
import copy
from numbers import Number
import time
import importlib

"""
使用Agent训练的Model和收集的数据训练一个SAC Agent。
"""

class SACPolicy(Controller):
    def __init__(self, agent, is_torch=True, writer=None):
        
        super(SACPolicy, self).__init__(agent, is_torch, writer)
        env = self.env
        config = self.config
        self.config.agent.SACPolicy.epoch_length = self.config.experiment.horizon
        args = self.config.agent.SACPolicy
        args.crossq_config = self.config.agent.MBAC.toDict()
        
        self.dO, self.dU = env.observation_space.shape[0], env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.action_dim = self.ac_ub.shape[0]
        
        actions_num = config.agent.predict_length
        self.actions_num = actions_num

        self.lower_bound = torch.tile(torch.tensor(self.ac_lb), [actions_num]).to(self.config.device)
        self.upper_bound = torch.tile(torch.tensor(self.ac_ub), [actions_num]).to(self.config.device)
        
        self.init_sol = (self.lower_bound + self.upper_bound) / 2

        self.predict_env = PredictEnv(env, agent, config)
        # epoch_length = config.experiment.horizon
        # model_retain_epochs = args.model_retain_epochs
        # rollouts_per_epoch = args.rollout_batch_size * epoch_length / config.agent.model_train_freq
        # model_steps_per_epoch = int(1 * rollouts_per_epoch)

        self.rollout_length = 1
        self.train_policy_steps = 0

        # new_pool_size = model_retain_epochs * model_steps_per_epoch
        # self.model_pool = ReplayMemory(new_pool_size)
        self.model_pool = self.resize_model_pool(self.rollout_length)
        self.env_pool = ReplayMemory(args.replay_size)

        module = importlib.import_module(args.mfrl_agent_path)
        AgentClass = getattr(module, args.mfrl_agent_name)
        # print('env.MODEL_IN - env.action_space.shape[0]', env.MODEL_IN - env.action_space.shape[0])
        sa_dim = env.MODEL_IN
        if hasattr(env, 'POLICY_IN'):
            sa_dim = env.POLICY_IN
        self.SAC_agent = AgentClass(sa_dim - env.action_space.shape[0], env.action_space, args, self.agent)

        self.episode_num = 0
        self.check_episode_num = 5
        self.episode_rewards = []

        self.smooth_return = 0

        if args.mpc_rollout:
            Optimizer = optimizers.get_item(config.agent.optimizer)
            self.optimizer = Optimizer(
                sol_dim=actions_num * self.action_dim,
                config=self.config,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound
            )
        self.best_epoch_reward = -1e5
        self.best_SAC_agent = None
        self.best_start_SAC_agent = None
        self.encoder_enable = False

        self.checkpoint_args = self.config.agent.SACPolicy.checkpoint
        self.training_steps = 0
        self.all_training_steps = 0
        self.timesteps_since_rollout = 0

        self.use_checkpoint = self.config.agent.SACPolicy.use_checkpoint
        # Checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = self.checkpoint_args.max_eps_before_update
        # self.min_return = 1e8
        self.min_return = 1e8
        self.max_return = -1e8
        self.mean_return = 0
        # self.best_min_return = -1e8
        self.best_min_return = -1e8
        self.best_max_return = -1e8
        self.best_mean_return = -1e8

        self.obs_max = -np.ones(self.dO)*1000
        self.obs_min = np.ones(self.dO)*1000
    
    def get_parameters_by_name(self, model: torch.nn.Module, included_names):
        return [param for name, param in model.state_dict().items() if any([key in name for key in included_names])]
    
    def hard_update(self, target, source):
        if type(target) == list:
            for target_param, param in zip(target, source):
                target_param.data.copy_(param.data)
        else:
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)
        
        for target_buffer, buffer in zip(target.buffers(), source.buffers()):
            target_buffer.data.copy_(buffer.data)
        
        batch_norm_stats = self.get_parameters_by_name(source, ["running_"])
        target_batch_norm_stats = self.get_parameters_by_name(target, ["running_"])
        for target_param, param in zip(target_batch_norm_stats, batch_norm_stats):
            target_param.data.copy_(param.data)

    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):

        self.timesteps_since_update += ep_timesteps
        self.writer.add_scalar(f'mbrl/episode_rewards', ep_return, self.episode_num)
        smooth_rate = self.checkpoint_args.smooth_rate

        if ep_return >= self.smooth_return:
            self.writer.add_scalar(f'mbrl/checkpoint_rewards', ep_return, self.episode_num)
            self.hard_update(self.SAC_agent.checkpoint_policy, self.SAC_agent.policy)
        
        if self.episode_num == 1:
            self.smooth_return = ep_return
        else:
            weighted_smooth_rate = 1-(1-smooth_rate)*(ep_timesteps/self.config.experiment.horizon)
            self.smooth_return = (weighted_smooth_rate * self.smooth_return) + ((1-weighted_smooth_rate) * ep_return)
            # self.smooth_return = (smooth_rate * self.smooth_return) + ((1-smooth_rate) * ep_return)
        self.train_and_reset()
    
    def maybe_train_and_checkpoint_back2(self, ep_timesteps, ep_return):

        self.timesteps_since_update += ep_timesteps
        self.writer.add_scalar(f'mbrl/episode_rewards', ep_return, self.episode_num)
        smooth_episode_num = self.checkpoint_args.smooth_episode_num

        if self.episode_num <= smooth_episode_num or ep_return >= np.array(self.episode_rewards[-smooth_episode_num:]).mean():
            
            self.writer.add_scalar(f'mbrl/checkpoint_rewards', ep_return, self.episode_num)
            self.hard_update(self.SAC_agent.checkpoint_policy, self.SAC_agent.policy)
        self.episode_rewards.append(ep_return)
        self.train_and_reset()
    
    def maybe_train_and_checkpoint_back(self, ep_timesteps, ep_return):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        # self.episode_num > self.check_episode_num

        # self.check_episode_num

        self.writer.add_scalar(f'mbrl/episode_rewards', ep_return, self.episode_num)

        self.min_return = min(self.min_return, ep_return)
        self.max_return = max(self.max_return, ep_return)

        self.mean_return += ep_return

        mean_return = self.mean_return / self.eps_since_update
        if self.min_return < self.best_min_return:
            self.best_min_return *= self.checkpoint_args.decay_weight
            self.train_and_reset()

        elif self.eps_since_update == self.max_eps_before_update:

            best_return_diff = self.best_max_return - self.best_min_return
            best_return_diff = max(best_return_diff, 0)

            if mean_return > self.best_mean_return - 0.2*best_return_diff:
                self.best_min_return = self.min_return
                self.best_max_return = self.max_return
                self.best_mean_return = mean_return

                self.writer.add_scalar(f'mbrl/checkpoint_rewards', ep_return, self.exp_epoch)
                
                # ep_return

                self.hard_update(self.SAC_agent.checkpoint_policy, self.SAC_agent.policy)
            # self.SAC_agent.checkpoint_policy.load_state_dict(self.SAC_agent.policy.state_dict())
            self.train_and_reset()

    # Batch training
    def train_and_reset(self):
        # if self.timesteps_since_update > 1000:
        #     self.timesteps_since_update = 1000
        for i in range(self.timesteps_since_update):
            if self.training_steps == self.checkpoint_args.steps_before_checkpointing:
                self.best_min_return *= self.checkpoint_args.reset_weight
                self.max_eps_before_update = self.checkpoint_args.max_eps_when_checkpointing
            # if (i+1) % model_train_freq == 0 and self.timesteps_since_update - i >= model_train_freq:
            #     self.train_epoch(epoch_reward=0)
            self.train_policy_steps += self.train_step()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.mean_return = 0
        self.min_return = 1e8
        self.max_return = -1e8
        self.timesteps_since_rollout = 0
    # If using checkpoints: run when each episode terminates
    # def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
    #     self.eps_since_update += 1
    #     self.timesteps_since_update += ep_timesteps

    #     self.min_return = min(self.min_return, ep_return)

    #     if self.min_return < self.best_min_return:
    #         self.best_min_return *= self.checkpoint_args.decay_weight
    #         self.train_and_reset()

    #     elif self.eps_since_update == self.checkpoint_args.max_eps_before_update:
    #         self.best_min_return = self.min_return
    #         self.hard_update(self.SAC_agent.checkpoint_policy, self.SAC_agent.policy)
    #         # self.SAC_agent.checkpoint_policy.load_state_dict(self.SAC_agent.policy.state_dict())
    #         self.train_and_reset()

    # # Batch training
    # def train_and_reset(self):
    #     for i in range(self.timesteps_since_update):
    #         if self.training_steps == self.checkpoint_args.steps_before_checkpointing:
    #             self.best_min_return *= self.checkpoint_args.reset_weight
    #             self.max_eps_before_update = self.checkpoint_args.max_eps_when_checkpointing
    #         model_train_freq = self.config.agent.model_train_freq
    #         if (i+1) % model_train_freq == 0 and self.timesteps_since_update - i >= model_train_freq:
    #             self.train_epoch(epoch_reward=0)
    #         self.train_policy_steps += self.train_step()

    #     self.eps_since_update = 0
    #     self.timesteps_since_update = 0
    #     self.min_return = 1e8

    def reset(self):
        self.train_policy_steps = 0
        pass

    def end_episode(self, step, episode_reward):
        if self.exp_epoch < 0:
            return
        self.episode_num += 1
        if self.use_checkpoint:
            self.maybe_train_and_checkpoint(step, episode_reward)
    
    def run_train_step(self):
        if not self.use_checkpoint or (self.use_checkpoint and self.checkpoint_args.online_train):
            self.train_policy_steps += self.train_step(full_env=self.use_checkpoint)

    def sample(self, states, epoch=-1, step=-1, evaluation=False, train_step=True):
        # if not evaluation and not self.use_checkpoint:
        if not evaluation and train_step:
            self.run_train_step()
            
        # 使用SAC训练完成的Policy
        with torch.no_grad():
            # states = torch.from_numpy(states).to(self.config.device).float()
            states = self.env.obs_preproc(states)
            if self.encoder_enable:
                states = self.agent.model.encoder(states)
            # print('states.shape', states.shape)
            action = self.SAC_agent.select_action(states, eval=evaluation, use_checkpoint=self.use_checkpoint)
            # action = torch.flatten(self.policy_model(states))  # Ensure ndims=1
        # action = action.data.cpu().numpy()
        
        return action
    
    def add_data_step(self, cur_state, action, reward, next_state, done, is_start=False):
        self.env_pool.push(cur_state, action, reward, next_state, done)

        if 'Hand' in self.config.env and self.exp_epoch >= 0 and False:
            
            env_batch_size = 5
            env_state, _, _, _, _ = self.env_pool.sample(int(env_batch_size))

            desired_goal_dim = self.env.desired_goal_dim
            for i in range(env_state.shape[0]):
                state_item = env_state[i]
                desired_goal = state_item[-desired_goal_dim:]
                new_state = np.concatenate((cur_state[:-desired_goal_dim], desired_goal), axis=-1)
                new_next_state = np.concatenate((next_state[:-desired_goal_dim], desired_goal), axis=-1)

                costs = self.env.obs_cost_fn_cost(new_next_state) + self.env.ac_cost_fn_cost(action)
                # costs = costs[None]
                new_reward = -costs

                self.env_pool.push(new_state, action, new_reward, new_next_state, done)

        
        # 获取已知观测范围
        self.obs_max = np.maximum(self.obs_max, next_state)
        self.obs_min = np.minimum(self.obs_min, next_state)

    def train_epoch(self, epoch_reward=0):

        # if self.use_checkpoint:
        #     return
        
        args = self.config.agent.SACPolicy
        train_num = args.epoch_train_num

        self.SAC_agent.model_update()

        test_params = self.config.test_params
        if self.config.test and test_params.policy_fallback:
            
            print('epoch_reward', epoch_reward, 'best_epoch_reward', self.best_epoch_reward)
            if (epoch_reward + test_params.best_policy_margin <= self.best_epoch_reward) and self.exp_epoch > 2:
                # self.SAC_agent = copy.deepcopy(self.best_SAC_agent)
                self.SAC_agent.load_dict(self.best_SAC_agent)
                # 若出现回退，则使用最新的真实数据再训练一定步数
                fallback_train_num = test_params.fallback_train_num
                if fallback_train_num:
                    for o in range(fallback_train_num):
                        self.train_step()
                print('load best agent')
            elif epoch_reward >= self.best_epoch_reward:
                self.best_SAC_agent = self.SAC_agent.get_dict()
                self.best_epoch_reward = epoch_reward
            else:
                self.best_SAC_agent = self.SAC_agent.get_dict()
                pass
            if test_params.model_train_close_aet:
                # self.SAC_agent.automatic_entropy_tuning = False
                self.SAC_agent.set_automatic_entropy_tuning(False)
        
        if self.exp_epoch >= args.epoch_train_min_epoch:
            for i in range(train_num):
                self.rollout_model()
                if args.epoch_num_train_repeat > 0:
                    self.train_step(is_epoch=(i+1))
        if self.config.test and test_params.policy_fallback:
            if test_params.model_train_close_aet:
                self.SAC_agent.set_automatic_entropy_tuning(True)
            # self.SAC_agent.automatic_entropy_tuning = True
        
    def rollout_model(self):
        if self.exp_epoch < 0:
            return 0
        
        self.timesteps_since_rollout = 0
        args = self.config.agent.SACPolicy
        rollout_batch_size = args.rollout_batch_size

        total_step = self.exp_epoch*self.config.experiment.horizon + self.exp_step
        # 生成训练数据
        if total_step <= 0:
            return
        env_pool = self.env_pool
        args = self.config.agent.SACPolicy
        
        epoch_step = self.exp_epoch

        new_rollout_length = self.set_rollout_length(epoch_step)
        if self.rollout_length != new_rollout_length:
            self.rollout_length = new_rollout_length
            self.model_pool = self.resize_model_pool(self.rollout_length, self.model_pool)
        
        state, action, reward, next_state, done = env_pool.sample_all_batch(rollout_batch_size)

        if 'Hand' in self.config.env and False:
            desired_goal_dim = self.env.desired_goal_dim
            desired_goals = state[..., -desired_goal_dim:]
            new_desired_goals = self.shuffle_along_axis(desired_goals, 0)
            new_state = np.concatenate((state[..., :-desired_goal_dim], new_desired_goals), axis=-1)

            state = new_state

        # num_particles = self.config.agent.num_particles
        # num_example = state.shape[0]
        # spare_len = num_example % num_particles
        # new_num_example = num_example - spare_len
        # state = state[:new_num_example]
        # state_reward = state
        pre_nonterm_mask = (np.zeros(state.shape[0]) == 0)

        for i in range(self.rollout_length):
            # TODO: Get a batch of actions
            pre_state = self.env.obs_preproc(state)
            action = self.SAC_agent.select_action(pre_state)
            next_states, _, rewards, terminals, _ = self.predict_env.step(state, action, i, self.rollout_length)
            # s clip
            if args.clamp_state:
                obs_min = self.obs_min - (self.obs_max - self.obs_min)*args.clamp_state_rate
                obs_max = self.obs_max + (self.obs_max - self.obs_min)*args.clamp_state_rate

                # obs_min = self.obs_min
                # obs_max = self.obs_max

                in_bounds_dim = (next_states >= obs_min) * (next_states <= obs_max)
                in_bounds = np.all(in_bounds_dim, axis=-1).reshape(-1)
                in_bounds_num = np.sum(in_bounds)
                print('exp_epoch', self.exp_epoch, 'in_bounds_num', in_bounds_num)

                # out_of_bounds = ~in_bounds
                next_states_clamped = np.clip(next_states, obs_min, obs_max)
                self.model_pool.push_batch([(state[j], action[j], rewards[j], next_states_clamped[j], terminals[j]) for j in range(state.shape[0]) if pre_nonterm_mask[j]])
                # self.model_pool.push_batch([(state[j], action[j], rewards[j], next_states_clamped[j], terminals[j]) for j in range(state.shape[0]) if pre_nonterm_mask[j] and in_bounds[j]])
            else:
                next_states_clamped = next_states
                self.model_pool.push_batch([(state[j], action[j], rewards[j], next_states_clamped[j], terminals[j]) for j in range(state.shape[0]) if pre_nonterm_mask[j]])
            # s clip end
            
            # TODO: Push a batch of samples
            
            # self.model_pool.push_batch([(state[j], action[j], rewards[j], next_states_clamped[j], terminals[j]) for j in range(state.shape[0]) if pre_nonterm_mask[j]])
            nonterm_mask = ~terminals.squeeze(-1)
            if nonterm_mask.sum() == 0:
                break
            state = next_states_clamped
            # state_reward = next_states_clamped
            pre_nonterm_mask[nonterm_mask == False] = False
            if args.clamp_state:
                pre_nonterm_mask[in_bounds == False] = False

    def save(self, num=-1):
        # return
        self.SAC_agent.save(self.config.run_dir, num=num)
    
    # 每一步训练策略
    def train_step(self, is_epoch=False, full_env=False):

        if self.exp_epoch < 0:
            return 0
        self.all_training_steps += 1
        args = self.config.agent.SACPolicy
        if full_env:
            # if self.all_training_steps % 3 != 0:
            #     return 0
            real_ratio = 1.0
            num_train_repeat = 1
        else:
            real_ratio = args.real_ratio
            num_train_repeat = args.num_train_repeat
            self.training_steps += 1
            self.timesteps_since_rollout += 1
            if self.use_checkpoint:
                model_train_freq = self.config.agent.model_train_freq
                if self.timesteps_since_rollout % model_train_freq == 0:
                    self.rollout_model()

        model_pool = self.model_pool
        env_pool = self.env_pool
        agent = self.SAC_agent
        total_step = self.exp_epoch * self.config.experiment.horizon + self.exp_step
        train_step = self.train_policy_steps
        if args.train_every_n_steps > 0 and total_step % args.train_every_n_steps > 0:
            return 0
        
        if len(env_pool) <= args.min_pool_size:
            return 0
        
        max_train_repeat_per_step = max(args.max_train_repeat_per_step, args.num_train_repeat*args.num_train_data_repeat*args.num_train_env_data_repeat)
        if train_step > max_train_repeat_per_step * total_step:
            return 0

        for i in range(num_train_repeat):
            
            env_batch_size = int(args.policy_train_batch_size * real_ratio)
            model_batch_size = args.policy_train_batch_size - env_batch_size

            # print('model_batch_size', model_batch_size, 'env_batch_size', env_batch_size)
            env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

            if 'Hand' in self.config.env and False:
                desired_goal_dim = self.env.desired_goal_dim
                desired_goals = env_state[..., -desired_goal_dim:]
                # new_desired_goals = np.random.shuffle(desired_goals)
                new_desired_goals = self.shuffle_along_axis(desired_goals, 0)

                new_env_state = np.concatenate((env_state[..., :-desired_goal_dim], new_desired_goals), axis=-1)
                new_env_next_state = np.concatenate((env_next_state[..., :-desired_goal_dim], new_desired_goals), axis=-1)

                costs = self.env.obs_cost_fn_cost(new_env_next_state) + self.env.ac_cost_fn_cost(env_action)
                costs = costs[:, None]
                new_env_reward = -costs

                env_state = new_env_state
                env_reward = new_env_reward
                env_next_state = new_env_next_state

            # num_train_env_data_repeat = args.num_train_env_data_repeat
            # print(rewards[0], env_reward[0])

            env_state = self.env.obs_preproc(env_state)
            env_next_state = self.env.obs_preproc(env_next_state)
            # 有模型数据则使用真实数据和模型数据
            for j in range(args.num_train_env_data_repeat):
                if model_batch_size > 0 and len(model_pool) > 0:
                    model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
                    
                    model_state = self.env.obs_preproc(model_state)
                    model_next_state = self.env.obs_preproc(model_next_state)
                    
                    batch_state = np.concatenate((env_state, model_state), axis=0)
                    batch_action = np.concatenate((env_action, model_action), axis=0)
                    batch_reward = np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0)
                    batch_next_state = np.concatenate((env_next_state, model_next_state), axis=0)
                    batch_done = np.concatenate( (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
                else:
                    # 无模型数据使用真实数据
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

                batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
                batch_done = (~batch_done).astype(int)
                # 策略梯度
                # for j in range(args.num_train_data_repeat):
                for k in range(args.num_train_data_repeat):
                    start_time = time.time()
                    if self.encoder_enable:
                        batch_state = self.agent.model.encoder(batch_state)
                        batch_next_state = self.agent.model.encoder(batch_next_state)
                        # batch_action = self.agent.model.encoder_a(batch_action)
                    agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    # print('elapsed_time', elapsed_time)
                    
        return args.num_train_repeat*args.num_train_env_data_repeat*args.num_train_data_repeat

    def set_rollout_length(self, epoch_step):
        args = self.config.agent.SACPolicy
        rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                                / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                                args.rollout_min_length), args.rollout_max_length))
        return int(rollout_length)
    
    def set_real_ratio(self, epoch_step):
        args = {
            'rollout_min_length': 100,
            'rollout_max_epoch': 200,
            'rollout_min_rate': self.config.agent.SACPolicy.real_ratio,
            'rollout_max_rate': self.config.agent.SACPolicy.real_ratio*0.2
        }

        rollout_rate = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                                / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_rate - args.rollout_min_rate),
                                args.rollout_min_rate), args.rollout_max_rate))
        return rollout_rate
    
    def resize_model_pool(self, rollout_length, model_pool = None):
        args = self.config.agent.SACPolicy
        model_train_freq = self.config.agent.model_train_freq
        rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / model_train_freq
        model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
        new_pool_size = args.model_retain_epochs * model_steps_per_epoch
        if args.mpc_rollout and self.exp_epoch >= args.mpc_rollout_min_epoch:
            predict_length = self.config.agent.predict_length
            num_particles = self.config.agent.num_particles
            mpc_rollout_batch_size = args.mpc_rollout_batch_size
            new_pool_size += predict_length * num_particles * mpc_rollout_batch_size * int(args.epoch_length / model_train_freq)
        new_model_pool = ReplayMemory(new_pool_size)

        if model_pool is not None:
            sample_all = model_pool.return_all()
            new_model_pool.push_batch(sample_all)

        return new_model_pool
    
    def shuffle_along_axis(self, a, axis):
        if axis < 0 or axis >= a.ndim:
            raise ValueError("axis out of range")
        idx = np.random.permutation(a.shape[axis])
        indices = [slice(None)] * a.ndim
        indices[axis] = idx
        return a[tuple(indices)]
    
class PredictEnv:
    def __init__(self, env, agent, config):
        self.agent = agent
        self.env = env
        self.config = config
        self.env_name = config.env
        
    def step(self, obs, act, step=-1, rollout_length=-2):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
        with torch.no_grad():
            if self.config.agent.aleatoric == 'dmbpo':
                # add_var
                next_obs, next_obs_reward = self.agent.prediction(obs, act, return_reward_states=True)
                next_obs = next_obs.cpu().numpy()
                next_obs_reward = next_obs_reward.cpu().numpy()
            elif self.config.agent.aleatoric == 'dmbpo_test':
                if step == rollout_length - 1:
                    next_obs = self.agent.prediction(obs, act, add_var=True)
                else:
                    next_obs = self.agent.prediction(obs, act)
                next_obs = next_obs.cpu().numpy()
                next_obs_reward = next_obs
            else:
                next_obs = self.agent.prediction(obs, act)
                next_obs = next_obs.cpu().numpy()
                next_obs_reward = next_obs
        
        costs = self.env.obs_cost_fn_cost(next_obs_reward) + self.env.ac_cost_fn_cost(act)
        costs = costs[:, None]
        rewards = -costs
        terminals = termination_fn(self.env_name, obs, act, next_obs_reward)
        return next_obs, next_obs_reward, rewards, terminals, {}
    
    def _get_logprob(self, x, means, variances):
        k = x.shape[-1]
        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)
        ## [ batch_size ]
        log_prob = np.log(prob)
        stds = np.std(means, 0).mean(-1)
        return log_prob, stds
    
