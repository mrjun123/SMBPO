import mymbrl.agents as agents
import mymbrl.envs as envs
import numpy as np
import torch
import os
import time
import inspect
from mymbrl.utils import LogDict
import random

class MBRLSampleNew:
    def __init__(self, config, writer):
        config.writer = writer
        Env = envs.get_item(config.env, config.env_type)
        # signature = inspect.signature(Env)
        # if 'acs' in signature.parameters:
        if 'sim_dog_dmbpo' in config.env:
            env = Env(config)
            eval_env = Env(config)
        else:
            env = Env()
            eval_env = Env()
        print('env.observation_space.shape', env.observation_space.shape)
        print('env.MODEL_IN', env.MODEL_IN)
        print('env.action_space', env.action_space)
        
        if config.experiment.monitor:
            monitor_dir = os.path.join(config.run_dir, "monitor")
            eval_env.env._max_episode_steps = config.experiment.horizon
            import gymnasium as gym
            eval_env.env = gym.wrappers.RecordVideo(eval_env.env, video_folder=monitor_dir, episode_trigger=lambda x: x % 10 == 0)
            eval_env.init()
            # gymnasium.experimental.wrappers.RecordVideoV0
            # eval_env = gym.wrappers.Monitor(eval_env, monitor_dir, video_callable=lambda episode_id: True, force=True)
        # else:
        # if config.experiment.monitor:
            # from gym.wrappers.monitoring.video_recorder import VideoRecorder
            # eval_env = gym.wrappers.Monitor(eval_env, video_folder=os.path.join(config.run_dir, 'videos'), episode_trigger=lambda x: x % 10 == 0)
        
        self.eval_env = eval_env
        self.env = env
        self.writer = writer
        if hasattr(env, 'set_config'):
            env.set_config(config)
            eval_env.set_config(config)
        
        Agent = agents.get_item(config.agent.name)
        self.agent = Agent(config, env, writer)
        self.config = config
        
        self.env.seed(config.random_seed)
        self.eval_env.seed(config.random_seed)
        self.eval_epoch = -1000
        self.log_writer = LogDict(config.run_dir, 'run_data')

    def run(self):
        random_ntrain_iters = self.config.experiment.random_ntrain_iters
        ntrain_iters = self.config.experiment.ntrain_iters
        # def one_exp(self, epoch = 0, is_random = False, print_step=True):
        # self.agent.controller.reset()
        done = False
        cur_states = self.env.reset()
        pre_states = None
        pre_action = None
        self.agent.reset()
        actions = []
        rewards = []
        states = [cur_states]
        
        step = 0
        train_step = 0
        epoch_reward = 0
        all_step = 0

        # model_train_freq = self.config.agent.model_train_freq
        # random_horizon = self.config.experiment.random_horizon
        # horizon = self.config.experiment.horizon

        # train_epoch_num = horizon // model_train_freq
        # random_train_epoch_num = random_horizon // model_train_freq
        e_num = 0
        miniepoch_reward = 0
        episode_record_epoch = -100
        
        for i in range(random_ntrain_iters + ntrain_iters):
            epoch = i - random_ntrain_iters
            is_random = (epoch < 0)
            # epoch = epoch + 1
            self.agent.set_epoch(epoch)
            horizon = self.config.experiment.horizon
            if is_random:
                horizon = self.config.experiment.random_horizon
            epoch_reward = 0
            episode_reward = 0
            for epoch_step in range(horizon):
                # if self.config.experiment.monitor:
                #     self.env.render()
                self.agent.set_step(epoch_step)
                if hasattr(self.env, 'set_step'):
                    self.env.set_step(epoch_step)
                path_done = (done or step >= horizon)
                
                start_time = time.time()

                # need_train = (not is_random and all_step % self.config.agent.model_train_freq == 0)
                all_step += 1
                need_train = (all_step % self.config.agent.model_train_freq == 0)
                
                train_step += 1
                if not self.config.experiment.random_train:
                    need_train = (not is_random and need_train)
                # need_train = ((epoch_step + 1) % self.config.agent.model_train_freq == 0)
                
                # need_train = (epoch > 0 and all_step % self.config.agent.model_train_freq == 0)
                if path_done or need_train:
                    # 若结束则默认该动作对状态没有变化
                    # states[-1] = cur_states
                    if len(actions) > 0:
                        states, actions = tuple(map(lambda l: np.stack(l, axis=0),
                                                    (states, actions)))
                        self.agent.add_data(states, actions, path_done=path_done)
                    
                    if epoch > 0 and self.config.experiment.evaluation and path_done:
                        self.evaluation(epoch)
                    
                    if need_train:
                        if not self.config.agent.SACPolicy.use_checkpoint:
                            train_loss_log, mse_loss_log, hold_mse_loss_log = self.agent.train(epoch_reward=miniepoch_reward, return_log=True)
                            self.log_writer.log(f'train-loss-epoch-{epoch}', train_loss_log)
                            self.log_writer.log(f'mse-loss-epoch-{epoch}', mse_loss_log)
                            self.log_writer.log(f'hold-mes-loss-epoch-{epoch}', hold_mse_loss_log)
                            miniepoch_reward = 0

                    if path_done:

                        cur_states = self.env.reset()
                        self.agent.reset()
                        if not is_random:
                            if self.config.agent.SACPolicy.use_checkpoint:
                                self.agent.train(epoch_reward=miniepoch_reward)
                                # if train_step >= self.config.agent.model_train_freq:
                                #     self.agent.train(epoch_reward=miniepoch_reward)
                                #     train_step = 0
                                self.agent.controller.end_episode(step, episode_reward)
                            if episode_record_epoch != epoch:
                                episode_record_epoch = epoch
                                self.writer.add_scalar(f'mbrl/episode_rewards', episode_reward, epoch)
                        step = 0
                        episode_reward = 0
                        if self.config.experiment.dog_monitor:
                            self.env.episode_id += 1
                            self.env.close_video_recorder()
                            self.env.start_video_recorder()
                        e_num += 1
                    actions = []
                    states = [cur_states]
                    rewards = []
                
                if is_random and epoch_step % self.config.agent.model_train_freq == 0:
                    miniepoch_reward = 0
                action = None
                if is_random:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.sample(cur_states)
                # print(action)
                end_time = time.time()
                step_time = end_time - start_time
                if epoch_step != 0 and not is_random:
                    self.log_writer.log(f'step-time-epoch-{epoch}', step_time)
                # run_times.append(step_time)
                self.log_writer.log(f'step-time-enum-{e_num}', step_time)
                for i in range(self.config.experiment.step_num):
                    next_state, reward, done, info= self.env.step(action*self.env.action_space.high[0])
                
                epoch_reward += reward
                episode_reward += reward
                miniepoch_reward += reward
                if self.config.experiment.noise > 0:
                    next_state = np.array(next_state) + np.random.uniform(
                        low=-self.config.experiment.noise, high=self.config.experiment.noise, size=next_state.shape
                    )
                # if not is_random:
                #     print("action_step:", epoch_step, "action_reward:", reward)
                
                if self.config.experiment.record_each_epoch:
                    self.writer.add_scalar(f'mbrl/rewards/epoch{epoch}', reward, epoch_step)
                
                step += 1
                path_done = (done or step >= horizon)


                # terminal = path_done
                terminal = done
                # if "TimeLimit.truncated" in info and info["TimeLimit.truncated"]:
                #     terminal = False

                actions.append(action)
                states.append(next_state)
                rewards.append(reward)

                self.agent.controller.add_data_step(cur_states, action, reward, next_state, terminal, is_start=(len(actions) == 1))

                cur_states = next_state

                
                if len(actions) > 1:
                    s = states[-3]
                    a = actions[-2]
                    ns = states[-2]
                    na = actions[-1]
                    nns = states[-1]
                    reward = rewards[-2]
                    is_start = (len(actions) == 2)
                    
                    self.agent.controller.add_two_step_data(s, a, ns, na, nns, reward, done=False, is_start=is_start, is_end=False)
                    if path_done:
                        self.agent.controller.add_two_step_data(ns, na, nns, None, None, rewards[-1], done, is_start=is_start, is_end=True)
                
            # rewards = np.array(rewards)
            self.writer.add_scalar('mbrl/rewards', epoch_reward, epoch)
            # rewards = []
            self.log_writer.save()
        self.log_writer.save()
            
    def evaluation(self, epoch):
        if epoch == self.eval_epoch:
            return
        self.eval_epoch = epoch
        test_step = 0
        sum_reward = 0
        # if 'HandManipulateEgg' in self.config.env:
        #     seeds = [1,10,100,1000,10000,22,33,55,66,44]
        #     seed = random.randint(0, len(seeds)-1)
        #     cur_states = self.eval_env.reset(seed=seed)
        # else:
        cur_states = self.eval_env.reset()

        self.log_writer.log(f'true-states-epoch-{epoch}', cur_states)

        done = False
        actions = []
        states = [cur_states]
        while (not done) and (test_step != self.config.experiment.horizon):
            action = self.agent.sample(cur_states, evaluation=True)
            if self.config.experiment.monitor:
                self.eval_env.render()

            next_state, reward, done, info= self.eval_env.step(action)
            
            cur_states = next_state

            self.log_writer.log(f'true-states-epoch-{epoch}', cur_states)
            self.log_writer.log(f'true-actions-epoch-{epoch}', action)

            sum_reward += reward
            test_step += 1
            actions.append(action)
            states.append(next_state)
        # if self.config.experiment.monitor:
        #     self.eval_env.env.close()
        #     self.eval_env.env.stats_recorder.save_complete()
        #     self.eval_env.env.stats_recorder.done = True
        # self.env.close()
        states, actions = tuple(map(lambda l: np.stack(l, axis=0), (states, actions)))
        # self.agent.add_data_hold(states, actions)
        if epoch % 5 == 0 and self.config.experiment.plt_info:
            self.plt_long_pred(states, actions, epoch)
        if self.config.experiment.save_controller:
            self.agent.controller.save_model()
        
        self.writer.add_scalar('mbrl/evaluation/rewards', sum_reward, epoch)

    def plt_long_pred(self, states, actions, epoch):
        states = np.array(states)
        actions = np.array(actions)
        # states: 26, actions: 25
        max_predict_length = 5
        states_len = states.shape[0]
        self.log_writer.log(f'true-states-epoch-{epoch}', states)
        self.log_writer.log(f'true-actions-epoch-{epoch}', actions)
        
        if states_len < 2:
            return
        # 数据开始位置
        for l in range(states_len):
            start_status = states[l]
            predict_length = max_predict_length
            if states_len - 1 - l < predict_length:
                predict_length = states_len - l - 1
            if predict_length <= 0:
                break

            cur_obs = start_status
            ob_dim = cur_obs.shape[-1]
            cur_obs = cur_obs.reshape(1, ob_dim)
            cur_obs = np.tile(cur_obs, (self.config.agent.num_particles, 1))
            
            obs_log = []
            acs_log = []
            obs_log.append(cur_obs)
            
            for t in range(predict_length):
                cur_acs = actions[l+t]
                with torch.no_grad():
                    next_obs = self.agent.prediction(cur_obs, cur_acs, t)
                cur_obs = next_obs
                obs_log.append(cur_obs.detach().clone().cpu().numpy())
                acs_log.append(cur_acs)
            
            # obs_log = np.array(obs_log)
            # acs_log = np.array(acs_log)
            
            self.log_writer.log(f'states-epoch-{epoch}', obs_log)
            self.log_writer.log(f'actions-epoch-{epoch}', acs_log)
        
