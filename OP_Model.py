# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils
import warnings
import math
from sensor_budget_config import get_sensor_budget

warnings.filterwarnings("ignore")

# ==========================================
# 共享基础网络模块：图卷积层
# ==========================================
class GraphConvolution(nn.Module): 
    def __init__(self, f_in, f_out, use_bias=True, activation=utils.mish):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.use_bias = use_bias
        self.activation = activation
        self.weight = nn.Parameter(torch.FloatTensor(f_in, f_out))
        self.bias = nn.Parameter(torch.FloatTensor(f_out)) if use_bias else None
        self.initialize_weight()
        # 当输出和输出维度不一致的时候，残差连接可能会有维度mismatch的问题，这里就是为了解决这个问题
        if f_in != f_out: 
            self.res_proj = nn.Linear(f_in, f_out, bias=False)
        else:
            self.res_proj = None
        self.norm = nn.LayerNorm(f_out)

    def initialize_weight(self): 
        if self.activation is None: 
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, nonlinearity='leaky_relu')

        if self.use_bias: 
            nn.init.zeros_(self.bias)

    def forward(self, input, Gs_matrix_diag): 
        # input为batch_size*n_nodes*n_features_aug的特征tensor
        support = torch.matmul(input, self.weight) 
        output = torch.matmul(Gs_matrix_diag, support) 
        if self.use_bias: 
            output.add_(self.bias)

        if self.res_proj is not None:
            output = output + self.res_proj(input)
        else:
            output = output + input

        if self.norm is not None:
            output = self.norm(output)

        if self.activation is not None: 
            output = self.activation(output)
        
        return output

# ==========================================
# DQN 模型相关 (Value-Based)
# ==========================================
class GCN_Qnet(nn.Module): #定义 Q 网络
    def __init__(self, f_in, dens_obs, hidden=[16], dropout_p=0.3, training=True):
        super().__init__()
        self.dens_obs = dens_obs
        self.sensor_budget_override = None
        layers = []

        # 根据参数构建多层网络
        for input_dim, output_dim in zip([f_in] + hidden[:-1], hidden):
            layers += [GraphConvolution(input_dim, output_dim)]  

        self.layers = nn.Sequential(*layers)
        self.dropout_p = dropout_p
        self.fc1 = nn.Linear(hidden[-1], 1, bias=True)  
        self.training = training

    def forward(self, states, b_features, Gs_matrix_diag): 
        n_nodes = Gs_matrix_diag.shape[1]
        sensor_budget = get_sensor_budget(n_nodes, override=self.sensor_budget_override)
        remain_steps = (sensor_budget - torch.sum(states, dim=1)) / sensor_budget
        remain_steps = remain_steps.view(-1, 1, 1).repeat(1, n_nodes, 1) 
        states = states.view(-1, n_nodes, 1)
        
        features_aug = torch.cat((b_features, states, remain_steps), 2) 

        for i, layer in enumerate(self.layers):
            features_aug = layer(features_aug, Gs_matrix_diag)
            
        F.dropout(features_aug, self.dropout_p, training=self.training, inplace=True) 
        outputs = self.fc1(features_aug).squeeze(2)

        return outputs 

class GCN_DoubleDQN(nn.Module):
    def __init__(self, n_features_aug, dens_obs, hidden, learning_rate, lr_gamma, gamma, epsilon, target_update, device):
        super(GCN_DoubleDQN, self).__init__()
        self.q_net = GCN_Qnet(n_features_aug, dens_obs, hidden).to(device) #Q网络
        self.target_q_net = GCN_Qnet(n_features_aug, dens_obs, hidden).to(device) #目标网络
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate) 
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_gamma) 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.target_update = target_update 
        self.count = 0 
        self.device = device
    
    def forward(self, state, features, G_matrix_diag):
        return self.q_net(state, features, G_matrix_diag)
    
    def take_action(self, state, features, G_matrix_diag, i_episode, G_index): 
        zero = utils.getAllIndex(state, 0) 

        epsilon_start = 1.0
        epsilon_min   = 0.01
        delta = 0.01           
        k = -math.log(delta) / self.num_episodes
        epsilon = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-k * i_episode)

        if np.random.random() < epsilon: 
            action = np.random.choice(zero)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device) 
            action_max = self.q_net(state, features, G_matrix_diag)[:,zero].argmax().item()
            action = zero[action_max]
            
        return action

    def max_q_value(self, state, features, G_matrix_diag): 
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state, features, G_matrix_diag).max().item() 

    def predict(self, state, features, G_matrix_diag):
        self.n_features = features.shape[1]  
        features = torch.tensor(features, dtype=torch.float).view(1, -1, self.n_features).to(self.device)
        n_nodes = features.shape[1]
        G_matrix_diag = torch.tensor(G_matrix_diag, dtype=torch.float).view(1, n_nodes, -1).to(self.device)
        zero = utils.getAllIndex(state, 0)  
        state = torch.tensor([state], dtype=torch.float).to(self.device)  

        self.q_net.training = False 
        
        action_max = self.q_net(state, features, G_matrix_diag)[:, zero].argmax().item()
        action = zero[action_max]
        
        return action

    def predict_topk(self, state, features, G_matrix_diag, k=1):
        self.n_features = features.shape[1]  
        features = torch.tensor(features, dtype=torch.float).view(1, -1, self.n_features).to(self.device)
        n_nodes = features.shape[1]
        G_matrix_diag = torch.tensor(G_matrix_diag, dtype=torch.float).view(1, n_nodes, -1).to(self.device)

        zero = utils.getAllIndex(state, 0)  
        k = min(k, len(zero))  
        state = torch.tensor([state], dtype=torch.float).to(self.device)  

        self.q_net.training = False

        topk_value, topk_idx = torch.topk(self.q_net(state, features, G_matrix_diag)[:, zero], k=k, dim=1)  
        action_list = [zero[i] for i in topk_idx[0].tolist()]

        return action_list

    def search(self, env, features, G_matrix_diag, beta=4, gamma=2, max_steps=200):
        import copy
        device = self.device
        n_nodes = features.shape[1]

        state = env.reset()  
        beam = [ (state, [], 0.0, False) ]

        for step in range(max_steps):
            if all(done for _,_,_,done in beam):
                break

            # --- Expansion ---
            expansion_candidates = []
            for b_state, b_actions, b_reward, b_done in beam:
                if b_done:
                    expansion_candidates.append((b_state, b_actions, b_reward, b_done))
                    continue

                valid_actions = utils.getAllIndex(b_state, 0)
                if not valid_actions:
                    expansion_candidates.append((b_state, b_actions, b_reward, True))
                    continue

                state_tensor = torch.tensor([b_state], dtype=torch.float, device=device)
                q_values = self.q_net(state_tensor, features, G_matrix_diag)[0]  

                q_valid = q_values[valid_actions]
                top_indices = torch.topk(q_valid, min(gamma, len(valid_actions))).indices.cpu().numpy()
                top_actions = [valid_actions[i] for i in top_indices]

                for a in top_actions:
                    env_copy = copy.deepcopy(env)
                    next_state, reward, done, _ = env_copy.step(b_state, a)
                    expansion_candidates.append( (next_state, b_actions + [a], b_reward + reward, done) )

            # --- Simulation ---
            sim_results = []
            for s_state, s_actions, s_reward, s_done in expansion_candidates:
                if s_done:
                    sim_results.append( (s_state, s_actions, s_reward, s_done) )
                    continue

                env_sim = copy.deepcopy(env)
                cur_state, total_reward, done = s_state, s_reward, s_done
                action_hist = list(s_actions)

                while not done:
                    valid_actions = utils.getAllIndex(cur_state, 0)
                    if not valid_actions:
                        break
                    state_tensor = torch.tensor([cur_state], dtype=torch.float, device=device)
                    q_values = self.q_net(state_tensor, features, G_matrix_diag)[0]
                    max_idx = q_values[valid_actions].argmax().item()
                    action = valid_actions[max_idx]
                    cur_state, reward, done, _ = env_sim.step(cur_state, action)
                    action_hist.append(action)
                    total_reward += reward

                sim_results.append( (cur_state, action_hist, total_reward, done) )

            # --- Pruning ---
            seen = set()
            unique_results = []
            for s in sim_results:
                actions_tuple = tuple(s[1])
                if actions_tuple not in seen:
                    unique_results.append(s)
                    seen.add(actions_tuple)
            unique_results.sort(key=lambda x: x[2], reverse=True)
            beam = unique_results[:beta]

        best_state, best_action_seq, best_reward, _ = max(beam, key=lambda x: x[2])
        return best_action_seq, best_reward

    def update(self, transitions, idxs, is_weights):
        states, actions, rewards, next_states, dones, b_index = zip(*transitions)
        states = torch.tensor(states, dtype=torch.float, device=self.device)  
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).view(-1, 1)  
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).view(-1, 1)  
        next_states = torch.tensor(next_states, dtype=torch.float, device=self.device)  
        dones = torch.tensor(dones, dtype=torch.float, device=self.device).view(-1, 1)  
        is_weights = torch.tensor(is_weights, dtype=torch.float, device=self.device).view(-1, 1)
        
        b_features = self.train_data['features_tensor'][b_index,:,:]
        Gs_matrix_diag = self.train_data['G_matrix_diag_tensor'][b_index,:,:]

        q_values = self.q_net(states, b_features, Gs_matrix_diag).gather(1, actions)  
        with torch.no_grad(): 
            max_action = self.q_net(next_states, b_features, Gs_matrix_diag).max(1)[1].view(-1, 1)  
            max_next_q_values = self.target_q_net(next_states, b_features, Gs_matrix_diag).gather(1, max_action)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  

        td_errors = q_values - q_targets
        dqn_loss = (is_weights * (td_errors.pow(2))).mean() 
        
        self.optimizer.zero_grad() 
        dqn_loss.backward() 
        self.optimizer.step()
        self.scheduler.step() 

        if self.count % self.target_update == 0: 
            self.target_q_net.load_state_dict(self.q_net.state_dict()) 
        self.count += 1

        for idx, error in zip(idxs, td_errors.detach().cpu().numpy().squeeze()):
            self.replay_buffer.update(idx, error)

        return dqn_loss.item() 

    def train(self, env, train_data, num_episodes, replay_buffer_params, minimal_size, batch_size, G_num, num_sample, device):
        return_list_all, max_q_value_list_all, buffer_fill_all, loss_list_all = [],[],[],[]
        train_data['G_matrix_diag_tensor'] = torch.cat([torch.tensor(mat_diag, dtype=torch.float).unsqueeze(0)
                                                    for mat_diag in train_data['G_matrix_diag']], 0).to(device)
        train_data['features_tensor'] = torch.cat([torch.tensor(mat_diag, dtype=torch.float).unsqueeze(0)
                                                        for mat_diag in train_data['features']], 0).to(device)
        
        self.G_num = G_num 
        self.num_sample = num_sample 
        self.num_episodes = num_episodes 
        self.dens_obs = env.dens_obs 
        self.q_net.dens_obs = self.dens_obs 
        self.target_q_net.dens_obs = self.dens_obs 
        self.n_features = train_data['features_tensor'].shape[2]
        self.train_data = train_data
        
        if replay_buffer_params['replay_buffer_type'] == 'PriorityReplayBuffer': 
            self.replay_buffer = utils.PriorityReplayBuffer(replay_buffer_params['buffer_size'])
        else: 
            self.replay_buffer = utils.ReplayBuffer(replay_buffer_params['buffer_size']) 

        i_episode = 0
        with tqdm(total=int(self.G_num * self.num_sample * self.num_episodes), desc="Iteration %d" % i_episode) as pbar:
            for i_episode in range(int(self.G_num * self.num_sample * self.num_episodes)): 
                G_index = np.random.randint(0, self.G_num)
                G, G_matrix_diag, G_distance, G_distance_delay, features = self.train_data['G'][G_index], self.train_data['G_matrix_diag'][G_index], \
                        self.train_data['G_distance'][G_index], self.train_data['G_distance_delay'][G_index],  self.train_data['features'][G_index]
                env.get_graph(G, G_matrix_diag, G_distance, G_distance_delay)

                G_matrix_diag = torch.tensor(G_matrix_diag, dtype=torch.float).view(1,len(G.nodes),-1).to(device)
                features = torch.tensor(features, dtype=torch.float).view(1,-1,self.n_features).to(device)

                max_q_value = 0
                episode_return = 0
                
                state = env.reset() 
                done = 0.0
                while not done:
                    action = self.take_action(state, features, G_matrix_diag, i_episode, G_index) 
                    max_q_value = self.max_q_value(state, features, G_matrix_diag) * 1.000 + max_q_value * 0.000 
                    max_q_value_list_all.append(max_q_value) 
                    next_state, reward, done, shaping = env.step(state, action) 
                    q_value = self.q_net(torch.tensor([state], dtype=torch.float).to(device), features, G_matrix_diag)[:, action].item() 
                    max_next_q_value = self.q_net(torch.tensor([next_state], dtype=torch.float).to(device), features, G_matrix_diag).max().item() 
                    q_target = reward + self.gamma * max_next_q_value * (1 - done)  
                    error = abs(q_target - q_value)  

                    self.replay_buffer.add(error, (state, action, reward, next_state, done, G_index))
                    buffer_fill_all.append(self.replay_buffer.size()) 
                    state = next_state
                    episode_return += reward 
                    
                    if self.replay_buffer.size() > minimal_size:
                        transitions, idxs, is_weights = self.replay_buffer.sample(batch_size)
                        loss = self.update(transitions, idxs, is_weights)
                        loss_list_all.append(loss)

                return_list_all.append(episode_return)
                pbar.update(1)
    
        return (return_list_all, max_q_value_list_all, buffer_fill_all, loss_list_all)

# ==========================================
# PPO/Actor-Critic 模型相关 (Policy-Based)
# ==========================================
class GCN_ActorCritic(nn.Module):
    def __init__(self, f_in, dens_obs, hidden=[16], dropout_p=0.3, training=True):
        super().__init__()
        self.dens_obs = dens_obs
        self.sensor_budget_override = None
        layers = []

        for input_dim, output_dim in zip([f_in] + hidden[:-1], hidden):
            layers += [GraphConvolution(input_dim, output_dim)]

        self.layers = nn.Sequential(*layers)
        self.dropout_p = dropout_p
        
        # Actor Head: 输出每个节点的偏好值 (Logits)
        self.actor_head = nn.Linear(hidden[-1], 1, bias=True)
        # Critic Head: 输出当前整张图的状态价值 (Value)
        self.critic_head = nn.Linear(hidden[-1], 1, bias=True)
        
        self.training = training

    def forward(self, states, b_features, Gs_matrix_diag):
        n_nodes = Gs_matrix_diag.shape[1]
        sensor_budget = get_sensor_budget(n_nodes, override=self.sensor_budget_override)
        remain_steps = (sensor_budget - torch.sum(states, dim=1)) / sensor_budget
        remain_steps = remain_steps.view(-1, 1, 1).repeat(1, n_nodes, 1)
        states_view = states.view(-1, n_nodes, 1)
        
        features_aug = torch.cat((b_features, states_view, remain_steps), 2)

        for layer in self.layers:
            features_aug = layer(features_aug, Gs_matrix_diag)
            
        F.dropout(features_aug, self.dropout_p, training=self.training, inplace=True)
        
        # 计算 Actor 输出 (batch_size, n_nodes)
        logits = self.actor_head(features_aug).squeeze(-1)
        
        # 计算 Critic 输出: 先对图节点特征进行全局平均池化，再评估价值
        graph_pooled_features = torch.mean(features_aug, dim=1)  # (batch_size, hidden[-1])
        value = self.critic_head(graph_pooled_features)           # (batch_size, 1)

        return logits, value

class PPOAgent(nn.Module):
    def __init__(self, n_features_aug, dens_obs, hidden, learning_rate, gamma, gae_lambda, clip_param, ppo_epochs, entropy_coef, entropy_decay=0.9995, min_entropy_coef=0.01, dropout_p=0.0, device='cpu'):
        super(PPOAgent, self).__init__()
        self.ac_net = GCN_ActorCritic(
            n_features_aug, dens_obs, hidden, dropout_p=dropout_p
        ).to(device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.min_entropy_coef = min_entropy_coef
        self.device = device
    
    def take_action(self, state, features, G_matrix_diag):
        """PPO 专用的动作采样函数（带有 Action Masking）"""
        self.ac_net.eval()
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)
            features = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(self.device)
            G_matrix_diag = torch.tensor(G_matrix_diag, dtype=torch.float).unsqueeze(0).to(self.device)
            
            logits, value = self.ac_net(state_tensor, features, G_matrix_diag)
            
            # Action Masking: 已经选过的节点 (state == 1) 的 logit 设为极小值
            mask = state_tensor.bool()
            logits = logits.masked_fill(mask, -1e9)
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action).item(), value.item()

    def predict_topk(self, state, features, G_matrix_diag, k=1):
        """用于 SGBS 搜索及 One_shot 的推理函数，使用 actor logits 排序"""
        self.ac_net.eval()
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)
            features = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(self.device)
            G_matrix_diag = torch.tensor(G_matrix_diag, dtype=torch.float).unsqueeze(0).to(self.device)
            
            logits, _ = self.ac_net(state_tensor, features, G_matrix_diag)
            
            mask = state_tensor.bool()
            logits = logits.masked_fill(mask, -1e9)
            
            topk_idx = torch.topk(logits, k=k, dim=1).indices[0].tolist()
            return topk_idx

    def predict_topk_with_log_probs(self, state, features, G_matrix_diag, k=1):
        """Return topology-policy candidates and conditional log-probabilities.

        SGBS uses this method at inference time, so the search depends only on
        graph features, adjacency and the current placement state.  The
        realized edge-delay matrix is deliberately not an input.
        """
        self.ac_net.eval()
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.float, device=self.device)
            feature_tensor = torch.tensor(
                features, dtype=torch.float, device=self.device
            ).unsqueeze(0)
            adjacency_tensor = torch.tensor(
                G_matrix_diag, dtype=torch.float, device=self.device
            ).unsqueeze(0)

            logits, _ = self.ac_net(state_tensor, feature_tensor, adjacency_tensor)
            mask = state_tensor.bool()
            logits = logits.masked_fill(mask, -1e9)

            available = int((~mask).sum().item())
            k = min(int(k), available)
            if k <= 0:
                return []

            distribution = Categorical(logits=logits)
            topk_actions = torch.topk(logits, k=k, dim=1).indices[0]
            log_probs = distribution.log_prob(topk_actions)
            return list(zip(topk_actions.tolist(), log_probs.tolist()))

    def update(self, transition_dict, features_tensor, G_matrix_diag_tensor):
        """PPO 网络更新"""
        self.ac_net.train()
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = transition_dict['rewards']
        old_log_probs = torch.tensor(transition_dict['log_probs'], dtype=torch.float).view(-1, 1).to(self.device)
        values = transition_dict['values']
        dones = transition_dict['dones']
        b_indices = transition_dict['b_indices']
        
        b_features = features_tensor[b_indices]
        b_G_matrix = G_matrix_diag_tensor[b_indices]

        # 1. 计算 GAE
        returns = []
        advantages = []
        gae = 0
        with torch.no_grad():
            next_state_tensor = torch.tensor([transition_dict['next_states'][-1]], dtype=torch.float).to(self.device)
            _, next_value = self.ac_net(next_state_tensor, b_features[-1].unsqueeze(0), b_G_matrix[-1].unsqueeze(0))
            next_value = next_value.item()

        values = values + [next_value]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        advantages = torch.tensor(advantages, dtype=torch.float).view(-1, 1).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float).view(-1, 1).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. PPO 多轮迭代更新
        loss_total = 0
        for _ in range(self.ppo_epochs):
            logits, current_values = self.ac_net(states, b_features, b_G_matrix)
            
            mask = states.bool()
            logits = logits.masked_fill(mask, -1e9)
            dist = Categorical(logits=logits)
            
            new_log_probs = dist.log_prob(actions.squeeze()).view(-1, 1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = F.mse_loss(current_values, returns)
            
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac_net.parameters(), max_norm=0.5) 
            self.optimizer.step()
            
            loss_total += loss.item()

        self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)
            
        return loss_total / self.ppo_epochs
