import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.legal_actions = []
        self.hxs = []
        self.cxs = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.legal_actions[:]
        del self.hxs[:]
        del self.cxs[:]

class PointCloudEmbed(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        B = input.shape[0]
        emb_input = self.embed(input.transpose(2,1))
        return emb_input.view(B, -1)

    def embed(self, x):
        B, D, N = x.shape

        # embedding: BxDxN -> BxFxN
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.conv3(x2)

        # pooling: BxFxN -> BxFx1
        x_pooled = torch.max(x3, 2, keepdim=True)[0]
        return x_pooled # global feature BxF


class Actor(nn.Module):
    def __init__(self, state_dim, lstm_out, action_dim):
        super(Actor, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, lstm_out),
            nn.Tanh()
        )
        self.lstm = nn.LSTMCell(lstm_out, lstm_out)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.fc2 = nn.Linear(lstm_out, action_dim)
    def forward(self, x, hx, cx):
        feature = self.MLP(x)
        hx, cx = self.lstm(feature, (hx, cx))
        feature = hx
        raw_action_probs = self.fc2(feature)
        return raw_action_probs, hx, cx

class ActorCritic(nn.Module):
    def __init__(self, state_dim, lstm_out, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.state_emb = PointCloudEmbed()
        self.softmax = nn.Softmax(dim=-1)
        self.action_dim = action_dim
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64)
            )
        else:
            self.actor = Actor(state_dim, lstm_out, action_dim)
            '''
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64)
            )
            '''
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )



    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state, hx, cx, legal_action, testing=False):

        # embed = self.state_emb(point_state)
        # state = torch.cat((embed, other_state.unsqueeze(0)), dim=1)
        raw_action_probs, hx, cx = self.actor(state, hx, cx)
        action_probs = self.softmax(raw_action_probs - torch.max(raw_action_probs))
        action_probs = action_probs.masked_fill(legal_action==0, value=torch.tensor(0))
        if torch.isnan(action_probs).sum() > 0:
            print("===============NAN DETECTED==============")
            print(raw_action_probs)
        dist = Categorical(action_probs)


        action = dist.sample()
        action_logprob = dist.log_prob(action)

        if testing:
            action_arg = torch.argmax(action_probs - torch.max(action_probs)).detach()
            return action_arg, action_logprob.detach(), hx, cx

        return action.detach(), action_logprob.detach(), hx, cx

    def evaluate(self, state, hx, cx, legal_action, action):

        B = state.shape[0]
        # embed_f = state[:,:-9].view(B,-1, 3)
        # feature_ex = state[:,-9:]
        if self.has_continuous_action_space:
            # embed = self.state_emb(embed_f)
            # state = torch.cat((embed, feature_ex), dim=1)
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            # embed = self.state_emb(embed_f)
            # state = torch.cat((embed, feature_ex), dim=1)
            raw_action_probs, hx, cx = self.actor(state, hx, cx)
            c = torch.max(raw_action_probs, dim=1)[0].reshape(-1,1)
            action_probs = self.softmax(raw_action_probs - c.repeat(1,self.action_dim))
            action_probs = action_probs.masked_fill(legal_action == 0, value=torch.tensor(0))
            if torch.isnan(action_probs).sum() > 0:
                print("===============NAN DETECTED==============")
                print(raw_action_probs)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, lstm_out, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space=False, action_std_init=0.6, episode_info = None):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, lstm_out, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, lstm_out, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, hx, cx, legal_action, testing=False):


        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                # pc_state = state[:-9].view(-1, 3)
                # other_state = state[-9:]
                action, action_logprob, hx, cx = self.policy_old.act(state, hx, cx, legal_action, testing)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.legal_actions.append(legal_action)
            self.buffer.hxs.append(hx)
            self.buffer.cxs.append(cx)


            return action.item(), hx, cx

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_legal_actions = torch.squeeze(torch.stack(self.buffer.legal_actions, dim=0)).detach().to(device)
        old_hxs = torch.squeeze(torch.stack(self.buffer.hxs, dim=0)).detach().to(device)
        old_cxs = torch.squeeze(torch.stack(self.buffer.cxs, dim=0)).detach().to(device)
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_hxs, old_cxs, old_legal_actions, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))




