import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.utils.prune
import torch.nn.functional as F
import torch.optim as optim
from PER import PER
from functools import partial
from Analytic import Analytics
from networks import ImpalaCNNLargeIQN, FactorizedNoisyLinear


class EpsilonGreedy:
    def __init__(self, eps_start, eps_steps, eps_final, action_space):
        self.eps = eps_start
        self.steps = eps_steps
        self.eps_final = eps_final
        self.action_space = action_space

    def update_eps(self):
        self.eps = max(self.eps - (self.eps - self.eps_final) / self.steps, self.eps_final)

    def choose_action(self):
        if np.random.random() > self.eps:
            return None
        else:
            return np.random.choice(self.action_space)


def randomise_action_batch(x, probs, n_actions):
    mask = torch.rand(x.shape) < probs

    # Generate random values to replace the selected elements
    random_values = torch.randint(0, n_actions, x.shape)

    # Apply the mask to replace elements in the tensor with random values
    x[mask] = random_values[mask]

    return x


def choose_eval_action(observation, eval_net, n_actions, device, rng):
    with torch.no_grad():
        state = T.tensor(observation, dtype=T.float).to(device)
        qvals = eval_net.qvals(state, advantages_only=True)
        x = T.argmax(qvals, dim=1).cpu()

        if rng > 0.:
            # Generate a mask with the given probability
            x = randomise_action_batch(x, 0.01, n_actions)

    return x


def create_network(input_dims, n_actions, device, model_size, linear_size):
    ''' Returns a new ImpalaCNNLargeIQN network.'''
    return ImpalaCNNLargeIQN(input_dims[0], n_actions, model_size=model_size, device=device, linear_size=linear_size)

class Agent:
    '''Class for the deep q agent. Creates the network and has functions for: storing memories, selecting actions
    saving and loading models and learning (i.e. updating parameters with a gradient step).
    '''
    def __init__(self, n_actions, input_dims, device, num_envs, agent_name, total_frames, testing=False, batch_size=256,
                 rr=1, lr=1e-4, target_replace=500, discount=0.997, taus=8, model_size=2, linear_size=512, ncos=64, 
                 replay_period=1, analytics=False, framestack=4, imagex=84, imagey=84, arch='impala', per_alpha=0.2, 
                 max_mem_size=1048576, eps_steps=2000000, activation="relu", n=3, munch_alpha=0.9, grad_clip=10):

        self.per_alpha = per_alpha

        self.procgen = True if input_dims[1] == 64 else False
        self.grad_clip = grad_clip

        self.n_actions = n_actions
        self.input_dims = input_dims
        self.device = device
        self.agent_name = agent_name
        self.testing = testing
        self.activation = activation

        self.per_beta = 0.45

        self.replay_ratio = int(rr) if rr > 0.99 else float(rr)
        self.total_frames = total_frames
        self.num_envs = num_envs

        if self.testing:
            self.min_sampling_size = 4000
        else:
            self.min_sampling_size = 1

        self.lr = lr

        self.analytics = analytics
        if self.analytics:
            self.analytic_object = Analytics(agent_name, testing)

        # Number of environment steps per gradient step
        self.replay_period = replay_period
        # Replay ratio however does not take into account parallel envs
        # Every {replay period} steps, take {replay_ratio} grad steps
        self.total_grad_steps = (self.total_frames - self.min_sampling_size) / (self.replay_period / self.replay_ratio)

        self.priority_weight_increase = (1 - self.per_beta) / self.total_grad_steps

        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.n = n

        self.gamma = discount
        self.batch_size = batch_size

        self.model_size = model_size  # Scaling of IMPALA network

        self.ncos = ncos

        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = munch_alpha

        # 1 Million rounded to the nearest power of 2 for tree implementation
        self.max_mem_size = max_mem_size

        self.replace_target_cnt = target_replace

        self.num_tau = taus

        self.loading_checkpoint = False
        
        if self.loading_checkpoint:
            self.min_sampling_size = 300000

        if not self.loading_checkpoint and not self.testing:
            self.eps_start = 1.0
            # divided by 4 is due to frameskip
            self.eps_steps = eps_steps
            self.eps_final = 0.01
        else:
            self.eps_start = 0.00
            self.eps_steps = eps_steps
            self.eps_final = 0.00

        self.epsilon = EpsilonGreedy(self.eps_start, self.eps_steps, self.eps_final, self.action_space)

        self.linear_size = linear_size
        self.arch = arch

        self.framestack = framestack

        # Prioritised experience replay
        self.memory = PER(self.max_mem_size, device, self.n, num_envs, self.gamma, alpha=self.per_alpha,
                          beta=self.per_beta, framestack=self.framestack, rgb=False, imagex=imagex, imagey=imagey)

        self.network_creator_fn = partial(create_network, self.input_dims, self.n_actions, self.device, self.model_size, self.linear_size)

        self.net = self.network_creator_fn()
        self.tgt_net = self.network_creator_fn()

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=0.005 / self.batch_size)  # 0.00015

        self.net.train()

        self.eval_net = None

        for param in self.tgt_net.parameters():
            param.requires_grad = False

        self.env_steps = 0
        self.grad_steps = 0

        self.replay_ratio_cnt = 0

        if self.loading_checkpoint:
            self.load_models("models/BTR_MTPO200M_5.8M.model")

        self.all_grad_mag = 0
        self.tot_churns = 0
        self.cum_churns = 0

    def get_grad_steps(self):
        return self.grad_steps

    @torch.no_grad()
    def reset_noise(self, net):
        for m in net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()


    def choose_action(self, observation):
        # this chooses an action for a batch. Can be used with a batch of 1 if needed though
        with T.no_grad():
            self.reset_noise(self.net)

            # state = T.tensor(observation, dtype=int).to(self.net.device)
            state = T.tensor(observation, dtype=T.float).to(self.net.device)

            qvals = self.net.qvals(state, advantages_only=True)
            x = T.argmax(qvals, dim=1).cpu()

            if self.env_steps < self.min_sampling_size or (self.env_steps < self.total_frames / 2):
                probs = self.epsilon.eps
                x = randomise_action_batch(x, probs, self.n_actions)

            return x

    def store_transition(self, state, action, reward, next_state, done, stream, prio=True):
        self.memory.append(state, action, reward, next_state, done, stream, prio=prio)

        self.epsilon.update_eps()
        self.env_steps += 1

    def replace_target_network(self):
        self.tgt_net.load_state_dict(self.net.state_dict())

    def save_model(self):
        self.net.save_checkpoint(self.agent_name + "_" + str(int((self.env_steps // 250000))) + "M")

    def load_models(self, name):
        self.net.load_checkpoint(name)
        self.tgt_net.load_checkpoint(name)

    def learn(self):
        if self.replay_ratio < 1:
            if self.replay_ratio_cnt == 0:
                self.learn_call()
            self.replay_ratio_cnt = (self.replay_ratio_cnt + 1) % (int(1 / self.replay_ratio))
        else:
            for i in range(self.replay_ratio):
                self.learn_call()

    def learn_call(self):
        if self.env_steps < self.min_sampling_size:
            return

        self.reset_noise(self.tgt_net)

        if self.grad_steps % self.replace_target_cnt == 0:
            self.replace_target_network()

        idxs, states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)

        self.optimizer.zero_grad()

        with torch.no_grad():
            Q_targets_next, _ = self.tgt_net(next_states)

            # (batch, num_tau, actions)
            q_t_n = Q_targets_next.mean(dim=1)

            actions = actions.unsqueeze(1)
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)
            weights = weights.unsqueeze(1)

            # calculate log-pi
            logsum = torch.logsumexp((q_t_n - q_t_n.max(1)[0].unsqueeze(-1)) / self.entropy_tau, 1).unsqueeze(-1)  # logsum trick
            tau_log_pi_next = (q_t_n - q_t_n.max(1)[0].unsqueeze(-1) - self.entropy_tau * logsum).unsqueeze(1)

            pi_target = F.softmax(q_t_n / self.entropy_tau, dim=1).unsqueeze(1)

            Q_target = (self.gamma ** self.n * (pi_target * (Q_targets_next - tau_log_pi_next) * (~dones.unsqueeze(-1))).sum(2)).unsqueeze(1)

            q_k_target = self.net.qvals(states)
            v_k_target = q_k_target.max(1)[0].unsqueeze(-1)
            tau_log_pik = q_k_target - v_k_target - self.entropy_tau * torch.logsumexp((q_k_target - v_k_target) / self.entropy_tau, 1).unsqueeze(-1)

            munchausen_addon = tau_log_pik.gather(1, actions)

            # calc munchausen reward:
            munchausen_reward = (rewards + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)

            # Compute Q targets for current states
            Q_targets = munchausen_reward + Q_target

        # Get expected Q values from local model
        q_k, taus = self.net(states)
        Q_expected = q_k.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.num_tau, 1))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        loss_v = torch.abs(td_error).sum(dim=1).mean(dim=1).data

        huber_l = calculate_huber_loss(td_error, 1.0, self.num_tau)
        quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)  # , keepdim=True if per weights get multipl

        loss = loss * weights.to(self.net.device)

        loss = loss.mean()

        self.memory.update_priorities(idxs, loss_v.cpu().detach().numpy())

        if self.analytics:
            with torch.no_grad():
                self.analytic_object.add_loss(loss.cpu().detach())

        loss.backward()

        if self.analytics:
            with torch.no_grad():
                grad_magnitude = self.compute_gradient_magnitude()
                self.analytic_object.add_grad_mag(grad_magnitude.cpu().detach().item())

                self.all_grad_mag += grad_magnitude.cpu().detach().item()

                qvals = q_k_target.mean(dim=1)

                self.analytic_object.add_qvals(qvals.cpu().detach())

                if self.grad_steps % 1 == 0:
                    _, churn_states, _, _, _, _, _ = self.memory.sample(self.batch_size)

                    churn_qvals_before = self.net.qvals(churn_states)
                    churn_actions_before = T.argmax(churn_qvals_before, dim=1).cpu()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.analytics and self.grad_steps % 1 == 0:
            with torch.no_grad():
                churn_qvals_after = self.net.qvals(churn_states)
                churn_actions_after = T.argmax(churn_qvals_after, dim=1).cpu()

                difference = torch.mean(churn_qvals_after - churn_qvals_before, dim=0)
                self.analytic_object.add_churn_dif(difference.cpu().detach())

                difference_actions = torch.sum((churn_actions_before != churn_actions_after).int(), dim=0)
                policy_churn = difference_actions / self.batch_size

                self.analytic_object.add_churn(policy_churn.cpu().detach().item())
                self.tot_churns += 1
                self.cum_churns += policy_churn.cpu().detach().item()

                # print(f"Churns: {self.cum_churns / self.tot_churns}")

                self.analytic_object.add_churn_actions(actions.cpu().detach())

        self.grad_steps += 1
        if self.grad_steps % 10000 == 0:
            print("Completed " + str(self.grad_steps) + " gradient steps")


    @torch.no_grad()
    def calculate_parameter_norms(self, norm_type=2):
        self.net.load_state_dict(self.net.state_dict())
        # Dictionary to store the norms
        norms = {}
        # Iterate through all named parameters
        for name, param in self.net.named_parameters():
            if 'weight' in name:
                # Calculate the norm of the parameter
                norm = torch.norm(param, p=norm_type).item()  # .item() converts a one-element tensor to a scalar
                # Store the norm in the dictionary
                norms[name] = norm

        norms_tot = 0
        count = 0
        for key, value in norms.items():
            count += 1
            norms_tot += value

        norms_tot /= count

        return norms_tot

    def compute_gradient_magnitude(self):
        # Calculate the magnitude of the average gradient
        total_grad = 0.0
        total_params = 0

        for param in self.net.parameters():
            if param.grad is not None:
                param_grad = param.grad.data
                total_grad += torch.sum(torch.abs(param_grad))
                total_params += param_grad.numel()

        average_grad_magnitude = total_grad / total_params
        return average_grad_magnitude

def calculate_huber_loss(td_errors, k=1.0, taus=8):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], taus, taus), "huber loss has wrong shape"
    return loss
