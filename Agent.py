import torch as T
from networks import ImpalaCNNLargeIQN

class Agent:
    def __init__(self, n_actions, input_dims, device, num_envs, agent_name, total_frames):

        self.net = ImpalaCNNLargeIQN(input_dims[0], n_actions, spectral=True, device=device, noisy=True,
                                     maxpool=True, model_size=3, num_tau=8, maxpool_size=6,
                                     dueling=True, linear_size=512, ncos=64,
                                     arch="impala", layer_norm=False, activation="relu")

        self.net.load_checkpoint("output/BTR_MTPO120M_5.8M.model")

    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor(observation, dtype=T.float).to(self.net.device)
            qvals = self.net.qvals(state, advantages_only=True)
            x = T.argmax(qvals, dim=1).cpu()

            return x
