import torch as T
from networks import ImpalaCNNLargeIQN

class Agent:
    def __init__(self, n_actions, input_dims, device, num_envs, agent_name, total_frames):

        models_dict = {0: "BTR_MTPO120M_28.5M.model", 1: "BTR_MTPO120M_5.8M.model"}

        model_type = 1

        self.net = ImpalaCNNLargeIQN(input_dims[0], n_actions, device=device, model_size=2 + model_type, linear_size=512)
        self.net.load_checkpoint("models/" + models_dict[model_type])

    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor(observation, dtype=T.float).to(self.net.device)
            qvals = self.net.qvals(state, advantages_only=True)
            x = T.argmax(qvals, dim=1).cpu()

            return x
