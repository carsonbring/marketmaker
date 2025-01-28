import torch
import torch.nn as nn
import torch.nn.functional as func


# Gradient optimization
# def gradient_train_hmm(hmm, sequences, num_epochs=50, lr=1e-2):
#     """
#     sequences: a list or tensor of observation sequences.
#     each sequence is shape (T,) with T being the number of time steps
#     Each sequence has integer observation indices
#     """
#     optimizer = torch.optim.Adam(hmm.parameters(), lr=lr)
#
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         total_log_prob = 0.0
#         for seq in sequences:
#             total_log_prob += hmm.log_likelihood(seq)
#
#         loss = -total_log_prob  # negative log-likelihood
#         loss.backward()  # pyright:ignore
#         optimizer.step()
#
#         print(f"Epoch {epoch}: NLL={loss.item():.3f}")  # pyright:ignore
#
#
class HMM(nn.Module):
    def __init__(self, num_states, num_observations):
        super(HMM, self).__init__()
        self.num_states = num_states
        self.num_observations = num_observations

        # Raw (unconstrained) parameters:
        # unnormalized state priors
        self.pi = nn.Parameter(torch.zeros(num_states))
        self.transition_model = nn.Parameter(torch.zeros(num_states, num_states))

        self.emission_model = nn.Parameter(torch.zeros(num_states, num_observations))

    def sample(self, T=10):
        state_priors = nn.functional.softmax(self.pi, dim=0)
        transition_matrix = torch.nn.functional.softmax(self.transition_model, dim=0)
        emission_matrix = torch.nn.functional.softmax(self.emission_model, dim=1)
        z_t = torch.distributions.categorical.Categorical(state_priors).sample().item()
        z = []
        x = []
        z.append(z_t)
        for t in range(0, T):
            x_t = (
                torch.distributions.categorical.Categorical(emission_matrix[z_t])  # pyright: ignore
                .sample()
                .item()
            )
            x.append(x_t)

            z_t = (
                torch.distributions.categorical.Categorical(transition_matrix[z_t])  # pyright: ignore
                .sample()
                .item()
            )
            if t < T - 1:
                z.append(x_t)

        return x, z

    def forward(self, observations):
        return self.log_likelihood(observations)

    # finding the likelihood of being in a each hidden state at each time step (forward propagation)
    def log_likelihood(self, observations):
        """
        Compute log P(X) = log P(observations) via the forward algorithm in log-space.
        a LongTensor of shape (T,) or (batch_size, T) if batched.
        """

        if observations.dim() == 1:
            return self._forward_log_observations(observations)
        # If batched: sum or average over the batch’s log-likelihood
        else:
            return torch.stack(
                [self._forward_log_observations(seq) for seq in observations]
            ).sum()  # or .mean()

    def _forward_log_observations(self, x, T):
        """
        x : IntTensor of shape (batch size, T_max)
        T : IntTensor of shape (batch_size,)

        Returns log p(x).
        T = length of each example
        """
        if self.is_cuda:
            x = x.cuda()
            T = T.cuda()

        batch_size = x.shape[0]
        max_time = x.shape[1]

        # Convert raw params to normalized log-probs
        # Initial probabilities of starting in each state
        log_pi = func.log_softmax(self.pi, dim=0)  # shape: (M,)
        log_alpha = torch.zeros(batch_size, max_time, self.num_states)
        if self.is_cuda:
            log_alpha = log_alpha.cuda()

        # Initialize alpha[0]
        # Represents the prob of being in each state at time 0 given the first observation
        # Adding all prior probs with the emission probs for the first observation.
        # Notice how what would have been multiplying probs turns into adding due to log-space
        log_alpha[:, 0, :] = self.emission_model[x[:, 0]] + log_pi

        # Recursively compute alpha for t=1..T-1
        # Alpha is a tensor representing the log of probs up to time t-1 for each state
        #
        for t in range(1, T):
            # alpha_next[i] = logsumexp over j of [alpha[j] + A[j, i]] + B[i, obs[t]]
            # This combines transition and emission probabiilities to compute the state probabilties at each time step
            # I do this by summing over columns (previous states) for each row (current state)
            log_alpha[:, t, :] = (
                self.emission_model[x[:, t]]
                + self.transition_model[log_alpha[:, t - 1, :]]
            )

        # P(obs_seq) = logsumexp(alpha[T-1, :])
        # Finding log of the probability of an observation sequence by looking summing at the final timestamp
        log_sums = log_alpha.logsumexp(dim=2)
        log_probs = torch.gather(log_sums, 1, T.view(-1, 1) - 1)
        return log_probs

    # def emission_model_forward(self, x_t):
    #     log_emission_matrix = func.log_softmax(self.emission_matrix, dim=1)
    #     out = log_emission_matrix[:, x_t].transpose(0, 1)
    #     return out
