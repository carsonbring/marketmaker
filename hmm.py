import torch
import torch.nn as nn
import torch.nn.functional as func

# Going to need a day just to wrap my head around Baum-Welch because it is very complicated


# Gradient optimization
def gradient_train_hmm(hmm, sequences, num_epochs=50, lr=1e-2):
    """
    sequences: a list or tensor of observation sequences.
    each sequence is shape (T,) with T being the number of time steps
    Each sequence has integer observation indices
    """
    optimizer = torch.optim.Adam(hmm.parameters(), lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_log_prob = 0.0
        for seq in sequences:
            total_log_prob += hmm.log_likelihood(seq)

        loss = -total_log_prob  # negative log-likelihood
        loss.backward()  # pyright:ignore
        optimizer.step()

        print(f"Epoch {epoch}: NLL={loss.item():.3f}")  # pyright:ignore


class DiscreteHMM(nn.Module):
    def __init__(self, num_states, num_observations):
        super().__init__()
        self.num_states = num_states
        self.num_observations = num_observations

        # Raw (unconstrained) parameters:
        self.log_pi = nn.Parameter(torch.zeros(num_states))  # shape: (M,)
        self.log_A = nn.Parameter(torch.zeros(num_states, num_states))  # shape: (M, M)
        self.log_B = nn.Parameter(
            torch.zeros(num_states, num_observations)
        )  # shape: (M, V)

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

    def _forward_log_observations(self, obs_seq):
        """
        obs_seq: shape (T,) with integer observation symbols in [0, num_observations-1].
        Returns log P(obs_seq).
        """
        # T = length of the sequence
        T = obs_seq.shape[0]

        # Convert raw params to normalized log-probs
        # Initial probabilities of starting in each state
        log_pi = func.log_softmax(self.log_pi, dim=0)  # shape: (M,)
        # Transition Prob Matrix
        log_A = func.log_softmax(self.log_A, dim=1)  # shape: (M, M)
        # Emission Prob matrix
        log_B = func.log_softmax(self.log_B, dim=1)  # shape: (M, V)

        # alpha[t, i] = log P(o_1, ..., o_t, z_t = i)
        # Initialize alpha[0]
        # Represents the prob of being in each state at time 0 given the first observation
        # Adding all prior probs with the emission probs for the first observation.
        # Notice how what would have been multiplying probs turns into adding due to log-space
        alpha = (
            log_pi + log_B[:, obs_seq[0]]
        )  # shape: (M,) where m is the number of states

        # Recursively compute alpha for t=1..T-1
        # Alpha is a tensor representing the log of probs up to time t-1 for each state
        #
        for t in range(1, T):
            # alpha_next[i] = logsumexp over j of [alpha[j] + A[j, i]] + B[i, obs[t]]
            # This combines transition and emission probabiilities to compute the state probabilties at each time step
            # I do this by summing over columns (previous states) for each row (current state)
            alpha_next = (
                torch.logsumexp(alpha.unsqueeze(1) + log_A, dim=0)
                + log_B[:, obs_seq[t]]
            )

            alpha = alpha_next

        # P(obs_seq) = logsumexp(alpha[T-1, :])
        # Finding log of the probability of an observation sequence.
        log_prob = torch.logsumexp(alpha, dim=0)
        return log_prob
