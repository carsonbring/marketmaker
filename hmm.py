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


def log_domain_matmul(log_A, log_B):
    """
    log_A : m x n
    log_B : n x p
    output : m x p matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
    """
    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    log_A_expanded = torch.reshape(log_A, (m, n, 1))
    log_B_expanded = torch.reshape(log_B, (1, n, p))

    elementwise_sum = log_A_expanded + log_B_expanded
    out = torch.logsumexp(elementwise_sum, dim=1)

    return out


def maxmul(log_A, log_B):
    """
    log_A : m x n
    log_B : n x p
    output : m x p matrix

    Similar to the log domain matrix multiplication,
    this computes out_{i,j} = max_k log_A_{i,k} + log_B_{k,j}
    """
    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    log_A_expanded = torch.stack([log_A] * p, dim=2)
    log_B_expanded = torch.stack([log_B] * m, dim=0)

    elementwise_sum = log_A_expanded + log_B_expanded
    out1, out2 = torch.max(elementwise_sum, dim=1)

    return out1, out2


class TransitionModel(nn.Module):
    def __init__(self, N):
        super(TransitionModel, self).__init__()
        self.N = N
        self.unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(N, N))

    def forward(self, log_alpha):
        log_transition_matrix = func.log_softmax(
            self.unnormalized_transition_matrix, dim=0
        )
        out = log_domain_matmul(
            log_transition_matrix, log_alpha.transpose(0, 1)
        ).transpose(0, 1)
        return out

    def maxmul(self, log_alpha):
        log_transition_matrix = torch.nn.functional.log_softmax(
            self.unnormalized_transition_matrix, dim=0
        )
        out1, out2 = maxmul(log_transition_matrix, log_alpha.transpose(0, 1))
        return out1.transpose(0, 1), out2.transpose(0, 1)


class EmissionModel(nn.Module):
    def __init__(self, N, M):
        super(EmissionModel, self).__init__()
        self.N = N
        self.M = M
        self.unnormalized_emission_matrix = torch.nn.Parameter(torch.randn(N, M))

    def forward(self, x_t):
        log_emission_matrix = func.log_softmax(self.unnormalized_emission_matrix, dim=1)
        out = log_emission_matrix[:, x_t].transpose(0, 1)
        return out


class HMM(nn.Module):
    def __init__(self, num_states, num_observations):
        super(HMM, self).__init__()
        self.num_states = num_states
        self.num_observations = num_observations

        # Raw (unconstrained) parameters:
        # unnormalized state priors
        self.pi = nn.Parameter(torch.zeros(num_states))
        self.transition_model = TransitionModel(num_states)

        self.emission_model = EmissionModel(num_states, num_observations)

    def sample(self, T=10):
        state_priors = nn.functional.softmax(self.pi, dim=0)
        transition_matrix = torch.nn.functional.softmax(
            self.transition_model.unnormalized_transition_matrix, dim=0
        )
        emission_matrix = torch.nn.functional.softmax(
            self.emission_model.unnormalized_emission_matrix, dim=1
        )
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
                z.append(z_t)

        return x, z

    def viterbi(self, x, T):
        """
        x : IntTensor of shape (batch size, T_max)
        T : IntTensor of shape (batch size)
        Find argmax_z log p(x|z) for each (x) in the batch.
        """
        if self.is_cuda:
            x = x.cuda()
            T = T.cuda()
        batch_size = x.shape[0]
        T_max = x.shape[1]
        log_state_priors = func.log_softmax(self.pi, dim=0)
        # Delta has the log-probabilities for each step of each batches for each state
        log_delta = torch.full(
            (batch_size, T_max, self.N), -float("inf"), dtype=torch.float32
        )
        # Psi stores the indices of the previous states that led to the current state with the highest prob
        psi = torch.zeros(batch_size, T_max, self.N).long()
        if self.is_cuda:
            log_delta = log_delta.cuda()
            psi = psi.cuda()
        # Intializing log_delta for time step 0 with emission probabilities and state priors
        log_delta[:, 0, :] = self.emission_model(x[:, 0]) + log_state_priors
        for t in range(1, T_max):
            # Computing the maximum log-probability for transitioning to each state from the previous time step (max_val) and the actual index of the probabilities (argmax_val) which acts as the representation of hidden state
            max_val, argmax_val = self.transitional_model.maxmul(log_delta[:, t - 1, :])
            # Adding that probability of transitioning to the next state to the emission probability of being in said state. Storing the probability for the time step
            # Computing the log-probability of the most probable path that ends in each state at time t
            log_delta[:, t, :] = self.emission_model(x[:, t]) + max_val
            # storing each index of the previous state that maximizes the log-prob for each observed state in psi
            psi[:, t, :] = argmax_val

        # Finding the max log-probability of each sequence at the each time step across all states
        # Shape (batch_size, T_max)
        log_max = log_delta.max(dim=2)[0]

        # retrieving the log-prob at the final valid time step
        best_path_scores = torch.gather(log_max, 1, T.view(-1, 1) - 1)

        z_star = []
        for i in range(0, batch_size):
            # z_star_i contains the index of the state with the highest log-prob at the final time step
            z_star_i = [log_delta[i, T[i] - 1, :].max(dim=0)[1].item()]
            # Going back through the time steps and assembling the most likely hidden states based on the last state probability and the backpointer matrix
            for t in range(T[i] - 1, 0, -1):
                z_t = psi[i, t, z_star_i[0]].item()
                z_star_i.insert(0, z_t)
            z_star.append(z_star_i)
        return z_star, best_path_scores

    def forward(self, x, T):
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

        log_alpha[:, 0, :] = self.emission_model(x[:, 0]) + log_pi
        # Recursively compute alpha for t=1..T-1
        # Alpha is a tensor representing the log of probs up to time t-1 for each state
        for t in range(1, max_time):
            # alpha_next[i] = logsumexp over j of [alpha[j] + A[j, i]] + B[i, obs[t]]
            # This combines transition and emission probabiilities to compute the state probabilties at each time step
            log_alpha[:, t, :] = self.emission_model(x[:, t]) + self.transition_model(
                log_alpha[:, t - 1, :]
            )

        # P(obs_seq) = logsumexp(alpha[T-1, :])
        # Finding log of the probability of an observation sequence by looking summing at the final timestamp
        log_sums = log_alpha.logsumexp(dim=2)
        log_probs = torch.gather(log_sums, 1, T.view(-1, 1) - 1)
        return log_probs

    def backward(self, x, T):
        """
        x (IntTensor): Observed sequences with shape (batch_size, T_max).
        T (IntTensor): Lengths of each sequence with shape (batch_size).

        Returns log_beta (Tensor).
        Backward probabilities with shape (batch_size, T_max, num_states).
        """
        batch_size, T_max = x.shape
        log_beta = torch.full(
            (batch_size, T_max, self.num_states), -float("inf"), dtype=torch.float32
        )
        log_beta[range(batch_size), T - 1, :] = 0.0

        if self.is_cuda:
            log_beta = log_beta.cuda()

        log_transition_matrix = func.log_softmax(
            self.transition_model.unnormalized_transition_matrix, dim=1
        )
        log_emission_matrix = func.log_softmax(
            self.emission_model.unnormalized_emission_matrix, dim=1
        )
        for t in range(T_max - 2, -1, -1):
            # Mask to identify which sequences are still active at time t+1
            mask = (
                (T > t + 1).float().unsqueeze(1).unsqueeze(2)
            )  # Shape: (batch_size, 1, 1)

            log_B_t1 = self.emission_model(x[:, t + 1])
            log_B_t1 = log_B_t1.unsqueeze(1)  # (batch_size, 1, N)
            log_beta_t1 = log_beta[:, t + 1, :].unsqueeze(1)  # (batch_size, 1, N)
            log_A = log_transition_matrix.unsqueeze(0)  # (1, N, N)

            elementwise_sum = log_A + log_B_t1 + log_beta_t1  # (batch_size, N, N)
            log_beta[:, t, :] = torch.logsumexp(elementwise_sum, dim=2) * mask.squeeze(
                1
            ).squeeze(1)  # (batch_size, N)

        return log_beta

    def gamma(self, log_alpha, log_beta, T):
        """
        log_alpha : Tensor of Forward probabilities with shape (batch_size, T_max, num_states)
        log_beta : Tensor of Backward probabilities with shape (batch_size, T_max, num_states)
        T : Tensor with lengths of each batch with shape (batch_size,)

        Returns Gamma: Tensor of Posterior probabilies of states with shape (batch_size, T_max, num_states)

        """
        batch_size, T_max, N = log_alpha.shape

        log_p_x = (
            torch.logsumexp(log_alpha[range(batch_size), T - 1, :], dim=1)
            .unsqueeze(1)
            .unsqueeze(2)
        )  # (batch_size, T_max, N)

        log_gamma = log_alpha + log_beta - log_p_x  # (batch_size, T_max, N)
        gamma = torch.exp(log_gamma)

        return gamma

    def baum_welch(self, X, T, num_iterations=10):
        """
        X (List[List[int]] or Tensor): A batch of observation sequences.
        T (List[int] or Tensor): Lengths of each observation sequence.
        num_iterations (int): Number of EM iterations.

        """
