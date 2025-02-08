import math
import torch
import torch.nn as nn
import torch.nn.functional as func


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

    log_A_expanded = log_A.unsqueeze(dim=2)
    log_B_expanded = log_B.unsqueeze(dim=0)

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
            self.unnormalized_transition_matrix, dim=1
        )
        out = log_domain_matmul(
            log_transition_matrix, log_alpha.transpose(0, 1)
        ).transpose(0, 1)
        return out

    def maxmul(self, log_alpha):
        log_transition_matrix = torch.nn.functional.log_softmax(
            self.unnormalized_transition_matrix, dim=1
        )
        out1, out2 = maxmul(log_transition_matrix, log_alpha.transpose(0, 1))
        return out1.transpose(0, 1), out2.transpose(0, 1)


class EmissionModel(nn.Module):
    def __init__(self, N, D):
        super(EmissionModel, self).__init__()
        self.N = N
        self.D = D
        self.means = torch.nn.Parameter(torch.randn(N, D))
        self.cholesky_factors = torch.nn.Parameter(torch.randn(N, D, D))

    def forward(self, x_t):
        """
        x_t: Tensor of shape (batch_size, D)
        Returns: Tensor of shape (batch_size, N) where each entry
                 [i, j] is the log-probability of observation i under state j.
        """
        # print("x_t device:", x_t.device)
        # print("means device:", self.means.device)
        x_t = x_t.to(self.means.device)
        batch_size = x_t.shape[0]
        log_c = self.D * math.log(2 * math.pi)

        L_unconstrained = torch.tril(self.cholesky_factors)
        diag_indices = torch.arange(self.D, device=L_unconstrained.device)

        diagonal = L_unconstrained[:, diag_indices, diag_indices]
        diagonal_exp = torch.exp(diagonal)
        L = L_unconstrained.clone()
        L[:, diag_indices, diag_indices] = diagonal_exp
        diag_L = torch.diagonal(L, offset=0, dim1=-2, dim2=-1)

        log_diag = torch.log(diag_L)
        log_det_Sigma = 2 * torch.sum(log_diag, dim=-1)  # (N,)

        # Using Cholesky Decomposition for efficient computation of quadratic
        delta = x_t.unsqueeze(1) - self.means.unsqueeze(0)
        delta_reshaped = delta.reshape(-1, self.D).unsqueeze(-1)
        L_expanded = (
            L.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(-1, self.D, self.D)
        )
        # print("x_t range:", x_t.min().item(), x_t.max().item())
        # print("means range:", self.means.min().item(), self.means.max().item())
        # print("delta range:", delta.min().item(), delta.max().item())

        y_reshaped = torch.linalg.solve_triangular(
            L_expanded, delta_reshaped, upper=False
        )
        y = y_reshaped.squeeze(-1).reshape(batch_size, self.N, self.D)
        quad = torch.sum(y**2, dim=-1)  # (batch_size, N)

        # Gaussian log-likelihood
        log_p = -0.5 * (log_c + log_det_Sigma.unsqueeze(0) + quad)
        return log_p


class GaussianHMM(nn.Module):
    def __init__(self, num_states, num_dimensions):
        super(GaussianHMM, self).__init__()
        self.num_states = num_states
        self.num_dimensions = num_dimensions

        # Raw (unconstrained) parameters:
        # unnormalized state priors
        self.pi = nn.Parameter(torch.zeros(num_states))
        self.transition_model = TransitionModel(num_states)

        self.emission_model = EmissionModel(num_states, num_dimensions)

    def sample(self, T=10):
        state_priors = nn.functional.softmax(self.pi, dim=0)
        transition_matrix = torch.nn.functional.softmax(
            self.transition_model.unnormalized_transition_matrix, dim=1
        )
        z_t = torch.distributions.categorical.Categorical(state_priors).sample().item()
        z = []
        x = []
        z.append(z_t)
        for t in range(0, T):
            mu = self.emission_model.means[int(z_t)]
            L = torch.tril(self.emission_model.cholesky_factors[int(z_t)])
            Sigma = L @ L.transpose(-2, -1)
            dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=Sigma)

            x_t = dist.sample().detach().cpu().numpy()
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
        x : IntTensor of shape (batch size, T_max, D)
        T : IntTensor of shape (batch size)
        Find argmax_z log p(x|z) for each (x) in the batch.
        """
        if next(self.parameters()).is_cuda:
            x = x.cuda()
            T = T.cuda()
        batch_size = x.shape[0]
        T_max = x.shape[1]
        log_state_priors = func.log_softmax(self.pi, dim=0)
        # Delta has the log-probabilities for each step of each batches for each state
        log_delta = torch.full(
            (batch_size, T_max, self.num_states), -float("inf"), dtype=torch.float32
        )
        # Psi stores the indices of the previous states that led to the current state with the highest prob
        psi = torch.zeros(batch_size, T_max, self.num_states).long()
        if next(self.parameters()).is_cuda:
            log_delta = log_delta.cuda()
            psi = psi.cuda()
        # Intializing log_delta for time step 0 with emission probabilities and state priors
        log_delta[:, 0, :] = self.emission_model(x[:, 0]) + log_state_priors
        for t in range(1, T_max):
            # Computing the maximum log-probability for transitioning to each state from the previous time step (max_val) and the actual index of the probabilities (argmax_val) which acts as the representation of hidden state
            max_val, argmax_val = self.transition_model.maxmul(log_delta[:, t - 1, :])
            # Adding that probability of transitioning to the next state to the emission probability of being in said state. Storing the probability for the time step
            # Computing the log-probability of the most probable path that ends in each state at time t
            log_delta[:, t, :] = self.emission_model(x[:, t, :]) + max_val
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

    def log_likelihood(self, log_alpha, T):
        # Assuming T is a tensor of lengths, gather the log_alpha at the final valid time for each sequence.
        batch_indices = torch.arange(log_alpha.shape[0])
        final_log_prob = log_alpha[batch_indices, T - 1, :].logsumexp(dim=1)
        return final_log_prob

    def forward(self, x, T):
        """
        x : IntTensor of shape (batch size, T_max, D)
        T : IntTensor of shape (batch_size,)

        Returns log p(x).
        T = length of each example
        """
        if next(self.parameters()).is_cuda:
            x = x.cuda()
            T = T.cuda()

        batch_size = x.shape[0]
        max_time = x.shape[1]

        # Convert raw params to normalized log-probs
        # Initial probabilities of starting in each state
        log_pi = func.log_softmax(self.pi, dim=0)  # shape: (M,)
        log_alpha = torch.zeros(batch_size, max_time, self.num_states)
        if next(self.parameters()).is_cuda:
            log_alpha = log_alpha.cuda()

        # Initialize alpha[0]
        # Represents the prob of being in each state at time 0 given the first observation
        # Adding all prior probs with the emission probs for the first observation.
        # Notice how what would have been multiplying probs turns into adding due to log-space

        log_alpha[:, 0, :] = self.emission_model(x[:, 0, :]) + log_pi
        # Recursively compute alpha for t=1..T-1
        # Alpha is a tensor representing the log of probs up to time t-1 for each state
        for t in range(1, max_time):
            # alpha_next[i] = logsumexp over j of [alpha[j] + A[j, i]] + B[i, obs[t]]
            # This combines transition and emission probabiilities to compute the state probabilties at each time step
            log_alpha[:, t, :] = self.emission_model(
                x[:, t, :]
            ) + self.transition_model(log_alpha[:, t - 1, :])

        # P(obs_seq) = logsumexp(alpha[T-1, :])
        # Finding log of the probability of an observation sequence by looking summing at the final timestamp
        # log_sums = log_alpha.logsumexp(dim=2)
        # log_probs = torch.gather(log_sums, 1, T.view(-1, 1) - 1)
        return log_alpha

    def backward(self, x, T):
        """
        x (IntTensor): Observed sequences with shape (batch_size, T_max, D).
        T (IntTensor): Lengths of each sequence with shape (batch_size).

        Returns log_beta (Tensor).
        Backward probabilities with shape (batch_size, T_max, num_states).
        """
        if next(self.parameters()).is_cuda:
            x = x.to(next(self.parameters()).device)
            T = T.to(next(self.parameters()).device)

        batch_size, T_max, _ = x.shape
        log_beta = torch.full(
            (batch_size, T_max, self.num_states), -float("inf"), dtype=torch.float32
        )
        log_beta[range(batch_size), T - 1, :] = 0.0

        if next(self.parameters()).is_cuda:
            log_beta = log_beta.cuda()

        log_transition_matrix = func.log_softmax(
            self.transition_model.unnormalized_transition_matrix, dim=1
        )
        for t in range(T_max - 2, -1, -1):
            # Mask to identify which sequences are still active at time t+1
            mask = (T > t + 1).float().unsqueeze(1)  # Shape: (batch_size, 1)

            log_B_t1 = self.emission_model(x[:, t + 1, :])
            log_B_t1 = log_B_t1.unsqueeze(1)  # (batch_size, 1, N)
            log_beta_t1 = log_beta[:, t + 1, :].unsqueeze(1)  # (batch_size, 1, N)
            log_A = log_transition_matrix.unsqueeze(0)  # (1, N, N)

            elementwise_sum = log_A + log_B_t1 + log_beta_t1  # (batch_size, N, N)
            log_sum = torch.logsumexp(elementwise_sum, dim=2)  # (batch_size, N)
            log_sum_masked = torch.where(
                mask == 1.0,
                log_sum,
                torch.full_like(log_sum, -float("inf")),
            )
            log_beta[:, t, :] = log_sum_masked

        return log_beta

    def gamma(self, log_alpha, log_beta, T):
        """
        log_alpha : Tensor of Forward probabilities with shape (batch_size, T_max, num_states)
        log_beta : Tensor of Backward probabilities with shape (batch_size, T_max, num_states)
        T : Tensor with lengths of each batch with shape (batch_size,)

        Returns Log Gamma: Tensor of Posterior probabilies of states with shape (batch_size, T_max, num_states)

        """
        batch_size, _, _ = log_alpha.shape
        # Computing the log probabilty of the entire observation sequence in the batch
        log_p_x = (
            torch.logsumexp(log_alpha[range(batch_size), T - 1, :], dim=1)
            .unsqueeze(1)
            .unsqueeze(2)
        )  # (batch_size, 1, 1)
        print(log_alpha.shape)
        print(log_beta.shape)
        print(log_p_x.shape)

        log_gamma = log_alpha + log_beta - log_p_x  # (batch_size, T_max, N)

        return log_gamma

    def xi(self, log_alpha, log_beta, x, T):
        """
        log_alpha : Tensor of Forward probabilities with shape (batch_size, T_max, num_states)
        log_beta : Tensor of Backward probabilities with shape (batch_size, T_max, num_states)

        x (IntTensor): Observed sequences with shape (batch_size, T_max, D).
        T : Tensor with lengths of each batch with shape (batch_size,)

        Returns Log Xi: Tensor of probabilies of being in state i at time t and transitioning to another state with shape (batch_size, T_max, num_states, num_states)
        """
        log_B_list = [
            self.emission_model(x[:, t, :]) for t in range(1, x.shape[1])
        ]  # Each element in has shape (batch_size, N)
        # Stack along a new time dimension to get shape (batch_size, T-1, N)
        log_B = torch.stack(log_B_list, dim=1)
        # Now unsqueeze to get (batch_size, T-1, 1, N)
        log_B = log_B.unsqueeze(2)

        log_A = (
            torch.log_softmax(
                self.transition_model.unnormalized_transition_matrix, dim=1
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, N, N)

        log_p_x = (
            torch.logsumexp(log_alpha[:, -1, :], dim=1)
            .unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
        )

        log_alpha_i = log_alpha[:, :-1, :].unsqueeze(3)  # (batch_size, T_max-1, N, 1)
        log_beta_j = log_beta[:, 1:, :].unsqueeze(2)  # (batch_size, T_max-1, 1, N)

        log_xi = log_alpha_i + log_A + log_B + log_beta_j - log_p_x

        return log_xi

    def baum_welch(self, X, T, num_iterations=100, threshold=1e-4) -> str:
        """
        X (batch_size, T, D): A batch of observation sequences.
        T (List[int] or Tensor): Lengths of each observation sequence.
        num_iterations (int): Number of EM iterations.

        """

        batch_size, T_max, _ = X.shape

        # Initializing means
        # TODO: Change to k-Means
        x_flat = X.float().reshape(-1, X.shape[-1])
        indices = torch.randperm(x_flat.shape[0])[: self.num_states]
        initial_means = x_flat[indices]
        self.emission_model.means.data.copy_(initial_means)

        if next(self.parameters()).is_cuda:
            X = X.to(next(self.parameters()).device)
            T = T.to(next(self.parameters()).device)

        prev_log_likelihood = -float("inf")

        for i in range(num_iterations):
            log_alpha = self.forward(X, T)
            log_beta = self.backward(X, T)
            log_gamma = self.gamma(log_alpha, log_beta, T)
            gamma_exp = torch.exp(log_gamma)
            print(gamma_exp.sum(dim=2))
            xi = self.xi(log_alpha, log_beta, X, T)
            _, _, N, _ = xi.shape

            # Update initial prbabilities
            self.pi.data.copy_(
                torch.logsumexp(log_gamma[:, 0, :], dim=0) - math.log(batch_size)
            )

            # Update Transition matrix
            # time mask for T[r] -2
            xi_time_ids = (
                torch.arange(T_max - 1).unsqueeze(0).expand(batch_size, T_max - 1)
            ).to(next(self.parameters()).device)

            gamma_time_ids = (
                torch.arange(T_max).unsqueeze(0).expand(batch_size, T_max)
            ).to(next(self.parameters()).device)

            xi_mask = (
                (xi_time_ids < (T - 1).unsqueeze(1))
                .unsqueeze(2)
                .unsqueeze(3)
                .expand(-1, -1, N, N)
            )
            gamma_mask = (
                (gamma_time_ids < (T - 1).unsqueeze(1)).unsqueeze(2).expand(-1, -1, N)
            )
            # print("Xi mask shape", xi_mask.shape)
            # print("Xi shape", xi.shape)

            masked_xi = torch.where(
                xi_mask, xi, torch.tensor(-float("inf"), device=xi.device)
            )
            masked_gamma = torch.where(
                gamma_mask,
                log_gamma,
                torch.tensor(-float("inf"), device=log_gamma.device),
            )

            log_xi_sum = torch.logsumexp(torch.logsumexp(masked_xi, dim=1), dim=0)
            log_gamma_sum = torch.logsumexp(torch.logsumexp(masked_gamma, dim=1), dim=0)
            self.transition_model.unnormalized_transition_matrix.data.copy_(
                log_xi_sum - log_gamma_sum.unsqueeze(1)
            )

            # Update Emission Model
            gamma = torch.exp(log_gamma).unsqueeze(3)  # (batch_size, T, N, 1)
            time_ids = torch.arange(T_max, device=T.device).unsqueeze(0)  # (1, T_max)
            time_mask = (
                (time_ids < T.unsqueeze(1)).unsqueeze(2).unsqueeze(3)
            )  # (batch_size, T, 1, 1)

            masked_gamma = torch.where(
                time_mask, gamma, torch.tensor(0.0, device=gamma.device)
            )

            X_expanded = X.unsqueeze(2)  # (batch_size, T, 1, D)
            weighted_sum = (masked_gamma * X_expanded).sum(dim=1)  # (batch_size, N, D)
            weight_total = masked_gamma.sum(dim=1)  # (batch_size, N, 1)
            new_means = weighted_sum.sum(dim=0) / weight_total.sum(dim=0)

            self.emission_model.means.data.copy_(new_means)

            delta = X.unsqueeze(2) - self.emission_model.means.unsqueeze(
                0
            )  # (batch_size,T, N, D)
            delta_expanded = delta.unsqueeze(-1)  # shape: (batch_size, T, N, D, 1)
            delta_t_expanded = delta.unsqueeze(-2)  # shape: (batch_size, T, N, 1, D)
            outer_product = torch.matmul(
                delta_expanded, delta_t_expanded
            )  # shape: (batch_size, T, N, D, D)

            masked_outer_product = masked_gamma.unsqueeze(-1) * outer_product

            new_covariance = masked_outer_product.sum(dim=1) / weight_total.unsqueeze(
                -1
            )  # shape: (batch_size, N, D, D)

            new_covariance_sum = new_covariance.sum(dim=0)  # shape: (N, D, D)
            # Ensuring strictly positive definite for covariance avg
            jitter = 1e-6 * torch.eye(
                self.emission_model.D, device=new_covariance_sum.device
            )
            new_covariance_sum = new_covariance_sum + jitter.unsqueeze(0)

            new_cholesky = torch.linalg.cholesky(new_covariance_sum)  # shape: (N, D, D)
            self.emission_model.cholesky_factors.data.copy_(new_cholesky)

            new_log_likelihood = self.log_likelihood(log_alpha, T)

            if abs(new_log_likelihood.mean().item() - prev_log_likelihood) < threshold:
                return f"Converged after {i} iterations"

            prev_log_likelihood = new_log_likelihood.mean().item()
        return f"Finished {num_iterations} iterations of Baum Welch with a log_likelihood of {new_log_likelihood}"
