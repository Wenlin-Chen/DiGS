import sys
sys.path.append("../")

import torch
from functools import partial
import copy


def target_density_and_grad_fn_full(x, inv_temperature, target_log_prob_fn):
    x = x.clone().detach().requires_grad_(True)
    log_prob = target_log_prob_fn(x) * inv_temperature
    log_prob_sum = log_prob.sum()
    log_prob_sum.backward()
    grad = x.grad.clone().detach()
    return log_prob.detach(), grad


class HamiltonianMonteCarlo(object):

    def __init__(self,
                 x,
                 energy_func: callable,
                 step_size: float,
                 num_leapfrog_steps_per_hmc_step: int,
                 inv_temperature: float = 1.0,
                 device: str = 'cpu'):
        super(HamiltonianMonteCarlo, self).__init__()

        self.x = x
        self.step_size = step_size
        self.target_density_and_grad_fn = partial(target_density_and_grad_fn_full, target_log_prob_fn=lambda x: -energy_func(x))
        self.device = device
        self.inv_temperature = inv_temperature
        self.num_leapfrog_steps_per_hmc_step = num_leapfrog_steps_per_hmc_step

        self.current_log_prob, self.current_grad = self.target_density_and_grad_fn(x, self.inv_temperature)

    def leapfrog_integration(self, p):
        """
        Leapfrog integration for simulating Hamiltonian dynamics.
        """
        x = self.x.detach()
        p = p.detach()

        # Half step for momentum
        p += 0.5 * self.step_size * self.current_grad

        # Full steps for position
        for _ in range(self.num_leapfrog_steps_per_hmc_step - 1):
            x += self.step_size * p
            _, grad = self.target_density_and_grad_fn(x, self.inv_temperature)
            p += self.step_size * grad  # this combines two half steps for momentum

        # Final update of position and half step for momentum
        x += self.step_size * p
        _, grad = self.target_density_and_grad_fn(x, self.inv_temperature)
        p += 0.5 * self.step_size * grad

        return x, p


    def sample(self):
        """
        Hamiltonian Monte Carlo step.
        """

        # Sample a new momentum
        p = torch.randn_like(self.x, device=self.device)

        # Simulate Hamiltonian dynamics
        new_x, new_p = self.leapfrog_integration(p)

        # Compute new log probability and gradient
        new_log_prob, new_grad = self.target_density_and_grad_fn(new_x, self.inv_temperature)

        # Hamiltonian (log probability + kinetic energy)
        current_hamiltonian = self.current_log_prob - 0.5 * p.pow(2).sum(-1)
        new_hamiltonian = new_log_prob - 0.5 * new_p.pow(2).sum(-1)
        
        log_acceptance_ratio = current_hamiltonian - new_hamiltonian
        is_accept = torch.rand_like(log_acceptance_ratio, device=self.device).log() < log_acceptance_ratio
        is_accept = is_accept.unsqueeze(-1)

        self.x = torch.where(is_accept, new_x.detach(), self.x)
        self.current_grad = torch.where(is_accept, new_grad.detach(), self.current_grad)
        self.current_log_prob = torch.where(is_accept.squeeze(-1), new_log_prob.detach(), self.current_log_prob)
        
        return copy.deepcopy(self.x.detach())

    
