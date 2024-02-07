import sys
sys.path.append("../")

import torch
from samplers.HamiltonianMonteCarlo import HamiltonianMonteCarlo


# Swap function for parallel tempering
def attempt_swap(chain_a, chain_b, inv_temp_a, inv_temp_b, target_density_and_grad_fn, device):
    log_prob_a, _ = target_density_and_grad_fn(chain_a, 1.0)  # this is untempered log prob
    log_prob_b, _ = target_density_and_grad_fn(chain_b, 1.0)  # this is untempered log prob
    
    log_acceptance_ratio = (inv_temp_a - inv_temp_b) * (log_prob_b - log_prob_a)
    is_accept = torch.rand_like(log_acceptance_ratio, device=device).log() < log_acceptance_ratio
    is_accept = is_accept.unsqueeze(-1)

    new_chain_a = torch.where(is_accept, chain_b.detach().clone(), chain_a.detach().clone())
    new_chain_b = torch.where(is_accept, chain_a.detach().clone(), chain_b.detach().clone())

    return new_chain_a, new_chain_b


def parallel_tempering_step(x, num_chains, energy_func, step_size, num_leapfrog_steps_per_hmc_step, num_hmc_steps, inv_temperatures, device="cpu"):
    chains = [None for _ in range(num_chains)]
    hmcs = [None for _ in range(num_chains)]

    for i in range(num_chains):
        hmcs[i] = HamiltonianMonteCarlo(x[i], energy_func, step_size, num_leapfrog_steps_per_hmc_step, inv_temperatures[i], device)

    for j in range(num_hmc_steps):

        # Update each chain with HMC
        for i in range(num_chains):
            chains[i] = hmcs[i].sample()
            
        # if (j + 1) < num_hmc_steps:
        # Attempt swaps between adjacent chains
        for i in range(num_chains - 1):
            chains[i], chains[i + 1] = attempt_swap(chains[i], chains[i + 1], inv_temperatures[i], inv_temperatures[i + 1], hmcs[i].target_density_and_grad_fn, device)

    return chains
