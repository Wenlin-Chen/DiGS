import sys
sys.path.append("../")

import torch
from samplers.Langevin import LangevinDynamics


def langevin_diffusive_gibbs_sampler_step(
    x, 
    energy, 
    alpha, 
    std,  
    num_gibbs_sweeps,
    num_langevin_steps_per_sweep,
    step_size,
    use_mala=True, 
    device="cpu"
):
    
    alpha=torch.tensor(alpha).to(device)
    std=torch.tensor(std).to(device)

    def denoising_LD_generator(energy, x, tx, alpha, std, step_size, mh):
        if std==0:
            negloglikelihood_func=lambda x:  energy(x)
        else:    
            negloglikelihood_func=lambda x:  energy(x) + ((tx-alpha*x)**2/(2*std**2)).sum(-1)
        return LangevinDynamics(x, 
                                negloglikelihood_func,
                                step_size=step_size,
                                mh=mh,
                                device=device)


    for gibbs_idx in range(num_gibbs_sweeps):
        tx=alpha*x+std*torch.randn_like(x, device=device)

        # propose to use tx/alpha as langevin init
        x_init = tx.detach().clone()/alpha + std/alpha * torch.randn_like(tx, device=device)

        # calculate accpetance rate for the proposal
        log_joint_prob_numerator = -energy(x_init) - ((tx-alpha*x_init)**2/(2*std**2)).sum(-1) - ((x-tx/alpha)**2/(2*(std/alpha)**2)).sum(-1)
        log_joint_prob_denominator = -energy(x) - ((tx-alpha*x)**2/(2*std**2)).sum(-1) - ((x_init-tx/alpha)**2/(2*(std/alpha)**2)).sum(-1)

        # accept the proposal (tx/alpha) as langevin init or reject it and use previous x as langevin init
        log_accept_rate = log_joint_prob_numerator - log_joint_prob_denominator
        is_accept = torch.rand_like(log_accept_rate, device=device).log() <= log_accept_rate
        x_init = torch.where(is_accept.unsqueeze(1), x_init.detach().clone(), x.detach().clone())

        langevin_dynamics=denoising_LD_generator(energy, x_init, tx.detach(), alpha, std, step_size=step_size, mh=use_mala)
        for _ in range(num_langevin_steps_per_sweep):
            x, _ = langevin_dynamics.sample()

    return x