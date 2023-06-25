import time
import numpy as np
import torch
from scipy.optimize import minimize


def get_progression(sigma, Nit, mode="constant", param=None):
    if mode == "constant":
        return sigma/np.sqrt(Nit) * np.ones(Nit)

def general_loss(u, d, model, dt=1.0, verbose=False, noise_loader=None, device=0):
    assert (isinstance(u, np.ndarray) and isinstance(d, np.ndarray)) \
        or (isinstance(u, torch.Tensor) and isinstance(d, torch.Tensor)), 'u and d must both be either numpy arrays or torch tensors'

    scipy_input = (isinstance(u, np.ndarray) and isinstance(d, np.ndarray))

    # Torch tensor conversion
    if scipy_input:
        u_torch = torch.from_numpy(u).to(device, dtype=torch.float32).view(d.shape).requires_grad_(True)
        d_torch = torch.from_numpy(d).to(device, dtype=torch.float32)
    else:
        u_torch = u.requires_grad_(True)
        d_torch = d.detach()

    # Select a random batch of noise maps
    n_all = noise_loader()

    # Compute loss
    rep_d = model(d_torch)
    rep_u = model(u_torch + dt*n_all)
    l = (torch.absolute(((rep_d - rep_u))) ** 2).mean()

    if verbose:
        print(l.item())

    if scipy_input:
        l.backward()
        grad = u_torch.grad
        return l.item(), grad.detach().cpu().numpy().flatten().astype(np.float64)
    else:
        return l

def wph_loss(u, d, wph_op, dt=1.0, perturb=False, norm="S11", verbose=False, noise_loader=None, device=0, backward=False):
    assert (isinstance(u, np.ndarray) and isinstance(d, np.ndarray)) \
        or (isinstance(u, torch.Tensor) and isinstance(d, torch.Tensor)), 'u and d must both be either numpy arrays or torch tensors'

    scipy_input = (isinstance(u, np.ndarray) and isinstance(d, np.ndarray))

    # Torch tensor conversion
    if scipy_input:
        u_torch = torch.from_numpy(u).to(device, dtype=torch.float32).view(d.shape).requires_grad_(True)
        d_torch = torch.from_numpy(d).to(device, dtype=torch.float32)
    else:
        u_torch = u.requires_grad_(True)
        d_torch = d.detach()

    if not perturb: # Not perturbative loss
        # Select a random batch of noise maps
        n_all = noise_loader()

        # Normalization
        if norm == "S11":
            norm = wph_op.compute_s11_norm(d_torch)
        else:
            raise NotImplementedError

        # Compute loss
        rep_d = wph_op.compute_stats(d_torch)
        l = torch.Tensor([0.0]).to(device, dtype=torch.float32)
        for n_batch in n_all:
            rep_u = wph_op.compute_stats(u_torch + dt*n_batch)
            l_step = (torch.absolute(((rep_d - rep_u) / norm)) ** 2).mean() / n_all.shape[0]
            if backward: # Custom backward mode (to save memory)
                l_step.backward()
            l += l_step
            del rep_u
    else: # Perturbative loss
        # Compute representations and its derivatives
        rep_d = wph_op.compute_stats(d_torch)
        rep_u, rep_u_grad, rep_u_hess = wph_op.compute_stats(u_torch, derivatives=True)

        # Normalization
        if norm == "S11":
            norm = wph_op.compute_s11_norm(d_torch)
        else:
            raise NotImplementedError

        # Compute loss
        rep_diff = (rep_u - rep_d) / norm
        l = (torch.absolute(rep_diff) ** 2).sum()
        l = l + dt**2*(torch.sum(rep_u_grad / norm**2) + torch.dot(rep_u_hess / norm, rep_diff).real)
        l = l / (rep_diff.shape[0])
    if verbose:
        print(l.item())

    if scipy_input:
        if not backward: # If custom backward was not used before
            l.backward()
        grad = u_torch.grad
        return l.item(), grad.detach().cpu().numpy().flatten().astype(np.float64)
    else:
        return l

def denoiser(d, objective, dt=None,
             optimizer='scipy_lbfgs', optim_params=None,
             analysis_func=None, index=None,
             verbose=True, loss_print_interval=10, ret_losses=False,
             device='cpu'):
    start_time = time.time()
    
    # Identify optimizer and optimization parameters
    if optimizer == 'scipy_lbfgs':
        torch_optim = False
        optim_params = optim_params if optim_params is not None else {"maxiter": 10, "gtol": 1e-12, "ftol": 1e-12, "maxcor": 20}
    elif callable(optimizer): # If optimizer is a function, assume it is a torch optimizer
        torch_optim = True
        scheduler = None
        if optim_params is None:
            optim_params = {"lr": 1e-4}
            nsteps = 20
        else:
            optim_params = optim_params.copy()
            if 'nsteps' in optim_params:
                nsteps = optim_params['nsteps']
                del optim_params['nsteps']
            else:
                nsteps = 20
            if 'backward_custom' in optim_params:
                backward_custom = optim_params['backward_custom']
                del optim_params['backward_custom']
            else:
                backward_custom = False
            if 'scheduler' in optim_params:
                scheduler = optim_params['scheduler']
                del optim_params['scheduler']
                if 'scheduler_params' in optim_params:
                    scheduler_params = optim_params['scheduler_params']
                    del optim_params['scheduler_params']
                else:
                    scheduler_params = {}
    else:
        raise ValueError("Unknown optimizer")
    
    # Format input data
    if isinstance(d, torch.Tensor):
        if not torch_optim:
            d = d.cpu().detach().numpy()
    elif isinstance(d, np.ndarray):
        if torch_optim:
            d = torch.from_numpy(d).to(device, dtype=torch.float32)
    else:
        raise ValueError("Input data must be a torch.Tensor or a numpy.ndarray")
    
    # Initial conditions
    u0 = d
    input_shape = tuple(u0.shape)

    # Optimization
    i = index
    if dt is None: dt = 1
    if verbose: print("Starting optimization...")
    losses_it = []
    if torch_optim:
        u = u0.clone().requires_grad_(True)
        opt = optimizer([u], **optim_params)
        if scheduler is not None:
            opt_scheduler = scheduler(opt, **scheduler_params)
        for j in range(nsteps):
            def closure():
                opt.zero_grad()
                loss = objective(u, u0.clone(), dt)
                if not backward_custom:
                    loss.backward()
                losses_it.append(loss.detach().item())
                if verbose and j % loss_print_interval == 0:
                    print(loss.detach().item())
                return loss
            opt.step(closure)
            if scheduler:
                opt_scheduler.step()
        
        if analysis_func is not None:
            analysis_func(u0.cpu().numpy(), u.detach().cpu().numpy(), dt, i)
        u0 = u.detach()
    else:
        nfev = 0
        def callback(u):
            nonlocal nfev
            losses_it.append(objective(u, u0, dt)[0])
            if verbose and nfev % loss_print_interval == 0:
                print(losses_it[-1])
            nfev += 1
        result = minimize(lambda u: objective(u, u0, dt), u0.ravel(), method='L-BFGS-B', jac=True, tol=None,
                            options=optim_params, callback=callback)
        final_loss, uf, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        uf = uf.reshape(input_shape).astype(np.float32)
        if analysis_func is not None:
            analysis_func(u0, uf, dt, i)
        u0 = uf
        if verbose:
            print(f"Final loss: {final_loss}")
            print(f"Optimization ended in {niter} iterations with optimizer message: {msg}")
    print(f"Optimization time for one iteration: {time.time() - start_time}s")
    losses_it = np.array(losses_it)
    
    # u0 is the final denoised image
    if ret_losses:
        return u0, losses_it
    else:
        return u0 

def diffusive_denoiser(d, Nit, objective,
                       prog='constant', prog_param=None,
                       optimizer='scipy_lbfgs', optim_params=None,
                       analysis_func=None,
                       verbose=True, loss_print_interval=10, ret_losses=False, ret_lrs=False,
                       device='cpu'):
    total_start_time = time.time()

    # First parsing of optimization parameters
    lr_prog, lr_range, lr_loss_smoothing = None, None, None
    if optim_params is not None:
        optim_params = optim_params.copy()
    if optimizer == 'scipy_lbfgs':
        pass
    elif callable(optimizer): # If optimizer is a function, assume it is a torch optimizer
        if 'lr_prog' in optim_params:
            lr_prog = optim_params['lr_prog']
            assert len(lr_prog) == Nit, "lr_prog must have length Nit"
            del optim_params['lr_prog']
        if 'lr_range' in optim_params:
            lr_range = optim_params['lr_range']
            del optim_params['lr_range']
        if 'lr_loss_smoothing' in optim_params:
            lr_loss_smoothing = optim_params['lr_loss_smoothing']
            del optim_params['lr_loss_smoothing']
        assert (lr_prog is None) or (lr_range is None), "Choose between lr_prog and lr_range options"
    else:
        raise ValueError("Unknown optimizer")
    
    # Get progression
    dt = get_progression(1.0, Nit, prog, prog_param)
    if verbose: print("Sum of dt^2:", np.sum(dt**2))
    
    # Initial conditions
    u0 = d.copy() if isinstance(d, np.ndarray) else d.clone().cpu().detach().numpy()

    # Optimization
    losses = []
    lrs = []
    for i in range(Nit):
        if verbose: print(f"Iteration: {i + 1}/{Nit} - dt[i] = {dt[i]}")
        if lr_range is not None:
            if optim_params is None:
                optim_params = {}
            u0s = []
            min_losses = []
            losses_it_all = []
            for lr in lr_range:
                if verbose: print("lr:", lr)
                optim_params['lr'] = lr
                u0_tmp, losses_it = denoiser(u0, objective, dt[i],
                    optimizer=optimizer, optim_params=optim_params,
                    analysis_func=analysis_func, index=i,
                    verbose=verbose, loss_print_interval=loss_print_interval, ret_losses=True,
                    device=device)
                u0s.append(u0_tmp)
                if lr_loss_smoothing is not None:
                    min_loss = np.mean(losses_it[-lr_loss_smoothing:])
                else:
                    min_loss = losses_it[-1]
                min_losses.append(min_loss)
                losses_it_all.append(losses_it)
            best_lr_index = np.argmin(min_losses)
            u0 = u0s[best_lr_index]
            losses.append(losses_it_all[best_lr_index])
            lrs.append(lr_range[best_lr_index])
            if verbose: print("Best lr:", lrs[-1])
        else:
            if lr_prog is not None:
                if optim_params is None:
                    optim_params = {}
                optim_params['lr'] = lr_prog[i]
                lrs.append(lr_prog[i])
            ret = denoiser(u0, objective, dt[i],
                    optimizer=optimizer, optim_params=optim_params,
                    analysis_func=analysis_func, index=i,
                    verbose=verbose, loss_print_interval=loss_print_interval, ret_losses=ret_losses,
                    device=device)
            if ret_losses:
                u0, losses_it = ret
                losses.append(losses_it)
            else:
                u0 = ret
    losses = np.array(losses)
    print(f"Total optimization time: {time.time() - total_start_time}s")
    
    # u0 is the final denoised image
    ret = [u0]
    if ret_losses:
        ret.append(losses)
    if ret_lrs:
        ret.append(lrs)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret
