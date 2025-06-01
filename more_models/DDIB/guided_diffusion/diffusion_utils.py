
import numpy as np
import torch as th

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = th.gather(th.tensor(a, dtype=th.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out

def sample_xt(x0, t, b):
    at = extract(np.cumprod((1.0 - b),axis=0), t, x0.shape)  # at is the \hat{\alpha}_t
    # print('at', at)
    xt = at.sqrt() * x0 + (1 - at).sqrt() * th.randn_like(x0)
    return xt

def denoising_step(xt, t, t_next, *,
                   model,
                   b,
                   logvars=None,
                   sampling_type='ddpm',
                   eta=0.0,
                   learn_sigma=False,
                   out_x0_t=False,
                   ):

    # Compute noise and variance
    et = model(xt, t)
    if et.shape != xt.shape:
            et, model_var_values = th.split(et, et.shape[1] // 2, dim=1)
    if learn_sigma:
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        bt = extract(b, t, xt.shape)
        at = extract(np.cumprod((1.0 - b),axis=0), t, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)
        if t_next.sum() == -t_next.shape[0]:  # if t_next is -1
            at_next = th.ones_like(at)
        else:
            at_next = extract(np.cumprod((1.0 - b),axis=0), t_next, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)
        posterior_variance = bt * (1.0 - at_next) / (1.0 - at)
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        min_log = th.log(posterior_variance.clamp(min=1e-6))
        max_log = th.log(bt)
        frac = (model_var_values + 1) / 2
        logvar = frac * max_log + (1 - frac) * min_log
    else:
        logvar = extract(logvars, t, xt.shape)


    # Compute the next x
    bt = extract(b, t, xt.shape)  # bt is the \beta_t
    at = extract(np.cumprod((1.0 - b),axis=0), t, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)

    if t_next.sum() == -t_next.shape[0]:  # if t_next is -1
        at_next = th.ones_like(at)
    else:
        at_next = extract(np.cumprod((1.0 - b),axis=0), t_next, xt.shape)  # at_next is the \hat{\alpha}_{t_next}

    xt_next = th.zeros_like(xt)
    if sampling_type == 'ddpm':
        weight = bt / th.sqrt(1 - at)

        mean = 1 / th.sqrt(1.0 - bt) * (xt - weight * et)
        noise = th.randn_like(xt)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
        xt_next = mean + mask * th.exp(0.5 * logvar) * noise
        xt_next = xt_next.float()

    elif sampling_type == 'ddim':
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # predicted x0_t
        if eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        elif at > (at_next):
            print('Inversion process is only possible with eta = 0')
            raise ValueError
        else:
            c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()  # sigma_t
            c2 = ((1 - at_next) - c1 ** 2).sqrt()  # direction pointing to x_t
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * th.randn_like(xt)

    if out_x0_t == True:
        return xt_next, x0_t
    else:
        return xt_next
    


def denoising_step_with_eps(xt, eps, t, t_next, *,
                            model,
                            b,
                            logvars=None,
                            sampling_type='ddpm',
                            eta=0.0,
                            learn_sigma=False,
                            out_x0_t=False,
                            ):
        assert eps.shape == xt.shape

        # Compute noise and variance
        et = model(xt, t)
        if et.shape != xt.shape:
            et, model_var_values = th.split(et, et.shape[1] // 2, dim=1)
        if learn_sigma:
            et, model_var_values = th.split(et, et.shape[1] // 2, dim=1)
            # calculations for posterior q(x_{t-1} | x_t, x_0)
            bt = extract(b, t, xt.shape)
            at = extract(np.cumprod((1.0 - b),axis=0), t, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)
            at_next = extract(np.cumprod((1.0 - b),axis=0), t_next, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)
            posterior_variance = bt * (1.0 - at_next) / (1.0 - at)
            # log calculation clipped because the posterior variance is 0 at the
            # beginning of the diffusion chain.
            min_log = th.log(posterior_variance.clamp(min=1e-6))
            max_log = th.log(bt)
            frac = (model_var_values + 1) / 2
            logvar = frac * max_log + (1 - frac) * min_log
        else:
            logvar = extract(logvars, t, xt.shape)


        # Compute the next x
        bt = extract(b, t, xt.shape)  # bt is the \beta_t
        at = extract(np.cumprod((1.0 - b),axis=0), t, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)

        if t_next.sum() == -t_next.shape[0]:  # if t_next is -1
            at_next = th.ones_like(at)
        else:
            at_next = extract(np.cumprod((1.0 - b),axis=0), t_next, xt.shape)  # at_next is the \hat{\alpha}_{t_next}

        xt_next = th.zeros_like(xt)
        if sampling_type == 'ddpm':
            weight = bt / th.sqrt(1 - at)

            mean = 1 / th.sqrt(1.0 - bt) * (xt - weight * et)
            noise = eps
            mask = 1 - (t == 0).float()
            mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
            xt_next = mean + mask * th.exp(0.5 * logvar) * noise
            xt_next = xt_next.float()

        elif sampling_type == 'ddim':
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # predicted x0_t
            if eta == 0:
                xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
            elif at > (at_next):
                print('Inversion process is only possible with eta = 0')
                raise ValueError
            else:
                c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()  # sigma_t
                c2 = ((1 - at_next) - c1 ** 2).sqrt()  # direction pointing to x_t
                xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * eps

        if out_x0_t == True:
            return xt_next, x0_t
        else:
            return xt_next
        

def sample_xt_next(model,x0, xt, t, t_next,b, eta=0.0,sampling_type='ddpm'):
    bt = extract(b, t, xt.shape)  # bt is the \beta_t
    at = extract(np.cumprod((1.0 - b),axis=0), t, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)

    assert not t_next.sum() == -t_next.shape[0]  # t_next should never be -1
    assert not t.sum() == 0  # t should never be 0
    at_next = extract(np.cumprod((1.0 - b),axis=0), t_next, xt.shape)  # at_next is the \hat{\alpha}_{t_next}

    if sampling_type == 'ddpm':
        w0 = at_next.sqrt() * bt / (1 - at)
        wt = (1 - bt).sqrt() * (1 - at_next) / (1 - at)
        mean = w0 * x0 + wt * xt

        var = bt * (1 - at_next) / (1 - at)

        xt_next = mean + var.sqrt() * th.randn_like(x0)
    elif sampling_type == 'ddim':
        et = (xt - at.sqrt() * x0) / (1 - at).sqrt()  # posterior et given x0 and xt
        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()  # sigma_t
        c2 = ((1 - at_next) - c1 ** 2).sqrt()  # direction pointing to x_t
        xt_next = at_next.sqrt() * x0 + c2 * et + c1 * th.randn_like(x0)
    else:
        raise ValueError()

    return xt_next



def compute_eps(xt, xt_next, t, t_next, model,b, logvars=None, eta=0.1, learn_sigma=False,sampling_type='ddpm'):
    assert eta is None or eta > 0
    # Compute noise and variance
    if type(model) != list:
        et = model(xt, t)
        if et.shape != xt.shape:
            et, model_var_values = th.split(et, et.shape[1] // 2, dim=1)
        if learn_sigma:
            # calculations for posterior q(x_{t-1} | x_t, x_0)
            bt = extract(b, t, xt.shape)
            at = extract(np.cumprod((1.0 - b),axis=0), t, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)
            at_next = extract(np.cumprod((1.0 - b),axis=0), t_next, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)
            posterior_variance = bt * (1.0 - at_next) / (1.0 - at)
            # log calculation clipped because the posterior variance is 0 at the
            # beginning of the diffusion chain.
            min_log = th.log(posterior_variance.clamp(min=1e-6))
            max_log = th.log(bt)
            frac = (model_var_values + 1) / 2
            logvar = frac * max_log + (1 - frac) * min_log
        else:
            logvar = extract(logvars, t, xt.shape)
    else:
        raise NotImplementedError()

    # Compute the next x
    bt = extract(b, t, xt.shape)  # bt is the \beta_t
    at = extract(np.cumprod((1.0 - b),axis=0), t, xt.shape)  # at is the \hat{\alpha}_t (DDIM does not use \hat notation)

    assert not t_next.sum() == -t_next.shape[0]  # t_next should never be -1
    assert not t.sum() == 0  # t should never be 0
    at_next = extract(np.cumprod((1.0 - b),axis=0), t_next, xt.shape)  # at_next is the \hat{\alpha}_{t_next}

    if sampling_type == 'ddpm':
        weight = bt / th.sqrt(1 - at)

        mean = 1 / th.sqrt(1.0 - bt) * (xt - weight * et)
        # print('torch.exp(0.5 * logvar).sum()', th.exp(0.5 * logvar).sum())
        eps = (xt_next - mean) / th.exp(0.5 * logvar)

    elif sampling_type == 'ddim':
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # predicted x0_t

        c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()  # sigma_t
        c2 = ((1 - at_next) - c1 ** 2).sqrt()  # direction pointing to x_t
        eps = (xt_next - at_next.sqrt() * x0_t - c2 * et) / c1
    else:
        raise ValueError()

    return eps