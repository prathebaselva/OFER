import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import math


class VarianceScheduleTestSampling(Module):
    def __init__(self, num_steps, beta_1, beta_T, eta=0,mode='linear'):
        super().__init__()
        assert mode in ('linear', 'cosine' )
        self.mode = mode
        #print(self.mode)
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.num_steps = num_steps


        if self.mode == 'linear':
            betas = torch.linspace(self.beta_1, self.beta_T, steps=(self.num_steps))
            self.num_steps = len(betas)

        elif self.mode == 'cosine':
            s = 0.008
            warmupfrac = 1
            frac_steps = int(self.num_steps * warmupfrac)
            rem_steps = self.num_steps - frac_steps
            ft = [math.cos(((t/self.num_steps + s)/(1+s))*(math.pi/2))**2 for t in range(num_steps+1)]
            alphabar = [(ft[t]/ft[0]) for t in range(frac_steps+1)]
            betas = np.zeros(self.num_steps)
            for i in range(1,frac_steps+1):
                betas[i-1] = min(1-(alphabar[i]/alphabar[i-1]), 0.999)
            self.num_steps = len(betas)

        self.num_steps = len(betas)

        betas = np.array(betas, dtype=np.float32)
        assert((betas > 0).all() and (betas <=1).all())

        alphas = 1 - betas
        alpha_cumprod = np.cumprod(alphas)
        alpha_cumprod_prev = np.append(1., alpha_cumprod[:-1])
        sigma = eta*np.sqrt((1-(alpha_cumprod/alpha_cumprod_prev))*(1-alpha_cumprod_prev)/(1-alpha_cumprod))
        sigma = torch.tensor(sigma)
        alphas_cumprod_prev = torch.tensor(alpha_cumprod_prev)
        sqrt_alpha_cumprod = torch.tensor(np.sqrt(alpha_cumprod))
        sqrt_one_minus_alpha_cumprod = torch.tensor(np.sqrt(1.0 - alpha_cumprod))
        log_one_minus_alpha_cumprod = torch.tensor(np.log(1.0 - alpha_cumprod))
        sqrt_recip_alpha_cumprod = torch.tensor(np.sqrt(1.0/alpha_cumprod))
        sqrt_recip_minus_one_alpha_cumprod = torch.tensor(np.sqrt((1.0/alpha_cumprod) -1))
        sqrt_recip_one_minus_alpha_cumprod = np.sqrt(1.0/(1 - alpha_cumprod))


        posterior_variance = (betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        posterior_log_variance_clipped = torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)))
        posterior_mean_coeff1 = torch.tensor(betas * np.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod))
        posterior_mean_coeff2 = torch.tensor((1.0 - alpha_cumprod_prev)*np.sqrt(alphas) / (1 - alpha_cumprod))
        posterior_mean_coeff3 = torch.tensor((betas * sqrt_recip_one_minus_alpha_cumprod))
        betas = torch.tensor(betas)
        alphas = torch.tensor(alphas)
        alphas_cumprod = torch.tensor(alpha_cumprod)
        posterior_variance = torch.tensor(posterior_variance)

        self.register_buffer('test_betas', betas)
        self.register_buffer('test_alphas', alphas)
        self.register_buffer('test_sigma', sigma)
        self.register_buffer('test_alphas_cumprod', alphas_cumprod)
        self.register_buffer('test_alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('test_sqrt_alpha_cumprod', sqrt_alpha_cumprod)
        self.register_buffer('test_sqrt_one_minus_alpha_cumprod', sqrt_one_minus_alpha_cumprod)
        self.register_buffer('test_log_one_minus_alpha_cumprod', log_one_minus_alpha_cumprod)
        self.register_buffer('test_sqrt_recip_alpha_cumprod', sqrt_recip_alpha_cumprod)
        self.register_buffer('test_sqrt_recip_minus_one_alpha_cumprod', sqrt_recip_minus_one_alpha_cumprod)

        self.register_buffer('test_posterior_variance', posterior_variance)
        self.register_buffer('test_posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('test_posterior_mean_coeff1', posterior_mean_coeff1)
        self.register_buffer('test_posterior_mean_coeff2', posterior_mean_coeff2)
        self.register_buffer('test_posterior_mean_coeff3', posterior_mean_coeff3)

class VarianceScheduleMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.num_steps = config.num_steps
        self.beta_1 = config.beta_1
        self.beta_T = config.beta_T
        assert self.mode in ('linear', 'cosine' )

        if self.mode == 'linear':
            betas = torch.linspace(self.beta_1, self.beta_T, steps=(self.num_steps))
            self.num_steps = len(betas)

        elif self.mode == 'cosine':
            s = 0.008
            warmupfrac = 1
            frac_steps = int(self.num_steps * warmupfrac)
            rem_steps = self.num_steps - frac_steps
            ft = [math.cos(((t/self.num_steps + s)/(1+s))*(math.pi/2))**2 for t in range(num_steps+1)]
            alphabar = [(ft[t]/ft[0]) for t in range(frac_steps+1)]
            betas = np.zeros(self.num_steps)
            for i in range(1,frac_steps+1):
                betas[i-1] = min(1-(alphabar[i]/alphabar[i-1]), 0.999)
            self.num_steps = len(betas)

        betas = np.array(betas, dtype=np.float32)
        assert((betas > 0).all() and (betas <=1).all())

        alphas = 1 - betas
        alpha_cumprod = np.cumprod(alphas)
        alpha_cumprod_prev = np.append(1., alpha_cumprod[:-1])

        alphas_cumprod_prev = torch.tensor(alpha_cumprod_prev)
        sqrt_alpha_cumprod = torch.tensor(np.sqrt(alpha_cumprod))
        sqrt_recip_alpha = torch.tensor(np.sqrt(1.0/alphas))
        sqrt_one_minus_alpha_cumprod = torch.tensor(np.sqrt(1.0 - alpha_cumprod))
        log_one_minus_alpha_cumprod = torch.tensor(np.log(1.0 - alpha_cumprod))
        sqrt_recip_alpha_cumprod = torch.tensor(np.sqrt(1.0/alpha_cumprod))
        sqrt_recip_minus_one_alpha_cumprod = torch.tensor(np.sqrt((1.0/alpha_cumprod) -1))
        sqrt_recip_one_minus_alpha_cumprod = np.sqrt(1.0/(1 - alpha_cumprod))

        posterior_variance = (betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        posterior_log_variance_clipped = torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)))
        posterior_mean_coeff1 = torch.tensor(betas * np.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod))
        posterior_mean_coeff2 = torch.tensor((1.0 - alpha_cumprod_prev)*np.sqrt(alphas) / (1 - alpha_cumprod))
        posterior_mean_coeff3 = torch.tensor((betas * sqrt_recip_one_minus_alpha_cumprod))
        betas = torch.tensor(betas)
        alphas = torch.tensor(alphas)
        sqrt_alpha = torch.tensor(np.sqrt(alphas))
        alphas_cumprod = torch.tensor(alpha_cumprod)
        one_minus_alpha_cumprod = torch.tensor(1.0 - alpha_cumprod)
        mean_coeff = (one_minus_alpha_cumprod / betas)
        posterior_variance = torch.tensor(posterior_variance)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alpha_cumprod', sqrt_alpha_cumprod)
        self.register_buffer('sqrt_alpha', sqrt_alpha)
        self.register_buffer('sqrt_recip_alpha', sqrt_recip_alpha)
        self.register_buffer('sqrt_one_minus_alpha_cumprod', sqrt_one_minus_alpha_cumprod)
        self.register_buffer('log_one_minus_alpha_cumprod', log_one_minus_alpha_cumprod)
        self.register_buffer('sqrt_recip_alpha_cumprod', sqrt_recip_alpha_cumprod)
        self.register_buffer('sqrt_recip_minus_one_alpha_cumprod', sqrt_recip_minus_one_alpha_cumprod)

        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coeff1', posterior_mean_coeff1)
        self.register_buffer('posterior_mean_coeff2', posterior_mean_coeff2)
        self.register_buffer('posterior_mean_coeff3', posterior_mean_coeff3)
        self.register_buffer('mean_coeff', mean_coeff)


    def uniform_sample_t(self, batch_size, visualize=False):
        ts = np.random.choice(np.arange(self.num_steps), batch_size)
        if visualize:
            ts[batch_size-1] = 999
        return ts.tolist()


class FlameParamDiffusion(Module):
    def __init__(self, net, var_sched:VarianceScheduleMLP, device, tag, nettype):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.tag = tag
        self.nettype= nettype
        self.device = device

    def decode(self, epoch, flameparam_x0, context, flame=None, visualize=False, codedict=None): 
        """
        Args:
            flameparam_x0:  Input flame parameters, (B, N, d) ==> Batch_size X Number of points X point_dim(3).
            context:  Image latent, (B, F). ==> Batch_size X Image_latent_dim 
            lossparam: NetworkLossParam object.
        """
        batch_size, _ = flameparam_x0.size()

        t = None
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size, visualize)
        flameparam_xt, e_rand = self.get_train_mesh_sample(flameparam_x0, t, epoch)
        predflameparam_x0 = None
        getflameparam_x0 = False
        pred_theta, pred_flameparam_x0 = self.get_network_prediction(flameparam_xt=flameparam_xt, t=t, context=context, prednoise=True, getmeshx0=True, codedict=codedict)

        return pred_theta.view(batch_size,-1), e_rand.view(batch_size, -1), pred_flameparam_x0

    def get_meshx0_from_meanpred(self, flameparam_xt, pred_theta, t):
        mean_coeff = self.var_sched.mean_coeff[t].view(-1,1)
        sqrt_alpha = self.var_sched.sqrt_alpha[t].view(-1,1)
        flameparam_x0 =   (flameparam_xt * (1 - mean_coeff)) + sqrt_alpha * pred_theta * mean_coeff
        return flameparam_x0

    def get_meshx0_from_noisepred(self, flameparam_xt, pred_theta, t):
        sqrt_recip_alpha_cumprod = self.var_sched.sqrt_recip_alpha_cumprod[t].view(-1,1)
        sqrt_recip_minus_one_alpha_cumprod = self.var_sched.sqrt_recip_minus_one_alpha_cumprod[t].view(-1,1)
        flameparam_x0 =  (sqrt_recip_alpha_cumprod * flameparam_xt) - (sqrt_recip_minus_one_alpha_cumprod * pred_theta)
        return flameparam_x0

    def get_meshx0_from_noisepred_sampling(self, flameparam_xt, pred_theta, t, varsched):
        t = torch.Tensor(t).long().to(self.device)
        sqrt_recip_alpha_cumprod = varsched.test_sqrt_recip_alpha_cumprod[t].view(-1,1).to(self.device)
        sqrt_recip_minus_one_alpha_cumprod = varsched.test_sqrt_recip_minus_one_alpha_cumprod[t].view(-1,1).to(self.device)
        flameparam_x0 =  (sqrt_recip_alpha_cumprod * flameparam_xt) - (sqrt_recip_minus_one_alpha_cumprod * pred_theta)
        return flameparam_x0

    def get_network_prediction(self, flameparam_xt, t, context=None, prednoise=True, getmeshx0=False, issampling=False, varsched=None, codedict=None):
        flameparam_xt = flameparam_xt.to(dtype=torch.float32).to(self.device)
        t = torch.Tensor(t).long().to(self.device)
        if context is not None:
            context = context.to(self.device)

        pred_theta = self.net(flameparam_xt.to(self.device), t=t, context=context.to(self.device))

        flameparam_x0 = None
        if prednoise:
            pred_theta = pred_theta
            if getmeshx0:
                if issampling and (varsched is not None):
                    flameparam_x0 = self.get_meshx0_from_noisepred_sampling(flameparam_xt, pred_theta, t, varsched)
                else:
                    flameparam_x0 = self.get_meshx0_from_noisepred(flameparam_xt, pred_theta, t)
        else:
            flameparam_x0 = pred_theta
            pred_theta = None
        return pred_theta, flameparam_x0

    def get_meanxt(self, flameparam_xt,t, e_rand):
        posterior_mean_coeff3 = self.var_sched.posterior_mean_coeff3[t].view(-1,1).to(self.device)
        mean = self.var_sched.sqrt_recip_alpha[t].view(-1,1) * (flameparam_xt - posterior_mean_coeff3 * e_rand)
        return mean

    def get_train_mesh_sample(self, flameparam_x0, t, epoch=None):
        e_rand = torch.zeros(flameparam_x0.shape).to(self.device)  # (B, N, d)
        batch_size = flameparam_x0.shape[0]
        e_rand = torch.randn_like(flameparam_x0)
        sqrt_alpha_cumprod = self.var_sched.sqrt_alpha_cumprod[t].view(-1,1)
        sqrt_one_minus_alpha_cumprod = self.var_sched.sqrt_one_minus_alpha_cumprod[t].view(-1,1)
        flameparam_xt = (sqrt_alpha_cumprod * flameparam_x0) + (sqrt_one_minus_alpha_cumprod * e_rand)
        return flameparam_xt, e_rand 

    def get_shapemlp_loss(self, epoch, flameparam_x0, context):
        """
        Args:
            flameparam_x0:  Input point cloud, (B, N, d) ==> Batch_size X Number of points X point_dim(3).
            context:  Image latent, (B, F). ==> Batch_size X Image_latent_dim 
            lossparam: NetworkLossParam object.
        """
        batch_size, num_points, point_dim = flameparam_x0.size()

        t = None
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        flameparam_xt, e_rand = self.get_train_mesh_sample(flameparam_x0, t, epoch)
        getflameparam_x0 = False
        predflameparam_x0 = None
        if (epoch >= 1) and (np.random.rand() < 0.001):
            getflameparam_x0 = True
        pred_theta, predflameparam_x0 = self.get_network_prediction(flameparam_xt, t, context, True, getflameparam_x0)

        if getflameparam_x0:
            indx = np.random.randint(batch_size, size=(15,))
            sampt = np.array(t)[indx]

        loss = F.mse_loss(pred_theta.view(-1, 3), e_rand.view(-1,3), reduction='mean')
        return loss

    def get_pposterior_sample(self, pred_mesh_x0, mesh_xt, t, varsched):
        posterior_mean_coeff1 = varsched.test_posterior_mean_coeff1[t].view(-1,1).to(self.device)
        posterior_mean_coeff2 = varsched.test_posterior_mean_coeff2[t].view(-1,1).to(self.device)
        posterior_variance = varsched.test_posterior_variance[t].view(-1,1).to(self.device)
        posterior_log_variance_clipped = varsched.test_posterior_log_variance_clipped[t].view(-1,1).to(self.device)
        mean = posterior_mean_coeff1 * pred_mesh_x0 + posterior_mean_coeff2 * mesh_xt
        return mean, posterior_log_variance_clipped, posterior_variance

    def get_pposterior_sample1(self, pred_flameparam_x0, pred_theta, t, varsched):
        mean_coeff = torch.sqrt(varsched.test_alphas_cumprod_prev[t]).view(-1,1).to(self.device)
        dir_xt = torch.sqrt(1- varsched.test_alphas_cumprod_prev[t] - (varsched.test_sigma[t] **2)).view(-1,1).to(self.device)
        mean = mean_coeff * pred_flameparam_x0  + dir_xt * pred_theta
        return mean

    def get_mean_var(self, mesh_xt, pred_theta, t, varsched):
        sqrt_recip_alphas = torch.sqrt(1.0/ varsched.test_alphas[t]).view(-1,1).to(self.device)
        posterior_variance = torch.sqrt(varsched.test_posterior_variance[t]).view(-1,1).to(self.device)
        posterior_log_variance_clipped = ((0.5 * varsched.test_posterior_log_variance_clipped[t]).exp()).view(-1,1).to(self.device)
        c1 = ((1 - varsched.test_alphas[t])/(torch.sqrt(1 - varsched.test_alphas_cumprod[t]))).view(-1,1).to(self.device)
        mean = sqrt_recip_alphas * (mesh_xt - c1 * pred_theta)
        return mean, posterior_variance, posterior_log_variance_clipped

    def project_mean(self, mesh_xt, pred_theta, t, varsched):
        sqrt_recip_alphas = torch.sqrt(1.0/ varsched.test_alphas[t]).view(-1,1).to(self.device)
        posterior_variance = torch.sqrt(varsched.test_posterior_variance[t]).view(-1,1).to(self.device)
        posterior_log_variance_clipped = ((0.5 * varsched.test_posterior_log_variance_clipped[t]).exp()).view(-1,1).to(self.device)
        c1 = ((1 - varsched.test_alphas[t])/(torch.sqrt(1 - varsched.test_alphas_cumprod[t]))).view(-1,1).to(self.device)
        mean = sqrt_recip_alphas * (mesh_xt - c1 * pred_theta)
        return mean, posterior_variance, posterior_log_variance_clipped

    def get_var(self, t, varsched):
        posterior_mean_coeff3 = varsched.test_posterior_mean_coeff3[t].view(-1,1).to(self.device)
        posterior_variance = torch.sqrt(varsched.test_posterior_variance[t]).view(-1,1).to(self.device)
        posterior_log_variance_clipped = ((0.5 * varsched.test_posterior_log_variance_clipped[t]).exp()).view(-1,1).to(self.device)
        return  posterior_variance, posterior_log_variance_clipped

    def sample(self, num_points, context, batch_size=1, point_dim=3, sampling='ddpm', shapeparam=None, expparam=None, fixed_noise=None, codedict=None): 
        mesh_xT = torch.randn(size=(batch_size, num_points)).to(self.device)
        context = context.to(self.device)
        if sampling == 'ddim':
            varsched = VarianceScheduleTestSampling(self.var_sched.num_steps, self.var_sched.beta_1, self.var_sched.beta_T, 0, self.var_sched.mode).to(self.device)
            iterator = [x for x in reversed(range(-1,varsched.num_steps,4))]
        else:
            varsched = VarianceScheduleTestSampling(self.var_sched.num_steps, self.var_sched.beta_1, self.var_sched.beta_T, 1, self.var_sched.mode).to(self.device)
            iterator = [x for x in reversed(range(0, varsched.num_steps))]

        traj = {varsched.num_steps-1: mesh_xT}
        iteri = 1

        r = np.random.randint(0, batch_size)
        count = 0
        for idx, t in enumerate(iterator):
            z = torch.zeros(mesh_xT.shape).to(self.device)  # (B, N, d)
            if t > 0:
                z = torch.normal(0,1, size=(mesh_xT.shape)).to(self.device)

            flameparam_xt = traj[t]
            
            batch_t = ([t]*batch_size)
            pred_theta, pred_flameparam_x0 = self.get_network_prediction(flameparam_xt, batch_t, context, getmeshx0=True, issampling=True, varsched=varsched)
            # When we use diffusion directly from data and not the noise
            if sampling == 'ddim1':
               mean, logvar, var = self.get_pposterior_sample(pred_theta, flameparam_xt, batch_t, varsched)
               flameparam_xprevt = mean + (0.5 * logvar).exp() * z
            elif sampling == 'ddim':
                mean = self.get_pposterior_sample1(pred_flameparam_x0, pred_theta, batch_t, varsched)
                flameparam_xprevt = mean + varsched.test_sigma[t].view(-1,1) * z
            elif sampling == 'ddm':
                mean, logvar, var = self.get_pposterior_sample(pred_flameparam_x0, flameparam_xt, batch_t, varsched)
                flameparam_xprevt = mean + (0.5*logvar).exp() * z
            else:
                mean, var, logvar = self.get_mean_var(flameparam_xt, pred_theta, batch_t, varsched)
                flameparam_xprevt = mean + logvar * z
            if t > 0:
                traj[iterator[idx+1]] = flameparam_xprevt.clone().detach()     # Stop gradient and save trajectory.
                del traj[t]
            else:
                traj[-1] = flameparam_xprevt.clone().detach()
            count += 1


        return traj[-1]
