import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.
        
        # Linear interpolation of beta_t
        all_beta_t = beta_1 + (beta_T - beta_1) * torch.arange(0, T+1,device=t_s.device) / (T - 1)
        beta_t = all_beta_t[t_s -1]
        
        # Compute other constants
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        oneover_sqrt_alpha = 1 / torch.sqrt(alpha_t)

        
        all_alpha_t = 1 - all_beta_t
        # Compute cumulative products of alpha_t
        alpha_t_bar = torch.cumprod(all_alpha_t, dim=0)[t_s -1]
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)



        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .
        x_0 = images
        c = conditions
        B = images.shape[0]
        p_uncond = self.dmconfig.mask_p

        c_0 = F.one_hot(c, num_classes = 10) #(B,10)
        mask = torch.bernoulli((torch.zeros_like(c)+p_uncond).to(device)).view(B,1)
        c_masked = c_0 * (1 - mask) + mask * self.dmconfig.condition_mask_value
        
        t = torch.randint(1, T+1, (B,)).to(device)  # t ~ Uniform(0, n_T)
        
        sqrt_alpha_bar = self.scheduler(t)['sqrt_alpha_bar'].to(device)
        sqrt_one_minus_alpha_bar = self.scheduler(t)['sqrt_oneminus_alpha_bar'].to(device)
        
        eps = torch.randn_like(x_0).to(device)
        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        x_t = sqrt_alpha_bar.view(-1,1,1,1) * x_0 + sqrt_one_minus_alpha_bar.view(-1,1,1,1) * eps  
        eps_theta = self.network(x_t, t/T, c_masked)
        # return MSE between added noise, and our predicted noise
        noise_loss = self.loss_fn(eps_theta, eps)

        # ==================================================== #
        
        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  

        B = conditions.size(0)
        c = conditions
        # Start by sampling noise from a normal distribution
        # X_t = torch.randn(batch_size, 1, 28, 28)  # Step 1: initialize X_T from N(0, I)
        X_t = torch.randn(B, self.dmconfig.num_channels, self.dmconfig.input_dim[0], self.dmconfig.input_dim[1]).to(device)
       

        # Convert conditions to one-hot encoded labels if they are not already
        if c.ndim < 2 or c.size(1) != self.dmconfig.num_classes:
           c = F.one_hot(c, num_classes=self.dmconfig.num_classes).float()
 
        with torch.no_grad():
          for t in reversed(range(1, T+1)):  # Step 2: loop backwards from T to 1
              
              # Get the variance and other constants for time t
              t_is = torch.full((B,), t).to(device).view(B, 1)
   
              # Sample random noise
              z = torch.randn_like(X_t).to(device) if t > 1 else torch.zeros_like(X_t).to(device)

              alpha_t = self.scheduler(t_is)['alpha_t'].to(device)
              sqrt_oneminus_alpha_bar = self.scheduler(t_is)['sqrt_oneminus_alpha_bar'].to(device)
              oneover_sqrt_alpha = self.scheduler(t_is)['oneover_sqrt_alpha'].to(device)
              sqrt_beta_t = self.scheduler(t_is)['sqrt_beta_t'].to(device)

              eps = self.network(X_t, t_is/T, c)

              
              c_uncond = torch.full_like(c, self.dmconfig.condition_mask_value).to(device)
              eps_uncond = self.network(X_t, t_is/T, c_uncond)
              # Corrected noise using classifier-free guidance
              eps_hat = (1 + omega) * eps - omega * eps_uncond
              
              X_t = oneover_sqrt_alpha.view(-1,1,1,1) * (X_t - (1- alpha_t).view(-1,1,1,1) * eps_hat / sqrt_oneminus_alpha_bar.view(-1,1,1,1))
              X_t += sqrt_beta_t.view(-1,1,1,1) * z
              
        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images