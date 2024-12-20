# importing 
import torch
import torch.nn as nn
import math

import warnings
warnings.filterwarnings('ignore')

# function to calculate the layer-direction mean and variance. 
def calculate_mean_std_for_forward(inputs, cal_dim_count, std = True) :    
    input_shape = inputs.shape
    if cal_dim_count < 1 : 
        cal_dim = [i for i in range(1, len(input_shape))]
    else : 
        if cal_dim_count > len(input_shape) - 1 :
            raise Exception('"Cal_dim_count" of LayerAct should be smaller than dimension of input')
        cal_dim = [-1*i for i in range(1, cal_dim_count+1)][::-1]
        #print(cal_dim)

    mean = inputs.mean(dim=cal_dim, keepdim=True)
    if std : 
        var = inputs.var(dim=cal_dim, keepdim=True)
        return mean, var, cal_dim
    else : 
        return mean, cal_dim

#############################################################

class LA_SiLU(nn.Module) : 
    """
    # alpha 
    - float 
    - the parameter for stability of activation 
    # save_less 
    - bool
    - if true, do not save mean, variance, standard deviation, and normalized input for "backward" by ctx.save_for_backward()
    - if false, save mean, variance, standard deviation, and normalized input for "backward" by ctx.save_for_backward()
    """
    def __init__(self, alpha=1e-5, cal_dim_count=0, save_less=False) : 
        super(LA_SiLU, self).__init__()
        self.alpha = alpha
        self.cal_dim_count = cal_dim_count
        self.save_less = save_less

    def forward(self, inputs) : 
        return la_silu.apply(inputs, self.alpha, self.cal_dim_count, self.save_less, self.training)

class la_silu(torch.autograd.Function) : 
    @staticmethod
    def forward(ctx, inputs, alpha, cal_dim_count, save_less, training=True) : 
        mean, var, cal_dim = calculate_mean_std_for_forward(inputs, cal_dim_count)

        #if save_less or not training : 
        #    z = torch.mul(torch.sigmoid(torch.div(torch.sub(inputs, mean), torch.sqrt(var+alpha))), inputs)
        #else : 

        var_ = var+alpha
        std = torch.sqrt(var_)
        n = torch.div(torch.sub(inputs, mean), std)
        s = torch.sigmoid(n)
        z = torch.mul(s, inputs)

        ctx.save_less = save_less
        ctx.alpha = alpha
        if save_less : 
            ctx.cal_dim_count = cal_dim_count
            ctx.save_for_backward(inputs)
        else : 
            ctx.save_for_backward(inputs, mean, var, std, n, s)
            ctx.cal_dim = cal_dim

        return z

    @staticmethod
    def backward(ctx, output_grad):
        alpha = ctx.alpha
        if ctx.save_less : 
            inputs, = ctx.saved_tensors
            cal_dim_count = ctx.cal_dim_count
            mean, var, cal_dim = calculate_mean_std_for_forward(inputs, cal_dim_count)
            std = torch.sqrt(var+alpha)
            n = torch.div(torch.sub(inputs, mean), std)
            s = torch.sigmoid(n)
        else : 
            inputs, mean, var, std, n, s = ctx.saved_tensors
            cal_dim = ctx.cal_dim

        inputs_grad = torch.mul(output_grad.clone(), s)
        dn = torch.div(
                torch.mul(
                    torch.mul(output_grad.clone(), inputs.clone()), 
                    torch.mul(s, 1-s)
                    ), 
                std
            )
        dn = torch.sub(
                dn, 
                torch.add(
                    torch.mean(dn, dim=cal_dim, keepdim=True), 
                    torch.mul(torch.mean(torch.mul(dn, n), dim=cal_dim, keepdim=True), n)
                    )
            )

        inputs_grad = torch.add(inputs_grad, dn)

        return inputs_grad, None, None, None, None
    

#############################################################
#############################################################
#############################################################


class LA_HardSiLU(nn.Module) : 
    def __init__(self, alpha=1e-5, cal_dim_count=0, save_less=False) : 
        super(LA_HardSiLU, self).__init__()
        self.alpha = alpha
        self.save_less = save_less

    def forward(self, inputs) : 
        return la_hardsilu.apply(inputs, self.alpha, self.save_less, self.training)


class la_hardsilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, alpha, save_less, training=True):
        shape = inputs.shape
        device = inputs.device
        
        ones = torch.ones(shape, device=device)
        zeros = torch.zeros(shape, device=device)

        mean, var, cal_dim = calculate_mean_std_for_forward(inputs)

        if save_less or not training : 
            n = torch.div(torch.sub(inputs, mean), torch.sqrt(var+alpha))
            z = torch.mul(inputs, torch.where(n<=3, torch.where(n<=-3, zeros.clone(), n/6+0.5), ones.clone()))
        else : 
            var_ = var+alpha
            std = torch.sqrt(var_)
            n = torch.div(torch.sub(inputs, mean), std)
            s = torch.where(n<=-3, zeros.clone(), n/6+0.5)
            s = torch.where(n<=3, s, ones.clone())
            z = torch.mul(inputs, s)

        if training : 
            ctx.save_less = save_less
            if save_less : 
                ctx.save_for_backward(inputs)
                ctx.alpha = alpha
            else : 
                ctx.save_for_backward(inputs, mean, std, n, s)
            ctx.cal_dim = cal_dim

        return z

    @staticmethod
    def backward(ctx, output_grad):
        if ctx.save_less : 
            inputs, = ctx.saved_tensors
            shape = inputs.shape
            device = inputs.device
            ones = torch.ones(shape, device=device)
            zeros = torch.zeros(shape, device=device)

            alpha = ctx.alpha
            mean, var, cal_dim = calculate_mean_std_for_forward(inputs)
            std = torch.sqrt(var+alpha)
            n = torch.div(torch.sub(inputs, mean), std)
            s = torch.where(
                n<=3, 
                torch.where(n<=-3, zeros.clone(), n/6+0.5), 
                ones.clone()
                )
        else : 
            cal_dim = ctx.cal_dim
            inputs, mean, std, n, s = ctx.saved_tensors
            shape = inputs.shape
            device = inputs.device
            ones = torch.ones(shape, device=device)
            zeros = torch.zeros(shape, device=device)

        inputs_grad = torch.mul(output_grad.clone(), s)
        ds = torch.where(
            n<=3, 
            torch.where(n<=-3, zeros.clone(), ones.clone()/6), 
            zeros.clone()
            )
        da = torch.mul(output_grad.clone(), inputs.clone())
        dn = torch.div(torch.mul(da, ds), std)
        dn = torch.sub(
            dn, 
            torch.add(
                torch.mean(dn, dim=cal_dim, keepdim=True), 
                torch.mul(torch.mean(torch.mul(dn, n), dim=cal_dim, keepdim=True), n)
                )
            )

        inputs_grad = torch.add(inputs_grad, dn)

        return inputs_grad, None, None, None
    

#############################################################