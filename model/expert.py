import torch.nn as nn
import torch

def tanh_range(l=0.5, r=2.0):
    def get_activation(left, right):
        def activation(x):
            return ((torch.tanh(x) * 0.5 + 0.5) * (right - left) + left)
        return activation
    return get_activation(l, r)

class E1(nn.Module):
    def __init__(self):
        super(E1, self).__init__()
    def forward(self, x,inputs):
        outputs = torch.zeros_like(inputs)
        for j in range(3):
            outputs[j] = x[j]*inputs[j]
        output = (outputs - torch.min(outputs)) / (torch.max(outputs) - torch.min(outputs))
        return outputs

class E2(nn.Module):
    def __init__(self):
        super(E2, self).__init__()
    def forward(self, x,inputs):
        inputs = inputs.unsqueeze(0)
        x = x.unsqueeze(0)
        param = torch.zeros((1,3,3),device="cuda")
        param[:,0,:] = x[:,3]
        param[:,1,:] = x[:,3:6]
        param[:,2,:] = x[:,6:9]
        outputs = torch.einsum('bchw,bcj->bjhw', inputs, param)
        outputs = outputs.squeeze(0)
        output = (outputs - torch.min(outputs)) / (torch.max(outputs)-torch.min(outputs))
        return outputs



class E3(nn.Module):
    def __init__(self):
        super(E3, self).__init__()
    def forward(self, x):
        return x

class E4(nn.Module):
    def __init__(self):
        super(E4, self).__init__()
        self.gamma_range = [1., 4.]
    def forward(self, x,inputs):
        x = tanh_range(self.gamma_range[0], self.gamma_range[1])(x)
        outputs = inputs ** (1.0/x)
        output = torch.clamp(outputs,0,1)
        return output

class E5(nn.Module):
    def __init__(self):
        super(E5, self).__init__()
        self.gamma_range = [6., 10.5]
    def forward(self, x,inputs):
        x = tanh_range(self.gamma_range[0], self.gamma_range[1])(x)
        outputs = inputs **(1.0/x)
        output = torch.clamp(outputs,0,1)
        return output
