import torch
import math

class TaylorActivation(torch.nn.Module):
    '''
    Implementation of taylor polynomial activation function
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - degree: - degree of polynomial (2 is ax^2 + bx + c)
        - alphas - coefficients of polynomial
        - beta - centering polynomial around point
    '''

    def __init__(self, degree=4, alphas=None, beta=0.0, clip_max=10, clip_min=-10):
        '''
        Initialization.
        INPUT:
            - alphas: trainable parameter
            - beta: trainable parameter
            - degree: degree of polynomial
            alpha is initialized with 1 value by default
            beta is initialized with 0 value by default
        '''
        super(TaylorActivation, self).__init__()
        # self.in_features = in_features
        
        self.beta = torch.nn.Parameter(torch.Tensor([beta]))

        # initialize alpha
        if alphas:
            self.alphas = torch.nn.Parameter(torch.Tensor(alphas))
        else:
            self.alphas = torch.nn.Parameter(torch.normal(0.0, 0.1, size=(degree + 1,)))

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.alphas.requiresGrad = True  # set requiresGrad to true!
        self.beta.requiresGrad = True  # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''

        # Polynomial = self.alphas[0] + self.alphas[1] * (x - self.beta) / 1! + self.alphas[1] * (x - self.beta) ** 2 / 2! ...
        # print(f"\nalphas: {self.alphas}\tbeta: {self.beta}")

        val = torch.zeros(x.size()).to(x.device)
        # print(f"PRE val: {val}")

        for d in range(len(self.alphas)):
            val += (self.alphas[d] / math.factorial(d)) * (x - self.beta) ** d

        # print(f"min: {val.min()}\tmean: {val.mean()}\tmax: {val.max()}")

        clip_min = torch.empty(val.size()).fill_(self.clip_min).to(x.device)
        clip_max = torch.empty(val.size()).fill_(self.clip_max).to(x.device)

        # print(f"PRECLIP val: {val.size()}")
        # print(f"clip_min: {clip_min.size()}\t")
        # print(f"PRE: min: {val.min()}\tmean: {val.mean()}\tmax: {val.max()}")
        val = torch.max(val, clip_min)
        val = torch.min(val, clip_max)
        # print(f"POST: min: {val.min()}\tmean: {val.mean()}\tmax: {val.max()}")
        # quit()
        return val