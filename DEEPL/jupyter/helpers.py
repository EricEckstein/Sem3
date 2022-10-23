from calendar import c
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from typing import Union
from copy import deepcopy
from typing import Callable

sns.set_theme()
sns.set_style("darkgrid")

### Plotting functions
def plot_fun(x: Union[torch.Tensor, float], f: Callable) -> None:
    # plot function
    fx = f(x)
    plt.plot(x,f(x))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    
    # plot coord
    border = 0.1*(max(fx) - min(fx))
    plt.ylim([min(fx) - border, max(fx) + border])
    plt.xlim([min(x), max(x)])
    plt.plot(x, x*0, 'k')
    plt.plot([0,0],[min(fx) - border,max(fx) + border] , 'k')

    
def plot_tangent(x_0: Union[torch.Tensor, float], 
                 x: Union[torch.Tensor, float],
                 f:Callable,
                 df:Callable, 
                 alpha: float=1) -> None:
    dfx_0 = df(x_0)
    fx_0 = f(x_0)
    g = dfx_0*(x - x_0) + fx_0
    
    # plot tangent
    plt.plot(x, g, color='orange', alpha=alpha)

def plot_indicator(x_0: Union[torch.Tensor, float], 
                   f:Callable, 
                   text: bool=True) -> Union[torch.Tensor, float]:
    # indicate function value 
    fx_0 = f(x_0)
    plt.plot([x_0,x_0], [0,fx_0], 'g')
    if text:
        plt.text(x_0, fx_0, f"f({x_0})", color='green')
    plt.plot(x_0,fx_0, 'o', color='green')
    return fx_0
    
def plot_connector(x_0: float, 
                   x_1: float, 
                   fx_0: float, 
                   fx_1: float, 
                   alpha: float=0.4) -> None:
    plt.plot([x_0, x_1], [fx_0, fx_1], color='r', alpha=alpha)




# Made up functions
# Made up ground truth function
def f(x: torch.tensor) -> torch.tensor:
    f = torch.exp(0.001*x)*torch.sin(0.5*x)/x + torch.log(x) + torch.sin(0.1*x)
    return f

# This function generates the observations
def obs(x: torch.tensor) -> torch.tensor:
    fx = f(x)
    eps = torch.randn(len(x)) * 0.5
    y = fx + eps
    return y



# tracker class
# helper function to transform parameter to flattened list
def tolist(p: torch.tensor):
    return list(list(p.data.view(-1).numpy()))



# nonlinear model
class NonLinearModel(torch.nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 n_intermediate: int,
                 intermediate_dim: int, 
                 output_dim: int,
                 act_fun: nn.Module) -> None:
        super(NonLinearModel, self).__init__()
        
        # we will store all our layers/operations here
        self.layers = torch.nn.Sequential()
        
        if n_intermediate > 0:  
            # add input layer
            self.layers.append(nn.Linear(in_features=input_dim, 
                                         out_features=intermediate_dim))
        
            # add intermediate layers and activation functions
            for _ in range(n_intermediate-1):
                self.layers.append(act_fun)
                self.layers.append(nn.Linear(in_features=intermediate_dim, 
                                             out_features=intermediate_dim))
        
            # add  output layer
            self.layers.append(act_fun)
            self.layers.append(nn.Linear(in_features=intermediate_dim, 
                                         out_features=output_dim))
        else:
            self.layers.append(nn.Linear(in_features=input_dim, 
                                         out_features=output_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # execute all operations
        out = self.layers(x)
        return out


# helper class for weight tracking
class Tracking:
    def __init__(self, n_layers=100):
       self.data = []
       self.head = ['epoch', 'layer', 'bias', 'weight', 'bias_name', 'weight_name']
       self.last_epoch = -1
       self.colors = None #['b', 'g', 'r', 'c', 'm', 'y']
       self.df = None
        
    def update(self, epoch, net):
        param_list = list(net.named_parameters()) 
        for i, (a, b) in enumerate(zip(param_list[::2], param_list[1::2])):
            w_name, weight = (a[0], a[1]) if 'weight' in a[0] else (b[0],b[1])
            b_name, bias = (b[0], b[1]) if 'bias' in b[0] else (a[0],a[1])
            self.data.append([
                epoch,
                i,
                tolist(bias),
                tolist(weight),
                b_name,
                w_name
            ])
    
    def update_df(self):
        self.df = pd.DataFrame(data=self.data, columns=self.head)
            
    def get_df(self):
        self.update_df()
        return self.df
    
    def _set_colors(self, n_layers):
        self.colors = cm.rainbow(np.linspace(0, 1, n_layers))
        
        
    def plot(self, l_idx=None, alpha=0.5, overlay=False, params=['weight', 'bias'], save_at=None):
        self.update_df()
        epochs = self.df['epoch'].unique()
        layers = self.df['layer'].unique()
        if self.colors is None:
            self._set_colors(n_layers=len(layers))
            
        net_old = None 
        for ei, e in enumerate(epochs):
            print(f'\r plotting epoch {ei}', end="")
            net = self.df[self.df['epoch'] == e]
            if net_old is not None:
                for i, l in enumerate(layers):
                    if l_idx is not None:
                        if l_idx != l: continue
                    for w in params:
                        w_new = net[net['layer'] == l][w].values[0]
                        w_old = net_old[net_old['layer'] == l][w].values[0]
                        for j in range(len(w_new)):
                            shift = i if not overlay else 0
                            plt_symb = '.-'
                            if w == 'bias':
                                plt_symb = '|-'
                            plt.plot([w_old[j]+shift,w_new[j]+shift], 
                                     [epochs[ei-1],e],
                                     plt_symb,
                                     c=self.colors[i], 
                                     alpha=alpha)
            net_old = net
        plt.xlabel('layer')
        plt.ylabel('epoch')
        title = ''
        title += 'weights: •  ' if 'weight' in params else ''
        title += 'bias: | ' if 'bias' in params else ''
        plt.title(title)
        if save_at is None:
            plt.show()
        else: 
            plt.savefig(save_at)
        

    def plot_diff(self, l_idx=None, alpha=0.5, overlay=False, params=['weight', 'bias'], save_at=None):
        self.update_df()
        epochs = self.df['epoch'].unique()
        layers = self.df['layer'].unique()
        net_old = None 
        for ei, e in enumerate(epochs):
            print(f'\r plotting epoch {ei}', end="")
            net = self.df[self.df['epoch'] == e]
            if net_old is not None:
                for i, l in enumerate(layers):
                    if l_idx is not None:
                        if l_idx != l: continue
                    for w in params:
                        w_new = net[net['layer'] == l][w].values[0]
                        w_old = net_old[net_old['layer'] == l][w].values[0]
                        for j in range(len(w_new)):
                            shift = i if not overlay else 0
                            plt_symb = '.'
                            if w == 'bias':
                                plt_symb = '|'
                            plt.plot([w_old[j] - w_new[j] + shift], 
                                     [e],
                                     plt_symb,
                                     c=self.colors[i], 
                                     alpha=alpha) 
            net_old = net
        title = ''
        title += 'weights: •  ' if 'weight' in params else ''
        title += 'bias: | ' if 'bias' in params else ''
        plt.title(title)
        plt.xlabel('layer')
        plt.ylabel('epoch')
        if save_at is None:
            plt.show()
        else: 
            plt.savefig(save_at)

    
    def plot_diff_trace(self, l_idx=None, alpha=0.5):
        raise NotImplementedError()
        self.update_df()
        epochs = self.df['epoch'].unique()
        layers = self.df['layer'].unique()
        net_old = None 
        diff_old = None
        for ei, e in enumerate(epochs):
            print(f'\r plotting epoch {ei}', end="")
            net = self.df[self.df['epoch'] == e]
            if net_old is not None:
                for i, l in enumerate(layers):
                    if l_idx is not None:
                        if l_idx != l: continue
                    for w in ['weight', 'bias']:
                        w_new = net[net['layer'] == l][w].values[0]
                        w_old = net_old[net_old['layer'] == l][w].values[0]
                        diff_new = [wo - wn + i for wo, wn in zip(w_old, w_new)]
                        if diff_old is not None:
                            for j in range(len(diff_new)):
                                plt.plot([diff_old[j], diff_new[j]], 
                                         [epochs[ei-1],e],
                                         'o-',
                                         c=self.colors[i], 
                                         alpha=alpha)
                        diff_old = diff_new
                        
            net_old = net
        plt.show()
        
    def get_model_at_epoch(self, model: NonLinearModel, epoch: int):
        df = self.df[self.df['epoch'] == epoch]
        new_model = deepcopy(model)
        for n, p in new_model.named_parameters():
            if n in list(df['bias_name']):
                w = torch.tensor(list(df[df['bias_name'] == n]['bias']))
            elif n in list(df['weight_name']):
                w = torch.tensor(list(df[df['weight_name'] == n]['weight']))
            else:
                raise ValueError(f'could not find parameter with name {n}')
            w = w.reshape(p.shape)
            p.data = w
        return new_model
    
    def plot_models(self,x_data, y_data, interval, fx, model, x_model=None, color=None, alpha_scaling=0.1, save_at=None):
        if x_model is None:
            x_model = x_data
        self.update_df()
        epochs = self.df['epoch'].unique()
        alpha = np.linspace(0.1,1, len(epochs))
        for i, e in enumerate(epochs):
            new_model = self.get_model_at_epoch(model, e)
            out = new_model(x_model.view(-1,1))
            if e < epochs[-1]:
                plt.plot(x_data, out.detach().squeeze(), alpha=alpha[i]*alpha_scaling, c=color)
            else:
                plt.plot(x_data, out.detach().squeeze(), alpha=1, label='model', c=color)
            
        plt.plot(x_data, y_data, 'o', label='data')
        
        if interval is not None and fx is not None:
            plt.plot(interval, fx, label='f')
        plt.legend()
        if save_at is None:
            plt.show()
        else: 
            plt.savefig(save_at)
        
    def get_epochs(self):
        epochs = self.df['epoch'].unique()
        return epochs
        


# Implementation of the MSE loss function
def MSE(y: torch.tensor, gx: torch.tensor) -> torch.tensor:
    delta = y - gx
    delta_sq = delta * delta
    mse = delta_sq.mean()
    return mse



# plotting
# implement plotting functions
def plot_fun(x: torch.Tensor, f, title='', save_at = None) -> None:
    # plot function
    fx = f(x)
    plt.plot(x,f(x))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    
    # plot coord
    border = 0.1*(max(fx) - min(fx))
    plt.ylim([min(fx) - border, max(fx) + border])
    plt.xlim([min(x), max(x)])
    plt.plot(x, x*0, 'k')
    plt.plot([0,0],[min(fx) - border,max(fx) + border] , 'k')
    if save_at is not None:
        plt.savefig(save_at)