import torch
from .dists import Mixture, Gamma
class PoissonMixtureModel(Mixture):
    def __init__(self,alpha_0,beta_0):
        dist = Gamma(alpha_0,beta_0).to_event(1)
        super().__init__(dist)

