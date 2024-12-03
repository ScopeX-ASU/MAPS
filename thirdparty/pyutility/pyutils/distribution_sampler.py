"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 01:20:26
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 01:20:26
"""

import torch

__all__ = ["SphereDistribution"]


class SphereDistribution(torch.distributions.multivariate_normal.MultivariateNormal):
    def __init__(self, radius, loc, *args, **kwargs):
        super(SphereDistribution, self).__init__(loc, *args, **kwargs)
        self.radius = radius
        self.loc = loc
        self.mvg_sample = super().sample

    def sample(self):
        p = self.mvg_sample()
        p = p.sub_(self.loc).mul_(self.radius / p.data.norm(p=2)).add_(self.loc)
        return p


if __name__ == "__main__":
    N = 64
    s2 = SphereDistribution(1, torch.zeros(N), torch.eye(N))
    print(s2.sample(), s2.sample().norm(p=2))
