"""
Distributions and ranges
"""

import math
import numpy as np
import scipy.stats

class DistBase:
    def __init__(self):
        pass

    def __contains__(self, item):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class Range(DistBase):
    def __init__(self, lower, upper):
        self.__lower = lower
        self.__upper = upper

    def __contains__(self, item):
        return bool((item >= self.__lower) and (item <= self.__upper))

    def sample(self):
        v = scipy.stats.uniform.rvs()
        return self.__lower + v * (self.__upper - self.__lower)


class Normal(DistBase):
    def __init__(self, mu, sigma, clip_mu=None):
        self.__mu = mu
        self.__sigma = sigma
        self.__clip_mu = clip_mu

    def clip_mu(self, clip_val):
        self.__clip_mu = clip_val
        return self

    def __contains__(self, item):
        """Returns true if the item is closer to the mean than the sampled item."""
        v = scipy.stats.norm.rvs(loc=self.__mu, scale=self.__sigma)
        v_delta = abs(v - self.__mu)
        if self.__clip_mu is not None:
            if v_delta > self.__clip_mu:
                v_delta = self.__clip_mu
        item_delta = abs(item - self.__mu)
        return bool(item_delta <= v_delta)


    def sample(self, attempt_limit=10_000):
        v = None
        for i in range(attempt_limit):
            v = scipy.stats.norm.rvs(loc=self.__mu, scale=self.__sigma)
            if self.__clip_mu is None:
                return v
            else:
                v_delta = abs(v - self.__mu)
                if v_delta <= self.__clip_mu:
                    return v

        raise RuntimeError(f"Sampling failed after {attempt_limit} attempts")
