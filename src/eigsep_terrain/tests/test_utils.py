"""Tests for eigsep_corr.io"""
import copy
import os
import pytest
import numpy as np
from eigsep_terrain.utils import calc_az_bin_range, az_bin


class TestUtils:
    def test_calc_az_bin_range(self):
        '''Unit test for calc_az_bin_range'''
        n_az = 256
        for f in (2, 4, 8):
            for e0, n0 in [(   -1, -1), (   -1, f+1), (   -1, 7*f+2),
                           (  f+1, -1), (  f+1, f+1), (  f+1, 7*f+2),
                           (7*f+2, -1), (7*f+2, f+1), (7*f+2, 7*f+2)]:
                u_max = np.zeros((4, 4))
                
                e_edges = f * np.arange(0, u_max.shape[1] + 1)
                n_edges = f * np.arange(0, u_max.shape[0] + 1)
                az_min, az_max = calc_az_bin_range(e_edges, n_edges, e0, n0, n_az=n_az)
                az_min = az_min * 360 / n_az
                az_max = az_max * 360 / n_az
                e_total = np.arange(0, f * u_max.shape[1]) - e0
                n_total = np.arange(0, f * u_max.shape[0]) - n0
                bins_total = az_bin(e_total, n_total, n_az) * 360 / n_az
                for _n in range(u_max.shape[0]):
                    for _e in range(u_max.shape[1]):
                        sub_array = bins_total[f*_n:f*(_n+1), f*_e:f*(_e+1)]
                        pred_min = az_min[_n, _e]
                        pred_max = az_max[_n, _e]
                        if az_min[_n, _e] > az_max[_n, _e]:
                            # wrap around case
                            pred_min = pred_min - 360
                            sub_array = np.where(sub_array > 180, sub_array - 360, sub_array)
