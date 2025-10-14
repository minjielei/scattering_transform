"""
Transforms performed on angles l1, l1p, l2
"""

import numpy as np
import torch

class FourierAngle:
    """
    Perform a Fourier transform along angles l1, l1p, l2.
    """
    def __init__(self):
        self.F = None

    def fft(self, s_cov, idx_info, axis='all', if_isotropic=False):
        '''
        do an angular fourier transform on 
        axis = 'all' or 'l1'
        '''
        cov_type, j1, a, b, l1, l2, l3 = idx_info.T

        L = l3.max() + 1  # number of angles # TODO. Hack, should better infer the value of L

        # computes Fourier along angles
        C01re = s_cov[:, cov_type == 'C01re'].reshape(len(s_cov), -1, L, L)
        C01im = s_cov[:, cov_type == 'C01im'].reshape(len(s_cov), -1, L, L)
        C11re = s_cov[:, cov_type == 'C11re'].reshape(len(s_cov), -1, L, L, L)
        C11im = s_cov[:, cov_type == 'C11im'].reshape(len(s_cov), -1, L, L, L)
        if axis == 'all':
            C01 = C01re + 1j * C01im
            C01_fp = torch.fft.fftn(C01, norm='ortho', dim=(-2,-1)) 
            
            C11 = C11re + 1j * C11im
            C11_fp = torch.fft.fftn(C11, norm='ortho', dim=(-3,-2,-1)) 
        if axis == 'l1':
            C01_fp = torch.fft.fftn(C01re + 1j * C01im, norm='ortho', dim=(-2))
            C11_fp = torch.fft.fftn(C11re + 1j * C11im, norm='ortho', dim=(-3))

        # idx_info for mean, P00, S1
        cov_no_fourier = s_cov[:, np.isin(cov_type, ['mean', 'P00', 'S1'])]
        idx_info_no_fourier = idx_info[np.isin(cov_type, ['mean', 'P00', 'S1']), :]
        
        # idx_info for C01
        C01_f_flattened = torch.cat([C01_fp.real.reshape(len(s_cov), -1), C01_fp.imag.reshape(len(s_cov), -1)], dim=-1)
        idx_info_C01 = idx_info[np.isin(cov_type, ['C01re', 'C01im']), :]

        # idx_info for C11
        C11_f_flattened = torch.cat([C11_fp.real.reshape(len(s_cov), -1), C11_fp.imag.reshape(len(s_cov), -1)], dim=-1)
        idx_info_C11 = idx_info[np.isin(cov_type, ['C11re', 'C11im']), :]

        idx_info_f = np.concatenate([idx_info_no_fourier, 
#                                      idx_info_P00, idx_info_S1, 
                                     idx_info_C01, idx_info_C11])
        s_covs_f = torch.cat([cov_no_fourier, 
#                               P00_f_flattened, S1_f_flattened, 
                              C01_f_flattened, C11_f_flattened], dim=-1)

        return s_covs_f, idx_info_f
    
    def ifft(self, s_cov_f, idx_info_f, axis='all', if_isotropic=False):
        '''
        do an inverse angular fourier transform on 
        axis = 'all' or 'l1'
        '''
        cov_type, j1, a, b, l1, l2, l3 = idx_info_f.T

        L = l3.max() + 1
        
        # computes Fourier along angles
        C01re_f = s_cov_f[:, cov_type == 'C01re'].reshape(len(s_cov_f), -1, L, L)
        C01im_f = s_cov_f[:, cov_type == 'C01im'].reshape(len(s_cov_f), -1, L, L)
        C11re_f = s_cov_f[:, cov_type == 'C11re'].reshape(len(s_cov_f), -1, L, L, L)
        C11im_f = s_cov_f[:, cov_type == 'C11im'].reshape(len(s_cov_f), -1, L, L, L)
        if axis == 'all':
            C01f = C01re_f + 1j * C01im_f
            C01 = torch.fft.ifftn(C01f, norm='ortho', dim=(-2,-1)) 
            
            C11f = C11re_f + 1j * C11im_f
            C11 = torch.fft.ifftn(C11f, norm='ortho', dim=(-3,-2,-1)) 
        if axis == 'l1':
            C01 = torch.fft.ifftn(C01re_f + 1j * C01im_f, norm='ortho', dim=(-2))
            C11 = torch.fft.ifftn(C11re_f + 1j * C11im_f, norm='ortho', dim=(-3))

        # idx_info for mean, P00, S1
        cov_no_fourier = s_cov_f[:, np.isin(cov_type, ['mean', 'P00', 'S1'])]
        idx_info_no_fourier = idx_info_f[np.isin(cov_type, ['mean', 'P00', 'S1']), :]
        
        # idx_info for C01
        C01_f_flattened = torch.cat([C01.real.reshape(len(s_cov_f), -1), C01.imag.reshape(len(s_cov_f), -1)], dim=-1)
        idx_info_C01 = idx_info_f[np.isin(cov_type, ['C01re', 'C01im']), :]

        # idx_info for C11
        C11_f_flattened = torch.cat([C11.real.reshape(len(s_cov_f), -1), C11.imag.reshape(len(s_cov_f), -1)], dim=-1)
        idx_info_C11 = idx_info_f[np.isin(cov_type, ['C11re', 'C11im']), :]

        idx_info = np.concatenate([idx_info_no_fourier, 
#                                      idx_info_P00, idx_info_S1, 
                                     idx_info_C01, idx_info_C11])
        s_covs = torch.cat([cov_no_fourier, 
#                               P00_f_flattened, S1_f_flattened, 
                              C01_f_flattened, C11_f_flattened], dim=-1)

        return s_covs, idx_info