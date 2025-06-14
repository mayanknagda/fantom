"""
This module contains tests for the VAE models.
"""
import os
import torch
import sys
import unittest

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.vae_helper import Decoder
from src.models.vae_helper import DirichletEncoder, Encoder
from src.models.vae_helper import DirPathwiseGrad, DirRSVI


class TestVAE(unittest.TestCase):
    def test_decoder(self):
        decoder = Decoder(10, 20)
        self.assertEqual(decoder.dec_base.in_features, 10)
        self.assertEqual(decoder.dec_base.out_features, 20)
        self.assertEqual(decoder.bath_norm.num_features, 20)
        self.assertEqual(decoder.log_softmax.dim, 1)

    def test_encoder(self):
        encoder = DirichletEncoder(10, 20)
        self.assertEqual(encoder.enc_base[0].in_features, 10)
        self.assertEqual(encoder.enc_base[0].out_features, 512)
        self.assertEqual(encoder.net[0].in_features, 512)
        self.assertEqual(encoder.net[0].out_features, 20)
        self.assertEqual(encoder.net[1].num_features, 20)

    def test_dirichlet_encoder(self):
        encoder = DirichletEncoder(10, 20)
        self.assertEqual(encoder.enc_base[0].in_features, 10)
        self.assertEqual(encoder.enc_base[0].out_features, 512)
        self.assertEqual(encoder.net[0].in_features, 512)
        self.assertEqual(encoder.net[0].out_features, 20)
        self.assertEqual(encoder.net[1].num_features, 20)

    def test_dirpathwise_grad(self):
        dirpathwise_grad = DirPathwiseGrad()
        alpha = torch.ones(10, 20)
        prior_alpha = torch.ones(10, 20)
        sample, kl_div = dirpathwise_grad(alpha, prior_alpha)
        self.assertEqual(sample.shape, (10, 20))
        self.assertEqual(kl_div.shape, (10,))

    def test_dir_rsvi(self):
        dir_rsvi = DirRSVI()
        alpha = torch.ones(10, 20)
        prior_alpha = torch.ones(10, 20)
        sample, kl_div = dir_rsvi(alpha, prior_alpha)
        self.assertEqual(sample.shape, (10, 20))
        self.assertEqual(kl_div.shape, (10,))


if __name__ == "__main__":
    unittest.main()
