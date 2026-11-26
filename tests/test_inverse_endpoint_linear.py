"""Tests for spectrum direction, the fixed physical band, and independent crop settings."""

import unittest

import torch

from dvd_scale_free.dvd_scale_free import InverseConfig, make_curriculum
from dvd_scale_free.dvd_scale_free.endpoint_linear_fast import EndpointLinearTrainingTransform
from dvd_scale_free.dvd_scale_free.rrc import RRCConfig, training_rrc
from dvd_scale_free.datasets.dataset_loader import SupervisedLearningDataset
from scripts.main import build_dvd_transformer, describe_dvd, parser


class SpectrumControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.x = torch.rand(2, 3, 128, 160, generator=torch.Generator().manual_seed(42))

    def test_independent_frequency_and_amplitude_flips(self):
        self.assertIsInstance(make_curriculum(), EndpointLinearTrainingTransform)
        patterns = ((True, False), (False, True), (True, True))
        plots = []
        for f,a in patterns:
            dvd = make_curriculum(invert_frequency=f, invert_amplitude=a)
            early, first = dvd(self.x, 0, return_diagnostics=True)
            later, second = dvd(self.x, 64, return_diagnostics=True)
            plots.append(early)
            self.assertEqual(first[0]["frequency_floor"], f)
            self.assertEqual(first[0]["amplitude_ceiling"], a)
            self.assertTrue(torch.isfinite(later).all())
            self.assertEqual(second[0]["frequency_boundary_cpd"] < first[0]["frequency_boundary_cpd"], f)
            self.assertEqual(second[0]["amplitude_boundary"] > first[0]["amplitude_boundary"], a)
            self.assertIs(dvd(self.x, 300), self.x)
        for i in range(len(plots)):
            for j in range(i):
                self.assertGreater(float((plots[i]-plots[j]).abs().max()), 1e-6)

    def test_resolution_band_is_fixed_and_nyquist_capped(self):
        dvd = make_curriculum(invert_frequency=True, invert_amplitude=True)
        rows = [dvd(torch.rand(1, n, n), 5, return_diagnostics=True)[1][0] for n in (128, 256, 384)]
        self.assertAlmostEqual(rows[0]["band_limit_cpd"], .98*128/30)
        self.assertEqual(rows[1]["band_limit_cpd"], rows[2]["band_limit_cpd"])
        native = make_curriculum(invert_frequency=True, band_limit_cpd=None)
        self.assertGreater(native(torch.rand(1, 384, 384), 5, return_diagnostics=True)[1][0]["band_limit_cpd"],
                           rows[2]["band_limit_cpd"])

    def test_alpha_and_exposure_are_preserved(self):
        dvd = make_curriculum(invert_amplitude=True)
        rgba = torch.cat((self.x, self.x[:, :1]), dim=1)
        out = dvd(rgba, 32)
        self.assertEqual(out.shape, rgba.shape)
        torch.testing.assert_close(out[:, 3:], rgba[:, 3:], atol=0, rtol=0)
        flat = torch.full((3, 80, 80), .45)
        torch.testing.assert_close(dvd(flat, 32), flat, atol=1e-6, rtol=0)
        with self.assertRaises(ValueError):
            InverseConfig(amplitude_quantile=1.)

    def test_rrc_is_off_by_default_and_not_used_in_validation(self):
        from kornia.augmentation import RandomResizedCrop
        self.assertIsNone(training_rrc(256, RRCConfig()))
        crop = RRCConfig(enabled=True, scale=(.14,.27), ratio=(.9,1.1))
        self.assertIsNone(training_rrc(256, crop, train=False))
        r = training_rrc(256,crop)
        self.assertIsInstance(r, RandomResizedCrop)
        self.assertEqual(tuple(map(float,r._param_generator.scale)), crop.scale)
        self.assertEqual(tuple(map(float,r._param_generator.ratio)), crop.ratio)
        self.assertEqual(r.p, 1.)
        self.assertFalse(r.same_on_batch)
        with self.assertRaises(ValueError):
            RRCConfig(enabled=True, scale=(.8,.2))

    def test_cli_keeps_crop_separate_from_filter_and_records_the_range(self):
        argv = ["--development_strategy","dvd","--invert-frequency","--invert-amplitude"]
        a = parser.parse_args(argv)
        default = build_dvd_transformer(a)
        self.assertFalse(default.config.apply_color)
        self.assertNotIn("rrc", describe_dvd(a,default))
        b = parser.parse_args([*argv,"--rrc","--rrc-scale",".08","1","--rrc-ratio",".8","1.25"])
        new = build_dvd_transformer(b)
        self.assertEqual(default.config,new.config)
        desc = describe_dvd(b,new)
        self.assertEqual(desc["grayscale_aug"],0)
        self.assertEqual(desc["rrc"],dict(enabled=True,scale=(.08,1.),ratio=(.8,1.25)))
        ds = SupervisedLearningDataset(".",b)
        train = ds.get_supervised_pipeline_transform(train=True)
        valid = ds.get_supervised_pipeline_transform(train=False)
        count = lambda p: sum(type(m).__name__ == "RandomResizedCrop" for m in p.modules())
        self.assertEqual(count(train),1)
        self.assertEqual(count(valid),0)


if __name__ == "__main__":
    unittest.main()
