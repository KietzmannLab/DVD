"""Numerical and API checks for the reference and batched Endpoint Linear paths."""

import math
import unittest

import torch

from dvd_scale_free.dvd_scale_free.endpoint_linear import (
    EndpointConfig, EndpointLinearDVD, REFERENCE_ADULT_CPD, contrast_sensitivity,
)
from dvd_scale_free.dvd_scale_free.endpoint_linear_fast import EndpointLinearTrainingTransform


class EndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.x = torch.rand((3, 64, 96), generator=torch.Generator().manual_seed(28))

    def test_speed_changes_only_contrast_progress(self):
        self.assertEqual(EndpointConfig().speed, 1)
        for age, expected in ((0, 0), (300, 1), (310, 1)):
            self.assertEqual(contrast_sensitivity(age), expected)
        progress = []
        for v in (.001, .5, 1., 4., 150.):
            _, rows = EndpointLinearDVD(EndpointConfig(speed=v))(self.x, 48, return_diagnostics=True)
            self.assertAlmostEqual(rows[0]["contrast_progress"], contrast_sensitivity(48)**(1/v))
            progress.append(rows[0]["contrast_progress"])
        self.assertTrue(all(a < b for a, b in zip(progress, progress[1:])))
        fixed = EndpointConfig(speed=150, apply_contrast=False)
        torch.testing.assert_close(EndpointLinearDVD(fixed)(self.x, 48),
                                   EndpointLinearDVD(EndpointConfig(apply_contrast=False))(self.x, 48),
                                   rtol=0, atol=0)

    def test_geometry_fixed_adult_and_nyquist_override(self):
        x = torch.rand(1, 224, 224)
        for width in (224, 256, 448):
            _, rows = EndpointLinearDVD(EndpointConfig(image_size=width))(x, 48, return_diagnostics=True)
            self.assertAlmostEqual(rows[0]["adult_anchor_cpd"], REFERENCE_ADULT_CPD)
        for override, expected in ((None, .98 * 128 / 30), (4., 4.), ("nyquist", .98 * 128 / 30)):
            cfg = EndpointConfig(image_size=128, adult_anchor_cpd=override)
            _, rows = EndpointLinearDVD(cfg)(x, 48, return_diagnostics=True)
            self.assertAlmostEqual(rows[0]["adult_anchor_cpd"], expected)
            self.assertEqual(rows[0]["adult_anchor_clamped"], override is None)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            EndpointLinearDVD(EndpointConfig(image_size=128, adult_out_of_band="error"))(x, 48)

    def test_native_rectangular_long_edge_and_rgba(self):
        alpha = torch.linspace(0, 1, self.x.shape[-1]).repeat(1, self.x.shape[-2], 1)
        rgba = torch.cat((self.x, alpha))
        native, rows = EndpointLinearDVD()(rgba, 48, return_diagnostics=True)
        torch.testing.assert_close(native[3:], alpha, atol=0, rtol=0)
        self.assertEqual(tuple(native.shape), (4, 64, 96))
        self.assertEqual(rows[0]["vertical_fov_deg"], 10.)
        resized, rows = EndpointLinearDVD(EndpointConfig(image_size=144))(rgba, 48, return_diagnostics=True)
        self.assertEqual(tuple(resized.shape), (4, 96, 144))
        self.assertEqual(rows[0]["vertical_fov_deg"], 10.)

    def test_training_endpoint_is_not_demo_identity(self):
        filtered = EndpointLinearDVD()(self.x, 300)
        slower = EndpointLinearDVD(EndpointConfig(speed=.5))(self.x, 300)
        torch.testing.assert_close(filtered, slower, atol=0, rtol=0)
        self.assertGreater((filtered - self.x).abs().max(), 1e-4)
        demo = EndpointLinearDVD(EndpointConfig(unfiltered_at_300=True))
        torch.testing.assert_close(demo(self.x, 299), EndpointLinearDVD()(self.x, 299), atol=0, rtol=0)
        for age in (300, 301):
            actual, rows = demo(self.x, age, return_diagnostics=True)
            torch.testing.assert_close(actual, self.x, atol=0, rtol=0)
            self.assertTrue(rows[0]["unfiltered"])

    def test_low_frequency_policy(self):
        for preset, policy, flag in ((.02, "extrapolate", True), (.02, "clamp", False),
                                     ("fundamental", "extrapolate", False)):
            cfg = EndpointConfig(infant_anchor_cpd=preset, low_frequency_policy=policy)
            output, rows = EndpointLinearDVD(cfg)(self.x, 48, return_diagnostics=True)
            self.assertEqual(rows[0]["infant_anchor_extrapolated"], flag)
            self.assertTrue(torch.isfinite(output).all())

    def test_mean_and_flat_grayscale(self):
        x = torch.linspace(0, 1, 96).repeat(1, 64, 1)
        raw = EndpointLinearDVD(EndpointConfig(apply_color=False, clamp_output=False))(x, 72)
        torch.testing.assert_close(x.mean(), raw.mean(), atol=2e-6, rtol=0)
        flat = x.new_full((1, 64, 96), .5)
        torch.testing.assert_close(EndpointLinearDVD()(flat, 1), flat, atol=2e-6, rtol=0)

    def test_container_and_half_precision_contract(self):
        transform = EndpointLinearDVD()
        y, d = transform((self.x, self.x[:1].to(torch.float16)), 16, return_diagnostics=True)
        self.assertIsInstance(y, tuple)
        self.assertEqual(y[1].dtype, torch.float16)
        self.assertEqual(y[0].shape, self.x.shape)
        self.assertEqual(len(d), 2)
        batched, meta = transform(torch.stack((self.x, self.x)), 16, return_diagnostics=True)
        torch.testing.assert_close(batched[0], y[0], atol=2e-5, rtol=1e-5)
        self.assertEqual(len(meta), 2)

    def test_bad_options_fail_early(self):
        for values in ({"speed": 0}, {"speed": math.nan}, {"image_size": (256,)},
                       {"fov_deg": -1}, {"adult_anchor_cpd": "wrong"}, {"sigmoid_width_log10": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                EndpointConfig(**values)
        with self.assertRaises(ValueError):
            EndpointLinearDVD()(self.x, -1)
        with self.assertRaises(TypeError):
            EndpointLinearDVD()(self.x.to(torch.uint8), 4)

    def test_batched_matches_reference_on_even_and_odd_grids(self):
        generator = torch.Generator().manual_seed(20)
        for channels, side, speed in ((3, 256, 1.), (1, 255, .5), (3, 224, 1.)):
            x = torch.rand((3, channels, side, side), generator=generator)
            x[0].fill_(.5)
            cfg = EndpointConfig(speed=speed)
            fast = EndpointLinearTrainingTransform(cfg, backend="batched", chunk=2)
            reference = EndpointLinearDVD(cfg)
            self.assertEqual(fast.backend_for(x), "batched")
            for age in (0., 9., 48., 299.9, 300.):
                with self.subTest(side=side, age=age, speed=speed):
                    torch.testing.assert_close(fast(x, age), reference(x, age), rtol=0, atol=4e-6)

    def test_auto_falls_back_for_non_default_geometry(self):
        x = self.x.unsqueeze(0)
        cfg = EndpointConfig(adult_anchor_cpd="nyquist")
        fast = EndpointLinearTrainingTransform(cfg)
        self.assertEqual(fast.backend_for(x), "reference")
        torch.testing.assert_close(fast(x, 16), EndpointLinearDVD(cfg)(x, 16), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "default physical geometry"):
            EndpointLinearTrainingTransform(cfg, backend="batched")
        with self.assertRaisesRegex(ValueError, "float32 NCHW"):
            EndpointLinearTrainingTransform(backend="batched")(x, 16)


if __name__ == "__main__":
    unittest.main()
