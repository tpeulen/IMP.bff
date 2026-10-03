"""Regression contract for the contiguous native chain export."""

import numpy as np

from test_sampler import make_sampler


def test_flat_chain_matches_nested_chain_exactly():
    sampler = make_sampler("stretch", seed=17, n_steps=12)
    nested = np.asarray(sampler.get_chain(), dtype=float)
    flat = np.asarray(sampler.get_chain_flat(), dtype=float).reshape(nested.shape)
    np.testing.assert_array_equal(flat, nested)


def test_empty_chain_properties_keep_the_legacy_empty_shape():
    sampler = __import__("IMP.bff", fromlist=["MCMCSampler"]).MCMCSampler("stretch", 17)
    assert np.asarray(sampler.get_chain_flat()).shape == (0,)
    assert np.asarray(sampler.chain).shape == (0,)
    sampler.set_parameter_ports([])
    sampler.reset()
    assert np.asarray(sampler.chain).shape == (0,)


def test_segmented_thinned_chain_keeps_walker_major_order_after_reset():
    sampler = make_sampler("stretch", seed=19, n_steps=6, thin=2)
    first = np.asarray(sampler.chain).copy()
    assert first.shape[1] == 2
    sampler.run(4, 2)
    combined = np.asarray(sampler.chain)
    assert combined.shape == (first.shape[0] + 2 * len(sampler.walkers), 2)
    np.testing.assert_array_equal(
        combined, np.asarray(sampler.get_chain_flat()).reshape(combined.shape)
    )
    sampler.reset()
    assert np.asarray(sampler.chain).shape == (0,)
