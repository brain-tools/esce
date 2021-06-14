"""This module provides unit tests for the gram matrix computation."""

from unittest import TestCase

import h5py
import numpy as np

from esce.models import (
    GRAM_PATH,
    KernelType,
    get_gram,
    get_gram_triu,
    get_gram_triu_key,
    setup_cache_file,
)


class TestGram(TestCase):
    """Provide test cases for gram matrix computation."""

    def test_gram_triu(self):
        """Test if triangular upper gram matrix computation works correctly."""
        setup_cache_file()
        num_samples = 100
        X = np.random.random((num_samples, 100))
        triu = get_gram_triu(X, KernelType.RBF, gamma=0.5)
        num_elements = num_samples * (num_samples + 1) / 2
        self.assertEqual(len(triu), num_elements)

    def test_gram(self):
        """Test if full gram matrix computation works."""
        setup_cache_file()
        num_samples = 100
        X = np.random.random((num_samples, 100))
        gram = get_gram(X, KernelType.RBF, gamma=0.5)
        self.assertEqual(gram.shape, (num_samples, num_samples))

    def test_gram_cache(self):
        """Test if a gram matrix can be retrieved from a cache file."""
        setup_cache_file()
        num_samples = 100
        X = np.random.random((num_samples, 100)).astype("f")
        get_gram(X, KernelType.SIGMOID, gamma=0.8, cache=True)

        key = get_gram_triu_key(X, KernelType.SIGMOID, gamma=0.8)
        with h5py.File(GRAM_PATH, "r") as f:
            found = key in f
        self.assertTrue(found)
