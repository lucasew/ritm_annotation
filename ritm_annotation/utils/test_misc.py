import itertools  # noqa: F401
import pytest  # noqa: F401
from ritm_annotation.utils.misc import incrf, try_tqdm


def test_incrf():
    counter = incrf()
    assert next(counter) == 1
    assert next(counter) == 2
    assert next(counter) == 3


def test_try_tqdm_basic():
    items = [1, 2, 3]
    result = try_tqdm(items, desc="Testing")
    assert list(result) == items


def test_try_tqdm_kwargs_support():
    items = range(5)
    result = try_tqdm(items, desc="Testing", leave=False)
    assert list(result) == list(items)
