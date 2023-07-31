# flake8: noqa

try:
    from ._get_dist_maps import get_dist_maps
except ImportError:
    import pyximport

    pyximport.install(pyximport=True, language_level=3)
    # noinspection PyUnresolvedReferences
    from ._get_dist_maps import get_dist_maps
