"""Playback settings and helpers."""

DEFAULTS = {
    'playback_rate': 1.0,
    'cache_size': 120,
    'auto_prefetch': True,
}


def get_defaults():
    return dict(DEFAULTS)


def describe():
    return [
        ('playback_rate', 'Default playback rate on start.'),
        ('cache_size', 'Frames cached per camera (int).'),
        ('auto_prefetch', 'Whether to prefetch frames at load (bool).'),
    ]
