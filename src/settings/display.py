"""Display settings helpers for multicam_annotator.

Contains small functions that return default display-related settings and
helpers to update them.
"""

DEFAULTS = {
    'window_scale': 1.0,
    'fit_mode': 'keep_aspect',  # options: keep_aspect, stretch
    'show_framerate': True,
}


def get_defaults():
    """Return the default display settings as a dict."""
    return dict(DEFAULTS)


def describe():
    """Return a list of (key, description) tuples for cataloging."""
    return [
        ('window_scale', 'Factor to scale displayed video windows (float).'),
        ('fit_mode', 'How to fit video into widget: keep_aspect or stretch.'),
        ('show_framerate', 'Whether to draw the framerate overlay (bool).'),
    ]
