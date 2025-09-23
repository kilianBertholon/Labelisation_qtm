"""Catalog of settings modules and helpers.

This module centralizes access to all settings categories and exposes a small
API to enumerate and fetch defaults and descriptions.
"""
from . import display, playback, transcode, annotations

MODULES = {
    'display': display,
    'playback': playback,
    'transcode': transcode,
    'annotations': annotations,
}


def list_categories():
    return list(MODULES.keys())


def get_defaults(category=None):
    if category is None:
        out = {}
        for k, m in MODULES.items():
            out[k] = m.get_defaults()
        return out
    m = MODULES.get(category)
    if m is None:
        return None
    return m.get_defaults()


def describe(category=None):
    if category is None:
        out = {}
        for k, m in MODULES.items():
            out[k] = m.describe()
        return out
    m = MODULES.get(category)
    if m is None:
        return None
    return m.describe()
