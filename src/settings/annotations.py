"""Annotation-related defaults and helpers."""

DEFAULTS = {
    'export_format': 'xlsx',
    'include_camera_column': True,
}


def get_defaults():
    return dict(DEFAULTS)


def describe():
    return [
        ('export_format', 'Default export format: xlsx or csv.'),
        ('include_camera_column', 'Include camera column in export (bool).'),
    ]
