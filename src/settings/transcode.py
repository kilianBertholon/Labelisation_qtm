"""Transcoding and ffmpeg related settings."""

DEFAULTS = {
    'use_hwaccel': False,
    'ffmpeg_path': 'ffmpeg',
    'keep_transcoded': False,
}


def get_defaults():
    return dict(DEFAULTS)


def describe():
    return [
        ('use_hwaccel', 'Use hardware-accelerated ffmpeg flags when available.'),
        ('ffmpeg_path', 'Path or command name for ffmpeg.'),
        ('keep_transcoded', 'Keep transcoded files instead of deleting them.'),
    ]
