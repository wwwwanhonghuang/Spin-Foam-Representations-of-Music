# pitch.py
from typing import Dict

# ─────────────────────────────────────────────
# Pitch Class Names
# ─────────────────────────────────────────────

NOTE_NAMES_SHARP = [
    'C', 'C#', 'D', 'D#', 'E', 'F',
    'F#', 'G', 'G#', 'A', 'A#', 'B'
]

NOTE_NAMES_FLAT = [
    'C', 'Db', 'D', 'Eb', 'E', 'F',
    'Gb', 'G', 'Ab', 'A', 'Bb', 'B'
]


# ─────────────────────────────────────────────
# Build 128 MIDI Map
# ─────────────────────────────────────────────

def build_midi_note_map(use_flats: bool = False) -> Dict[int, str]:
    names = NOTE_NAMES_FLAT if use_flats else NOTE_NAMES_SHARP
    midi_map: Dict[int, str] = {}

    for n in range(128):
        octave = (n // 12) - 1
        name = names[n % 12]
        midi_map[n] = f"{name}{octave}"

    return midi_map


def build_reverse_map(note_map: Dict[int, str]) -> Dict[str, int]:
    return {v: k for k, v in note_map.items()}
