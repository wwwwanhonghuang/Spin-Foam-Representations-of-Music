# __init__.py

from music.midi.pitch import (
    build_midi_note_map,
    build_reverse_map
)

# Precomputed global maps
MIDI_NOTE_MAP_SHARP = build_midi_note_map(use_flats=False)
MIDI_NOTE_MAP_FLAT  = build_midi_note_map(use_flats=True)

MIDI_NAME_TO_NUMBER_SHARP = build_reverse_map(MIDI_NOTE_MAP_SHARP)
MIDI_NAME_TO_NUMBER_FLAT  = build_reverse_map(MIDI_NOTE_MAP_FLAT)

__all__ = [
    "MIDI_NOTE_MAP_SHARP",
    "MIDI_NOTE_MAP_FLAT",
    "MIDI_NAME_TO_NUMBER_SHARP",
    "MIDI_NAME_TO_NUMBER_FLAT",
]
