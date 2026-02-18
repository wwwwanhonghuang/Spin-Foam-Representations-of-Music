# utils.py

from music.midi import MIDI_NOTE_MAP_SHARP, MIDI_NOTE_MAP_FLAT


def note_name(note: int, use_flats: bool = False) -> str:
    """
    Convert MIDI note number (0–127) to scientific pitch notation.

    Parameters
    ----------
    note : int
        MIDI pitch number.
    use_flats : bool
        If True, use flat naming (Db instead of C#).

    Returns
    -------
    str
        Note name like C4, F#3, Bb5.
    """
    mapping = MIDI_NOTE_MAP_FLAT if use_flats else MIDI_NOTE_MAP_SHARP
    return mapping.get(note, f"?{note}")
