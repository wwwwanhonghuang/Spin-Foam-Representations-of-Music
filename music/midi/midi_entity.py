"""
MIDIEntity: a structured representation of a MIDI file.
Loadable from .mid files via pretty_midi.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

try:
    import pretty_midi
except ImportError:
    pretty_midi = None


@dataclass
class NoteEvent:
    pitch: int          # 0–127 (piano: 21–108)
    velocity: int       # 0–127
    start: float        # seconds
    end: float          # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class MIDIEntity:
    """
    Unified representation of a MIDI file used across all compilation schemes.

    Attributes:
        notes        : list of NoteEvent (all instruments flattened)
        tempo        : estimated tempo in BPM
        duration     : total duration in seconds
        piano_roll   : np.ndarray of shape (88, T) — binary activation matrix
                       rows = piano keys 21..108, columns = time frames (10ms)
        source_path  : original file path if loaded from disk
    """
    notes: List[NoteEvent] = field(default_factory=list)
    tempo: float = 120.0
    duration: float = 0.0
    piano_roll: Optional[np.ndarray] = None   # shape (88, T)
    source_path: Optional[str] = None

    @classmethod
    def from_file(cls, path: str, fs: int = 100) -> MIDIEntity:
        """
        Load a MIDIEntity from a .mid file.

        Args:
            path : path to the .mid file
            fs   : piano roll frame rate in Hz (default 100 = 10ms per frame)
        """
        if pretty_midi is None:
            raise ImportError("pretty_midi is required: pip install pretty_midi")

        pm = pretty_midi.PrettyMIDI(path)

        notes = []
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            for n in instrument.notes:
                notes.append(NoteEvent(
                    pitch=n.pitch,
                    velocity=n.velocity,
                    start=n.start,
                    end=n.end,
                ))
        notes.sort(key=lambda n: n.start)

        # Piano roll: rows = MIDI pitches 21..108 (88 keys), cols = frames
        pr_full = pm.get_piano_roll(fs=fs)  # shape (128, T)
        piano_roll = (pr_full[21:109] > 0).astype(np.float32)  # shape (88, T)

        tempo_arr, _ = pm.get_tempo_changes()
        tempo = float(tempo_arr[0]) if len(tempo_arr) > 0 else 120.0

        return cls(
            notes=notes,
            tempo=tempo,
            duration=pm.get_end_time(),
            piano_roll=piano_roll,
            source_path=path,
        )

    @classmethod
    def from_piano_roll(cls, piano_roll: np.ndarray, tempo: float = 120.0, fs: int = 100) -> MIDIEntity:
        """
        Construct a MIDIEntity directly from a (88, T) piano roll array.
        """
        assert piano_roll.shape[0] == 88, "piano_roll must have 88 rows (keys)"
        notes = []
        T = piano_roll.shape[1]
        for key_idx in range(88):
            pitch = key_idx + 21
            active = False
            start_frame = 0
            for t in range(T):
                if piano_roll[key_idx, t] > 0 and not active:
                    active = True
                    start_frame = t
                elif piano_roll[key_idx, t] == 0 and active:
                    active = False
                    notes.append(NoteEvent(
                        pitch=pitch,
                        velocity=64,
                        start=start_frame / fs,
                        end=t / fs,
                    ))
            if active:
                notes.append(NoteEvent(
                    pitch=pitch,
                    velocity=64,
                    start=start_frame / fs,
                    end=T / fs,
                ))
        notes.sort(key=lambda n: n.start)
        return cls(
            notes=notes,
            tempo=tempo,
            duration=T / fs,
            piano_roll=piano_roll.astype(np.float32),
        )

    def sigma(self, t: int) -> np.ndarray:
        """
        Return the 88-dimensional activation vector σ at frame t.
        """
        if self.piano_roll is None:
            raise ValueError("piano_roll not available")
        return self.piano_roll[:, t]

    def __repr__(self) -> str:
        return (f"MIDIEntity(notes={len(self.notes)}, "
                f"duration={self.duration:.2f}s, tempo={self.tempo:.1f}bpm, "
                f"piano_roll={self.piano_roll.shape if self.piano_roll is not None else None})")