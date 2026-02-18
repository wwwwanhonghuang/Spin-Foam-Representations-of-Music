from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import mido
from typing import List, Optional, Dict
from music.midi.utils import note_name
from collections import defaultdict

@dataclass
class MIDIEvent:
    type: str           # 'note_on', 'note_off', 'control_change', 'tempo', 'time_sig'
    time: float         # Absolute time in seconds
    tick: int           # Absolute time in MIDI ticks (integer precision)
    channel: int = 0
    note: Optional[int] = None
    velocity: Optional[int] = None
    control: Optional[int] = None
    value: Optional[int] = None
    data: Optional[Dict] = None  # For Meta-events like tempo/time_signature

@dataclass
class MIDIEntity:
    source_path: Optional[str] = None
    ticks_per_beat: int = 480
    
    # 1. The Raw Event Stream (Chronological)
    events: List[MIDIEvent] = field(default_factory=list)
    
    # 2. The Discrete State Sequence (sigma_sequence)
    sigma_sequence: Optional[np.ndarray] = None 
    
    # 3. Temporal Metadata
    event_times: np.ndarray = field(default_factory=lambda: np.array([]))  # Absolute seconds
    delta_times: np.ndarray = field(default_factory=lambda: np.array([]))  # Interval between events

    # 4. Tempo Map: list of {'time': float (seconds), 'bpm': float}
    # Ordered chronologically. Captures all tempo changes in the file.
    # Use bpm_at(t) to query BPM at any point in time.
    tempo_map: List[Dict] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> "MIDIEntity":
        mid = mido.MidiFile(path)
        ticks_per_beat = mid.ticks_per_beat

        raw_events: List[MIDIEvent] = []

        # Structural time
        current_time_sec = 0.0
        current_tick = 0

        # Default tempo = 500000 microseconds per beat (120 BPM)
        tempo = 500000

        # Internal state for sigma tracking
        active_state = np.zeros(128, dtype=np.float32)
        snapshots = []
        timestamps = []

        # Tempo map
        tempo_map = [{'time': 0.0, 'bpm': 120.0}]

        # IMPORTANT: iterate merged message stream
        for msg in mid:

            # ─────────────────────────────
            # 1. Accumulate continuous time
            # msg.time is delta time in seconds
            # ─────────────────────────────
            delta_sec = msg.time
            current_time_sec += delta_sec

            # ─────────────────────────────
            # 2. Accumulate discrete ticks
            # Convert seconds back to ticks using current tempo
            # ─────────────────────────────
            delta_tick = mido.second2tick(
                delta_sec,
                ticks_per_beat,
                tempo
            )
            current_tick += int(round(delta_tick))

            # ─────────────────────────────
            # 3. Handle tempo changes
            # ─────────────────────────────
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                bpm = mido.tempo2bpm(tempo)

                tempo_map.append({
                    'time': current_time_sec,
                    'bpm': bpm
                })

            # ─────────────────────────────
            # 4. Create event
            # ─────────────────────────────
            new_event = MIDIEvent(
                type=msg.type,
                time=current_time_sec,
                tick=current_tick
            )

            state_changed = False

            # ─────────────────────────────
            # 5. Handle note events
            # ─────────────────────────────
            if msg.type == 'note_on':
                new_event.note = msg.note
                new_event.velocity = msg.velocity
                new_event.channel = msg.channel

                if msg.velocity > 0:
                    active_state[msg.note] = msg.velocity
                else:
                    active_state[msg.note] = 0

                state_changed = True

            elif msg.type == 'note_off':
                new_event.note = msg.note
                new_event.velocity = 0
                new_event.channel = msg.channel

                active_state[msg.note] = 0
                state_changed = True

            elif msg.type == 'control_change':
                new_event.control = msg.control
                new_event.value = msg.value
                new_event.channel = msg.channel

            elif msg.type == 'time_signature':
                new_event.data = {
                    'numerator': msg.numerator,
                    'denominator': msg.denominator,
                }

            raw_events.append(new_event)

            # ─────────────────────────────
            # 6. Snapshot sigma state
            # ─────────────────────────────
            if state_changed:
                snapshots.append(active_state[21:109].copy())
                timestamps.append(current_time_sec)

        return cls(
            source_path=path,
            ticks_per_beat=ticks_per_beat,
            events=raw_events,
            sigma_sequence=np.array(snapshots) if snapshots else np.empty((0, 128)),
            event_times=np.array(timestamps),
            delta_times=np.diff(timestamps, prepend=0.0),
            tempo_map=tempo_map,
        )

    # ── BPM queries ──────────────────────────────────────────────────────────

    @property
    def initial_bpm(self) -> float:
        """BPM at the start of the file (t = 0)."""
        return self.tempo_map[0]['bpm'] if self.tempo_map else 120.0

    @property
    def final_bpm(self) -> float:
        """BPM at the last tempo change."""
        return self.tempo_map[-1]['bpm'] if self.tempo_map else 120.0

    @property
    def is_constant_tempo(self) -> bool:
        """True if the file has no tempo changes."""
        return len(self.tempo_map) == 1

    def bpm_at(self, time: float) -> float:
        """
        Return the BPM in effect at the given absolute time (seconds).
        Uses the last tempo change that occurred at or before `time`.
        """
        bpm = self.tempo_map[0]['bpm']
        for entry in self.tempo_map:
            if entry['time'] <= time:
                bpm = entry['bpm']
            else:
                break
        return bpm

    def bpm_sequence(self) -> np.ndarray:
        """
        Return a (N,) array of BPM values aligned to sigma_sequence.
        One BPM value per state snapshot — useful as a feature in compilation schemes.
        """
        return np.array([self.bpm_at(t) for t in self.event_times])

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        duration = self.event_times[-1] if len(self.event_times) > 0 else 0.0
        bpm_info = (
            f"bpm={self.initial_bpm:.1f}"
            if self.is_constant_tempo
            else f"bpm={self.initial_bpm:.1f}→{self.final_bpm:.1f} ({len(self.tempo_map)} changes)"
        )
        return (f"MIDIEntity(source={self.source_path}, "
                f"events={len(self.events)}, "
                f"states={len(self.sigma_sequence)}, "
                f"duration={duration:.2f}s, "
                f"{bpm_info})")
        
    def print_notes_by_tick(
        self,
        limit: int | None = None,
        use_flats: bool = False,
        include_note_off: bool = False
    ) -> None:
        """
        Print notes grouped by tick.

        Parameters
        ----------
        limit : int | None
            Maximum number of tick lines to print.
        use_flats : bool
            Use flat note naming.
        include_note_off : bool
            If True, include note_off events.
        """

        tick_dict = defaultdict(list)
        time_dict = {}

        # ── Group events by tick ─────────────────────
        for ev in self.events:
            if ev.type == "note_on":
                # velocity 0 note_on is semantically note_off
                if ev.velocity is not None and ev.velocity > 0:
                    tick_dict[ev.tick].append(ev.note)
                    time_dict[ev.tick] = ev.time
                elif include_note_off:
                    tick_dict[ev.tick].append(ev.note)
                    time_dict[ev.tick] = ev.time

            elif include_note_off and ev.type == "note_off":
                tick_dict[ev.tick].append(ev.note)
                time_dict[ev.tick] = ev.time

        # ── Sort ticks chronologically ───────────────
        sorted_ticks = sorted(tick_dict.keys())

        count = 0
        for tick in sorted_ticks:
            notes = tick_dict[tick]

            if not notes:
                continue

            names = [note_name(n, use_flats=use_flats) for n in notes]
            notes_str = " ".join(names)

            print(f"Tick {tick:07d} | t={time_dict[tick]:8.3f}s | {notes_str}")

            count += 1
            if limit is not None and count >= limit:
                print("... (truncated)")
                break
