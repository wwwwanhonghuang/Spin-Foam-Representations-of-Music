"""
001. Iso-Direct-Indep + Topo-Vertex-PartialOrder
Classification: S = Iso-Direct-Independent   T = T.1.1 (Vertex partial order)

Two encoding variants:

    STANDARD_ENCODING (0):
        128 faces per vertex, j_f = 1/2 for all faces.
        m_f = +1/2 (activated) or -1/2 (silent), one per MIDI key.
        Full information, no loss. F = 128*N.

    COMPACT_ENCODING (1):
        One face per ACTIVE note per vertex. j_f = 63.5 = 127/2 (fixed).
        m_f = k - 63.5  where k in {0,...,127} is the MIDI pitch.
            key 0   -> m = -63.5
            key 1   -> m = -62.5
            ...
            key 127 -> m = +63.5
        Encodes pitch identity directly in m. Face count = n_active (sparse).
        Silent keys produce no face. F = sum_i n_active_i.

Two vertex event types:

    USE_EVENT_TYPE_NOTE_CHANGE (0):
        One vertex per sigma_sequence entry (event-driven, irregular spacing).

    USE_EVENT_TYPE_CLOCK_TICK (1):
        One vertex per 16th-note clock tick at current BPM (uniform grid).

Syntax Mapping (§1) — Standard:
    V = {v_i},       one per snapshot,  |V| = N
    F = {f_{i,k}},   128 per vertex,    j = 1/2,    m = ±1/2
    E = {e_i},       one per adjacent pair, diagonal CG intertwiner
    A_f = 2,  A_v = n_active,  time_index = i

Syntax Mapping (§1) — Compact:
    V = {v_i},       one per snapshot,  |V| = N
    F = {f_{i,p}},   one per active pitch p at snapshot i,  j = 63.5,  m = p - 63.5
    E = {e_i},       one per adjacent pair, intertwiner couples active pitch sets
    A_f = 2*63.5+1 = 128,  A_v = n_active,  time_index = i
"""

from __future__ import annotations
import numpy as np
from music.midi.midi_entity import MIDIEntity
from core.compilation.components.spinfoam import SpinfoamComplex, Face, Vertex, Edge
from core.compilation.compilation_scheme import CompilationScheme
from typing import Dict, Tuple
import mido
from tqdm import tqdm
from pathlib import Path
import os

USE_EVENT_TYPE_NOTE_CHANGE = 0
USE_EVENT_TYPE_CLOCK_TICK  = 1

STANDARD_ENCODING = 0
COMPACT_ENCODING  = 1

# Standard: one face per MIDI key, binary spin
J_STANDARD      = 0.5
A_FACE_STANDARD = 2.0 + 0j          # 2*j + 1 = 2

# Compact: one face per active note, pitch encoded in m
J_COMPACT       = 63.5              # 127/2 — m covers all 128 MIDI pitches
A_FACE_COMPACT  = 128.0 + 0j        # 2*63.5 + 1 = 128


class IsoDirectIndepTopoVertex(CompilationScheme):
    @property
    def scheme_id(self) -> str:
        return "IsoDirectIndepTopoVertex"

    @property
    def description(self) -> str:
        return "IsoDirectIndepTopoVertex"


    
    def __init__(self, ticks_per_rank: int = 480):
        self.ticks_per_rank = ticks_per_rank

    def decode_to_midi_file(self, sf: SpinfoamComplex, init_vertex_id: int, output_path: str):
        print(f"=== Dense Causal Recovery: {output_path} ===")
        
        # 1. Prepare Fast Lookups
        # Map edges by their starting vertex so we can walk the chain O(1)
        edge_map = {e.from_vertex: e for e in sf.edges}
        face_map = {f.id: f for f in sf.faces}

        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # State tracking to manage Note On / Note Off logic
        active_pitches = {} # pitch -> current_velocity
        curr_v = init_vertex_id
        total_reconstructed_ticks = 0

        # 2. Walk the Spinfoam Chain
        while curr_v in edge_map:
            edge = edge_map[curr_v]
            
            # Identify pitches active on THIS specific edge (this tick)
            pitches_on_this_edge = set()
            pitch_velocities = {}
            
            for fid in edge.face_ids:
                face = face_map.get(fid)
                if face:
                    # Extract pitch from label (e.g., "note:60" -> 60)
                    try:
                        p = int(face.semantic_label.split(":")[-1])
                        pitches_on_this_edge.add(p)
                        # Convert complex amplitude back to MIDI velocity
                        pitch_velocities[p] = int(abs(face.amplitude) * 127)
                    except (ValueError, AttributeError):
                        continue

            # 3. MIDI Logic: Resolve Differences
            # Case A: Note Offs (Present in active_pitches but NOT on this edge)
            for p in list(active_pitches.keys()):
                if p not in pitches_on_this_edge:
                    # time=0 because we use a global clock tick at the end of the loop
                    track.append(mido.Message('note_off', note=p, velocity=0, time=0))
                    del active_pitches[p]

            # Case B: Note Ons (On this edge but NOT in active_pitches)
            tick_delta_applied = False
            for p in pitches_on_this_edge:
                if p not in active_pitches:
                    # If this is the first message of the tick, we use time=1 to advance the clock
                    # Otherwise, time=0 (chord/simultaneous notes)
                    delta = 1 if not tick_delta_applied else 0
                    track.append(mido.Message('note_on', note=p, velocity=pitch_velocities[p], time=delta))
                    active_pitches[p] = pitch_velocities[p]
                    tick_delta_applied = True

            # Case C: Silence / Clock Advance
            # If no notes started, we still MUST move the MIDI clock forward
            if not tick_delta_applied:
                # Add a 'dummy' message or update the next message's time
                # We use a meta-message or note_off with 0 velocity to push 1 tick of time
                track.append(mido.Message('note_off', note=0, velocity=0, time=1))

            # Move to the next "Tick Vertex" in the lattice
            curr_v = edge.to_vertex
            total_reconstructed_ticks += 1

        # 4. Finalization
        mid.save(output_path)
        print(f"Recovery complete. Total Ticks: {total_reconstructed_ticks}")
        
    def compile_to_spinfoam(self, midi_entity: MIDIEntity) -> SpinfoamComplex:
        # 1. Determine Total Duration
        # We find the max absolute tick among all events
        if not midi_entity.events:
            return SpinfoamComplex(scheme_id="empty")
            
        total_ticks = max(e.tick for e in midi_entity.events)
        
        sf = SpinfoamComplex(
            scheme_id="dense_clock_recovery", 
            source_midi=midi_entity.source_path
        )
        
        print(f"Allocating Dense Lattice: {total_ticks} vertices...")

        # 2. BULK ALLOCATION (Pre-satisfying Vertex __init__)
        # Every vertex is a tick, every edge is a passage of 1 tick.
        sf.vertices = [Vertex(id=i, edge_ids=[]) for i in range(total_ticks + 1)]
        sf._vertex_index = {v.id: v for v in sf.vertices}
        
        sf.edges = [
            Edge(id=t, from_vertex=t, to_vertex=t+1, face_ids=[]) 
            for t in range(total_ticks)
        ]

        # 3. LINK CAUSAL CHAIN
        # This allows the preorder decoder to walk the timeline
        for t in tqdm(range(total_ticks)):
            e_id = t
            sf.vertices[t].edge_ids.append(e_id)     # Outgoing
            sf.vertices[t+1].edge_ids.append(e_id)   # Incoming

        # 4. MAP EVENTS TO FACES
        print("Mapping MIDIEntity events to Spinfoam faces...")
        active_notes = {}  # pitch -> (start_tick, velocity)

        for event in tqdm(midi_entity.events):
            t = event.tick
            
            # Check if the attribute is 'note' (standard MIDI terminology)
            # or 'pitch' (often used in piano-roll representations)
            p = getattr(event, 'note', None) 
            v = getattr(event, 'velocity', 0)
            
            if p is None: continue # Skip meta-events like tempo changes

            if event.type == 'note_on' and v > 0:
                active_notes[p] = (t, v)
                
            elif (event.type == 'note_off' or (event.type == 'note_on' and v == 0)):
                if p in active_notes:
                    start_t, velocity = active_notes.pop(p)
                    end_t = t
                    
                    if end_t > start_t:
                        # We map the time range to metadata since 'edge_ids' isn't in your Face class
                        edge_ids = list(range(start_t, end_t))
                        
                        # PHYSICAL MAPPING: 
                        # You might want a custom mapping for j and m. 
                        # For now, let's use pitch as j for simplicity.
                        j_val = float(p) / 127.0 
                        m_val = 0.0 # Or some variation based on velocity/channel
                        
                        new_face = Face(
                            id=len(sf.faces),
                            spin_j=j_val,
                            spin_m=m_val,
                            amplitude=complex(velocity / 127.0, 0.0),
                            semantic_label=f"note:{p}",
                            metadata={'edge_ids': edge_ids} # Store boundaries here!
                        )
                        
                        sf.faces.append(new_face)
                        sf._face_index[new_face.id] = new_face
                        
                        # IMPORTANT: Update the edges to know they are part of this face
                        for eid in edge_ids:
                            sf.edges[eid].face_ids.append(new_face.id)

        print(f"Lattice Complete: {len(sf.vertices)} vertices, {len(sf.faces)} faces.")
        return sf

    def _add_edge_to_vertex(self, sf: SpinfoamComplex, v_id: int, e_id: int):
        # Helper to maintain the tuple of edge_ids in Vertex
        for v in sf.vertices:
            if v.id == v_id:
                v.edge_ids = v.edge_ids + (e_id,)
                break
            
            
# encoding method:
# j=5.5, specific note in a musical scale encode in amplitude
# pitch directly encode in amplitude
# 128 faces carry j = 1/2 spin, which each face represent a note, m=1/2 when on, while m=-1/2 when off.