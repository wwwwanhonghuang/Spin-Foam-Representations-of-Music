"""
SpinfoamComplex: core data structure representing a 2-complex
with spin labels on faces, intertwiner labels on edges, and
amplitude data on vertices.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


@dataclass
class Face:
    """A 2-cell in the spinfoam complex."""
    id: int
    spin: float                        # j ∈ {0, 0.5, 1, 1.5, ...}
    semantic_label: Optional[str] = None   # e.g. "note:60", "freq:k3"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """A 1-cell bounding faces."""
    id: int
    face_ids: Tuple[int, ...]          # faces sharing this edge
    intertwiner: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Vertex:
    """A 0-cell where edges meet; carries the vertex amplitude."""
    id: int
    edge_ids: Tuple[int, ...]
    amplitude: complex = 0.0 + 0.0j
    time_index: Optional[float] = None    # populated by T-scheme
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpinfoamComplex:
    """
    A combinatorial 2-complex with spin labels.
    Produced by any compilation scheme in ./compilation/.
    """
    faces: List[Face] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    vertices: List[Vertex] = field(default_factory=list)

    # Provenance
    scheme_id: Optional[str] = None       # e.g. "iso_direct_indep_topo_vertex"
    source_midi: Optional[str] = None     # path to source MIDI

    # Derived structures (populated lazily)
    _face_index: Dict[int, Face] = field(default_factory=dict, repr=False)
    _vertex_index: Dict[int, Vertex] = field(default_factory=dict, repr=False)

    def build_index(self) -> None:
        self._face_index = {f.id: f for f in self.faces}
        self._vertex_index = {v.id: v for v in self.vertices}

    def face(self, fid: int) -> Face:
        return self._face_index[fid]

    def vertex(self, vid: int) -> Vertex:
        return self._vertex_index[vid]

    def adjacency_matrix(self) -> np.ndarray:
        """Vertex-vertex adjacency via shared edges."""
        n = len(self.vertices)
        vid_to_idx = {v.id: i for i, v in enumerate(self.vertices)}
        eid_to_vids: Dict[int, List[int]] = {}
        for v in self.vertices:
            for eid in v.edge_ids:
                eid_to_vids.setdefault(eid, []).append(v.id)
        A = np.zeros((n, n), dtype=int)
        for vids in eid_to_vids.values():
            for i in range(len(vids)):
                for j in range(i + 1, len(vids)):
                    a, b = vid_to_idx[vids[i]], vid_to_idx[vids[j]]
                    A[a, b] = A[b, a] = 1
        return A

    def total_amplitude(self) -> complex:
        return sum(v.amplitude for v in self.vertices)

    def summary(self) -> str:
        return (f"SpinfoamComplex(scheme={self.scheme_id}, "
                f"faces={len(self.faces)}, edges={len(self.edges)}, "
                f"vertices={len(self.vertices)})")

    def __repr__(self) -> str:
        return self.summary()