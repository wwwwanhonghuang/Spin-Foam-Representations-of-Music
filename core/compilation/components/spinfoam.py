"""
SpinfoamComplex: core data structure representing a 2-complex
with spin labels on faces, intertwiner labels on edges, and
amplitude data on vertices.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import zipfile
import json
import io

SP_FORMAT_VERSION = 1.0


@dataclass
class Face:
    """A 2-cell in the spinfoam complex."""
    id: int
    spin_j: float                        # j ∈ {0, 0.5, 1, 1.5, ...}  < - nature
    spin_m: float                        # specific value in (-j, j, 1/2)  < - nature
    amplitude: complex = 1.0 + 0.0j      # <- nature
    semantic_label: Optional[str] = None   # e.g. "note:60", "freq:k3"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """A 1-cell bounding faces."""
    id: int
    face_ids: Tuple[int, ...]          # faces sharing this edge
    intertwiner: Optional[float] = None
    from_vertex: Optional[int] = None    # The causal source
    to_vertex: Optional[int] = None      # The causal target
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
    faces: List[Face] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    vertices: List[Vertex] = field(default_factory=list)
    scheme_id: Optional[str] = None
    source_midi: Optional[str] = None

    def __post_init__(self):
        # This forces the attributes into existence immediately 
        # after the object is created, even if they aren't in the __init__
        object.__setattr__(self, '_face_index', {f.id: f for f in self.faces})
        object.__setattr__(self, '_vertex_index', {v.id: v for v in self.vertices})

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
    
        # ── Serialization ─────────────────────────────────────────────────────────

    def get_boundary_gamma_order(self, gamma_root_ids: List[int]) -> Dict[int, int]:
        """
        Reconstructs the causal order (Preorder) from a given boundary graph Gamma_0.
        
        This implements the 'Symmetry Breaking' step where the background-independent 
        complex is 'polarized' into a sequence of events.
        
        Args:
            gamma_root_ids: The list of vertex IDs belonging to the initial 
                           boundary graph (Gamma_0).
        Returns:
            A mapping of {vertex_id: rank_index} where rank 0 is Gamma_0.
        """
        self.build_index()
        
        # Initialize ranks: Gamma_0 is rank 0
        ranks = {vid: 0 for vid in gamma_root_ids}
        for vid in gamma_root_ids:
            if vid in self._vertex_index:
                self._vertex_index[vid].time_index = 0.0
        
        queue = list(gamma_root_ids)
        
        # Build out-flow adjacency based on directed edges
        # In a preorder ver, causality is encoded in edge orientation (from -> to)
        out_adjacency: Dict[int, List[int]] = {}
        for edge in self.edges:
            if edge.from_vertex is not None and edge.to_vertex is not None:
                out_adjacency.setdefault(edge.from_vertex, []).append(edge.to_vertex)

        # Breadth-First Traversal to establish topological depth (Time Emergence)
        while queue:
            u = queue.pop(0)
            current_rank = ranks[u]
            
            for v_id in out_adjacency.get(u, []):
                # Standard DAG preorder logic: 
                # A vertex's rank is the maximum depth of its causal past
                new_rank = current_rank + 1
                if v_id not in ranks or new_rank > ranks[v_id]:
                    ranks[v_id] = new_rank
                    if v_id in self._vertex_index:
                        self._vertex_index[v_id].time_index = float(new_rank)
                    queue.append(v_id)
                    
        return ranks
    
    def serialize(self, path: str) -> None:
        """
        Save the SpinfoamComplex to a .sp file (ZIP archive).

        Layout:
            header.json
            faces.npy                       float64 (N_f, 3): id, spin_j, spin_m
            face_amplitudes.npy             complex128 (N_f,): A_f per face
            face_labels.json                list[str|null]
            face_meta.json                  list[dict]
            edges.npy                       int64 (N_e, 3): id, from_vertex, to_vertex
            edge_face_ids.json              list[list[int]]
            edge_intertwiners/edge_N.npy    one file per edge with ndarray intertwiner
            edge_intertwiner_scalars.json   {str(id): float} for scalar intertwiners
            edge_meta.json                  list[dict]
            vertices.npy                    float64 (N_v, 4): id, amp_re, amp_im, time_index
            vertex_edge_ids.json            list[list[int]]
            vertex_meta.json                list[dict]
        """
        if not path.endswith(".sp"):
            path = path + ".sp"

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:

            # ── header ───────────────────────────────────────────────────────
            header = {
                "version":     SP_FORMAT_VERSION,
                "scheme_id":   self.scheme_id,
                "source_midi": self.source_midi,
                "n_faces":     len(self.faces),
                "n_edges":     len(self.edges),
                "n_vertices":  len(self.vertices),
            }
            zf.writestr("header.json", json.dumps(header, indent=2))

            # ── faces ─────────────────────────────────────────────────────────
            if self.faces:
                face_arr = np.array(
                    [[f.id, f.spin_j, f.spin_m] for f in self.faces],
                    dtype=np.float64,
                )
                zf.writestr("faces.npy", _npy_bytes(face_arr))

                amp_arr = np.array(
                    [f.amplitude for f in self.faces], dtype=np.complex128
                )
                zf.writestr("face_amplitudes.npy", _npy_bytes(amp_arr))

                zf.writestr("face_labels.json",
                    json.dumps([f.semantic_label for f in self.faces]))
                zf.writestr("face_meta.json",
                    json.dumps([_safe_meta(f.metadata) for f in self.faces]))

            # ── edges ─────────────────────────────────────────────────────────
            if self.edges:
                edge_arr = np.array(
                    [[e.id,
                      e.from_vertex if e.from_vertex is not None else -1,
                      e.to_vertex   if e.to_vertex   is not None else -1]
                     for e in self.edges],
                    dtype=np.int64,
                )
                zf.writestr("edges.npy", _npy_bytes(edge_arr))

                zf.writestr("edge_face_ids.json",
                    json.dumps([list(e.face_ids) for e in self.edges]))
                zf.writestr("edge_meta.json",
                    json.dumps([_safe_meta(e.metadata) for e in self.edges]))

                scalar_intertwiners = {}
                for e in self.edges:
                    if e.intertwiner is None:
                        continue
                    if isinstance(e.intertwiner, np.ndarray):
                        zf.writestr(
                            f"edge_intertwiners/edge_{e.id}.npy",
                            _npy_bytes(e.intertwiner),
                        )
                    else:
                        scalar_intertwiners[str(e.id)] = float(e.intertwiner)

                zf.writestr("edge_intertwiner_scalars.json",
                    json.dumps(scalar_intertwiners))

            # ── vertices ──────────────────────────────────────────────────────
            if self.vertices:
                vertex_arr = np.array(
                    [[v.id,
                      v.amplitude.real,
                      v.amplitude.imag,
                      v.time_index if v.time_index is not None else float("nan")]
                     for v in self.vertices],
                    dtype=np.float64,
                )
                zf.writestr("vertices.npy", _npy_bytes(vertex_arr))

                zf.writestr("vertex_edge_ids.json",
                    json.dumps([list(v.edge_ids) for v in self.vertices]))
                zf.writestr("vertex_meta.json",
                    json.dumps([_safe_meta(v.metadata) for v in self.vertices]))

        print(f"Serialized: {self.summary()} -> {path}")

    # ── Deserialization ───────────────────────────────────────────────────────
    def add_vertex(self, id: int, amplitude: float = 1.0) -> Vertex:
        """Adds a vertex if it doesn't exist, or returns the existing one."""
        if id in self._vertex_index:
            return self._vertex_index[id]
        
        v = Vertex(id=id, amplitude=amplitude)
        self.vertices.append(v)
        self._vertex_index[id] = v
        return v

    def add_edge(self, from_vertex: int, to_vertex: int, face_ids: List[int] = None) -> Edge:
        """
        Creates a directed edge between two vertices. 
        Automatically updates the edge_ids list of the involved vertices.
        """
        eid = len(self.edges)
        edge = Edge(
            id=eid, 
            from_vertex=from_vertex, 
            to_vertex=to_vertex, 
            face_ids=face_ids or []
        )
        self.edges.append(edge)
        
        # Maintain the link back from the vertices to this edge
        if from_vertex in self._vertex_index:
            self._vertex_index[from_vertex].edge_ids.append(eid)
        if to_vertex in self._vertex_index:
            self._vertex_index[to_vertex].edge_ids.append(eid)
            
        return edge

    def add_face(self, edge_ids: List[int], semantic_label: str = "", amplitude: float = 1.0) -> Face:
        """Creates a face (musical note) spanning a sequence of edges."""
        fid = len(self.faces)
        face = Face(
            id=fid, 
            edge_ids=edge_ids, 
            semantic_label=semantic_label, 
            amplitude=amplitude
        )
        self.faces.append(face)
        self._face_index[fid] = face
        
        # Back-link: Update all involved edges to know they carry this face
        for eid in edge_ids:
            if eid < len(self.edges):
                self.edges[eid].face_ids.append(fid)
                
        return face

    def get_edge_by_vertices(self, from_vid: int, to_vid: int) -> Optional[Edge]:
        """
        Optimized lookup to find an edge connecting two specific vertices.
        Useful for 'staining' the timeline during compilation.
        """
        if from_vid not in self._vertex_index:
            return None
        
        # Check edges connected to the start vertex
        for eid in self._vertex_index[from_vid].edge_ids:
            e = self.edges[eid]
            if e.from_vertex == from_vid and e.to_vertex == to_vid:
                return e
        return None
    
    @classmethod
    def deserialize(cls, path: str) -> SpinfoamComplex:
        """
        Load a SpinfoamComplex from a .sp file.
        """
        if not path.endswith(".sp"):
            path = path + ".sp"

        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())

            # header
            header = json.loads(zf.read("header.json"))
            sf = cls(
                scheme_id=header.get("scheme_id"),
                source_midi=header.get("source_midi"),
            )

            # faces
            if "faces.npy" in names:
                face_arr  = _load_npy(zf, "faces.npy")       # (N_f, 3)
                amp_arr   = _load_npy(zf, "face_amplitudes.npy")
                labels    = json.loads(zf.read("face_labels.json"))
                face_meta = json.loads(zf.read("face_meta.json"))

                for i, row in enumerate(face_arr):
                    sf.faces.append(Face(
                        id=int(row[0]),
                        spin_j=float(row[1]),
                        spin_m=float(row[2]),
                        amplitude=complex(amp_arr[i]),
                        semantic_label=labels[i],
                        metadata=face_meta[i],
                    ))

            # edges
            if "edges.npy" in names:
                edge_arr        = _load_npy(zf, "edges.npy")  # (N_e, 3)
                edge_face_ids   = json.loads(zf.read("edge_face_ids.json"))
                edge_meta       = json.loads(zf.read("edge_meta.json"))
                scalar_itwn     = json.loads(zf.read("edge_intertwiner_scalars.json"))

                for i, row in enumerate(edge_arr):
                    eid        = int(row[0])
                    from_v     = int(row[1]) if row[1] != -1 else None
                    to_v       = int(row[2]) if row[2] != -1 else None
                    itwn_key   = f"edge_intertwiners/edge_{eid}.npy"
                    if itwn_key in names:
                        intertwiner = _load_npy(zf, itwn_key)
                    elif str(eid) in scalar_itwn:
                        intertwiner = scalar_itwn[str(eid)]
                    else:
                        intertwiner = None

                    sf.edges.append(Edge(
                        id=eid,
                        face_ids=tuple(edge_face_ids[i]),
                        from_vertex=from_v,
                        to_vertex=to_v,
                        intertwiner=intertwiner,
                        metadata=edge_meta[i],
                    ))

            # vertices
            if "vertices.npy" in names:
                vertex_arr      = _load_npy(zf, "vertices.npy")  # (N_v, 4)
                vertex_edge_ids = json.loads(zf.read("vertex_edge_ids.json"))
                vertex_meta     = json.loads(zf.read("vertex_meta.json"))

                for i, row in enumerate(vertex_arr):
                    ti = float(row[3])
                    sf.vertices.append(Vertex(
                        id=int(row[0]),
                        edge_ids=tuple(vertex_edge_ids[i]),
                        amplitude=complex(float(row[1]), float(row[2])),
                        time_index=None if np.isnan(ti) else ti,
                        metadata=vertex_meta[i],
                    ))

        sf.build_index()
        print(f"Deserialized: {sf.summary()} <- {path}")
        return sf

    def __init__(self, scheme_id="", source_midi=None):
        # Explicit initialization of all lists and indices
        self.faces = []
        self.edges = []
        self.vertices = []
        self.scheme_id = scheme_id
        self.source_midi = source_midi
        
        # Internal indices created immediately
        self._face_index = {}
        self._vertex_index = {}
# ── Internal helpers ──────────────────────────────────────────────────────────

def _npy_bytes(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to raw .npy bytes (in-memory)."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _load_npy(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    """Load a numpy array from a named entry inside a ZipFile."""
    return np.load(io.BytesIO(zf.read(name)), allow_pickle=False)


def _safe_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure metadata is JSON-serializable.
    numpy scalars and arrays are converted to Python native types.
    """
    out = {}
    for k, v in meta.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out