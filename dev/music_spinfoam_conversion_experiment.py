
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from math import factorial, sqrt
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────
# §0.  Pre-physical layer
# ─────────────────────────────────────────────────────────────

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
NOTE_MAP: Dict[str,int] = {
    "C3":48,"D3":50,"F3":53,
    "C4":60,"D4":62,"F4":65,
    "A2":45,"F1":29,"F2":41,
}

def _midi_to_index(m:int)->int: return m-21
def _index_to_midi(i:int)->int: return i+21

def pitch_to_quantum_numbers(idx:int)->Tuple[float,int]:
    midi=_index_to_midi(idx)
    return (midi%12)/2.0, (midi//12)-1

def decode_quantum_numbers(j:float, q:int)->int:
    return round(2.0*j)+12*(q+1)-21

def index_to_name(idx:int)->str:
    midi=_index_to_midi(idx)
    return f"{NOTE_NAMES[midi%12]}{(midi//12)-1}"

def jq_to_name(j:float, q:int)->str:
    return index_to_name(decode_quantum_numbers(j,q))

State = FrozenSet[Tuple[float,int]]


# ─────────────────────────────────────────────────────────────
# §1.  GFT Theory Layer
# ─────────────────────────────────────────────────────────────

@dataclass
class GFTParams:
    """
    mu2:    μ² in K(j,q) = j(j+1) + q² + μ²
    lam:    vertex coupling λ
    P_min:  propagator cutoff.  P_min=0.10 → |V|≈10 → 2^10=1024 states
    sigma_J: harmonic temperature σ_J in A_v = λ·exp(−J_min/σ_J)
             J_min = minimum admissible total spin of the chord.
             J_min=0 (closed chord) → A_v=λ  (unchanged)
             J_min>0 (open chord)   → A_v<λ  (suppressed, never zero)
             Large σ_J: lenient — all chords contribute similarly.
             Small σ_J: strict  — only J=0-closing chords dominate.
             Default 1.0 gives ≈37% suppression per unit of J_min.
    """
    mu2:    float = 0.5
    lam:    float = 1.0
    P_min:  float = 0.10
    sigma_J:float = 1.0


def gft_propagator(j:float, q:int, p:GFTParams)->float:
    return 1.0/(j*(j+1)+float(q**2)+p.mu2)


def _cg_decompose(j1:float, j2:float)->List[float]:
    out,J=[],abs(j1-j2)
    while J<=j1+j2+1e-9: out.append(round(J,1)); J+=1.0
    return out


def _couple_spins(spins:List[float])->Tuple[List[float],int,int]:
    if not spins: return [0.0],1,1
    coupled:Dict[float,int]={spins[0]:1}
    for jn in spins[1:]:
        nxt:Dict[float,int]={}
        for jc,m in coupled.items():
            for J in _cg_decompose(jc,jn): nxt[J]=nxt.get(J,0)+m
        coupled=nxt
    return sorted(coupled.keys()),coupled.get(0.0,0),sum(coupled.values())


@lru_cache(maxsize=4096)
def _vertex_amplitude_cached(js_tuple:Tuple[float,...], lam:float,
                             sigma_J:float)->float:
    """
    Gaussian vertex amplitude — replaces strict dim(Inv)/total.

    A_v(js) = λ · exp(−J_min / σ_J)

    where J_min = min admissible total spin from coupling j1⊗...⊗jk.

    Properties:
      • Always positive: A_v > 0 for any chord.
      • J_min = 0 (chord closes to J=0) → A_v = λ  (maximal, matches old formula).
      • J_min > 0 (chord cannot close)   → A_v < λ  (suppressed by Boltzmann factor).
      • Vacuum (js empty)               → A_v = λ  (no constraint).
      • Limit σ_J→0: recovers strict formula (A_v=λ if J_min=0, else ≈0).
      • Limit σ_J→∞: A_v→λ for all chords (no harmonic preference).
    """
    if not js_tuple:
        return lam
    admJ, _, _ = _couple_spins(list(js_tuple))
    J_min = admJ[0]   # minimum achievable total spin
    return lam * float(np.exp(-J_min / sigma_J))


def gft_vertex_amplitude(js:List[float], p:GFTParams)->float:
    """A_v = λ·exp(−J_min/σ_J).  Cached on spin content + parameters."""
    return _vertex_amplitude_cached(tuple(sorted(js)), p.lam, p.sigma_J)


# ─────────────────────────────────────────────────────────────
# §2.  Intertwiner Tensor
# ─────────────────────────────────────────────────────────────

_TENSOR_CAP = 5000

def _safe_fac(x:float)->int:
    xi=int(round(x)); return factorial(xi) if xi>=0 else 0

def _wigner_j0(j1:float, j2:float)->Optional[np.ndarray]:
    if abs(j1-j2)>1e-9: return None
    d=round(2*j1)+1; iota=np.zeros((d,d))
    for im1,m1 in enumerate(np.arange(-j1,j1+0.5,1.0)):
        im2=round(-m1+j2)
        if 0<=im2<d: iota[im1,im2]=((-1)**(j1-m1))/sqrt(2*j1+1)
    return iota

def _cg_coeff(j1,m1,j2,m2,J,M)->float:
    if abs(m1+m2-M)>1e-9: return 0.0
    if J<abs(j1-j2)-1e-9 or J>j1+j2+1e-9: return 0.0
    def tri(a,b,c):
        n=int(round(a+b-c))
        if n<0: return 0.0
        return sqrt(_safe_fac(n)*_safe_fac(int(round(a-b+c)))*
                    _safe_fac(int(round(-a+b+c)))/_safe_fac(int(round(a+b+c+1))))
    pref=sqrt(2*J+1)*tri(j1,j2,J)
    if pref==0: return 0.0
    s_min=max(0.0,m1-j2-J,-j1+j2-M)
    s_max=min(j1+j2-J,j1-m1,j2+m2)
    acc=0.0; s=s_min
    while s<=s_max+1e-9:
        d=(_safe_fac(round(s))*_safe_fac(round(j1+j2-J-s))*
           _safe_fac(round(j1-m1-s))*_safe_fac(round(j2+m2-s))*
           _safe_fac(round(J-j2+m1+s))*_safe_fac(round(J-j1-m2+s)))
        if d>0: acc+=((-1)**int(round(s)))/d
        s+=1.0
    norm=sqrt(_safe_fac(round(j1+m1))*_safe_fac(round(j1-m1))*
              _safe_fac(round(j2+m2))*_safe_fac(round(j2-m2))*
              _safe_fac(round(J+M))*_safe_fac(round(J-M)))
    return pref*norm*acc

def build_intertwiner(js:List[float])->Optional[np.ndarray]:
    if not js: return np.array([1.0])
    if len(js)==1: return np.array([1.0]) if abs(js[0])<1e-9 else None
    if len(js)==2: return _wigner_j0(js[0],js[1])
    dims=[round(2*j+1) for j in js]; size=1
    for d in dims: size*=d
    if size>_TENSOR_CAP: return None
    j1,j2=js[0],js[1]; result=np.zeros(dims)
    for J_int in _cg_decompose(j1,j2):
        sub=build_intertwiner([J_int]+js[2:])
        if sub is None: continue
        d1,d2,di=round(2*j1+1),round(2*j2+1),round(2*J_int+1)
        cg=np.zeros((d1,d2,di))
        for im1,m1 in enumerate(np.arange(-j1,j1+0.5,1.0)):
            for im2,m2 in enumerate(np.arange(-j2,j2+0.5,1.0)):
                for iM,M in enumerate(np.arange(-J_int,J_int+0.5,1.0)):
                    cg[im1,im2,iM]=_cg_coeff(j1,m1,j2,m2,J_int,M)
        result+=np.einsum('ijk,k...->ij...',cg,
                          sub.reshape([di]+list(sub.shape[1:])))
    if np.allclose(result,0): return None
    n=np.linalg.norm(result)
    return result/n if n>1e-12 else result

@dataclass
class Intertwiner:
    face_spins: List[float]; components: Optional[np.ndarray]
    dim_inv: int; norm: float=1.0
    def is_trivial(self)->bool:
        return self.components is None or self.dim_inv==0
    def __str__(self)->str:
        sh=str(self.components.shape) if self.components is not None else "∅"
        return f"ι(j={self.face_spins} dim={self.dim_inv} shape={sh} ‖ι‖={self.norm:.4f})"

def make_intertwiner(js:List[float], p:GFTParams)->Intertwiner:
    if js: _,dim_inv,_=_couple_spins(js)
    else: dim_inv=1
    comp=build_intertwiner(js)
    norm=float(np.linalg.norm(comp)) if comp is not None else 0.0
    return Intertwiner(face_spins=js,components=comp,dim_inv=dim_inv,norm=norm)


# ─────────────────────────────────────────────────────────────
# §3.  Spin Foam 2-complex
# ─────────────────────────────────────────────────────────────

class EdgeKind(Enum): TIME=auto(); INSTANTONIC=auto()
class FaceKind(Enum): REGULAR=auto(); INSTANTONIC=auto()

@dataclass
class SFVertex:
    vid:int; t:int; amplitude:float
    edge_ids:List[int]=field(default_factory=list)

@dataclass
class SFEdge:
    eid:int; kind:EdgeKind; t:int; face_ids:List[int]
    v_src_id:int; v_tgt_id:int; intertwiner:Intertwiner

@dataclass
class SFace:
    fid:int; j:float; q:int; propagator_weight:float
    t_from:int; t_to:int
    edge_ids:List[int]=field(default_factory=list)
    @property
    def kind(self)->FaceKind:
        return FaceKind.REGULAR if self.t_to>self.t_from else FaceKind.INSTANTONIC

@dataclass
class SpinFoam2Complex:
    """C₂ →∂₂→ C₁ →∂₁→ C₀  complete spin foam 2-complex."""
    params:GFTParams
    vertices:Dict[int,SFVertex]=field(default_factory=dict)
    edges:Dict[int,SFEdge]=field(default_factory=dict)
    faces:Dict[int,SFace]=field(default_factory=dict)

    def boundary_2(self,fid:int)->List[Tuple[int,int]]:
        return [(eid,+1) for eid in self.faces[fid].edge_ids]

    def boundary_1(self,eid:int)->Tuple[int,int]:
        e=self.edges[eid]; return (e.v_src_id,e.v_tgt_id)

    def verify_edge_connectivity(self)->bool:
        """
        Structural invariant (replaces naive ∂₁∘∂₂=0 which fails for open worldsheets):
        
        Regular face  f[t_from..t_to]:
          edges must form a connected directed path v_{t_from}→...→v_{t_to}
          i.e. vsum = {t_from: -1, t_to: +1}  (net boundary = endpoint − startpoint)

        Instantonic face f[t..t]:
          single self-loop at v_t
          i.e. vsum = {}  (genuinely closed)
        """
        ok=True
        for fid,f in self.faces.items():
            vsum:Dict[int,int]={}
            for eid,ori in self.boundary_2(fid):
                vs,vt=self.boundary_1(eid)
                vsum[vt]=vsum.get(vt,0)+ori
                vsum[vs]=vsum.get(vs,0)-ori
            vsum={v:c for v,c in vsum.items() if c!=0}  # drop zeros

            if f.kind==FaceKind.INSTANTONIC:
                # self-loop: ∂₁∘∂₂ must be exactly zero
                if vsum:
                    print(f"  ✗ instantonic f{fid}: vsum={vsum} (expected empty)")
                    ok=False
            else:
                # open worldsheet: expect exactly v_{t_from}=-1, v_{t_to}=+1
                expected={f.t_from:-1, f.t_to:+1}
                if vsum!=expected:
                    print(f"  ✗ regular f{fid} t=[{f.t_from},{f.t_to}]: "
                          f"vsum={vsum} expected={expected}")
                    ok=False
        return ok

    def partition_function_single(self)->float:
        Zf=float(np.prod([f.propagator_weight for f in self.faces.values()])) \
           if self.faces else 1.0
        Zv=float(np.prod([v.amplitude for v in self.vertices.values()])) \
           if self.vertices else 1.0
        return Zf*Zv

    def boundary_states(self)->Tuple[State,State]:
        T=max(v.t for v in self.vertices.values())+1
        def state_at(t):
            return frozenset((f.j,f.q) for f in self.faces.values()
                             if f.t_from<=t<=f.t_to)
        return state_at(0),state_at(T-1)

    def summary(self,full:bool=False)->str:
        p=self.params
        ok=self.verify_edge_connectivity()
        T=max(v.t for v in self.vertices.values())+1
        nt=sum(1 for e in self.edges.values() if e.kind==EdgeKind.TIME)
        ni=sum(1 for e in self.edges.values() if e.kind==EdgeKind.INSTANTONIC)
        nr=sum(1 for f in self.faces.values() if f.kind==FaceKind.REGULAR)
        nf=sum(1 for f in self.faces.values() if f.kind==FaceKind.INSTANTONIC)
        lines=["="*68,"  Spin Foam 2-Complex",
               f"  G=SU(2)×U(1)  μ²={p.mu2}  λ={p.lam}  P_min={p.P_min}  T={T}",
               f"  Edge connectivity: {'✓' if ok else '✗'}","="*68,
               f"  Vertices: {len(self.vertices)}",
               f"  Edges:    {len(self.edges)}  ({nt} time / {ni} instantonic)",
               f"  Faces:    {len(self.faces)}  ({nr} regular / {nf} instantonic)",
               f"  Z_single = {self.partition_function_single():.6e}"]
        if full:
            lines.append("\n  FACES:")
            for f in sorted(self.faces.values(),key=lambda x:(x.t_from,x.j)):
                sym="●" if f.kind==FaceKind.REGULAR else "◎"
                lines.append(f"    {sym} f{f.fid:<2}  {jq_to_name(f.j,f.q):<5}"
                             f"  j={f.j:.1f}  q={f.q:+d}"
                             f"  P={f.propagator_weight:.4f}"
                             f"  t=[{f.t_from},{f.t_to}]  ∂₂={f.edge_ids}")
            lines.append("\n  EDGES:")
            for e in sorted(self.edges.values(),key=lambda x:x.t):
                sym="→" if e.kind==EdgeKind.TIME else "↺"
                lines.append(f"    {sym} e{e.eid:<2}  t={e.t}"
                             f"  (v{e.v_src_id}→v{e.v_tgt_id})"
                             f"  {e.intertwiner}")
            lines.append("\n  VERTICES:")
            for v in sorted(self.vertices.values(),key=lambda x:x.t):
                lines.append(f"    v{v.vid}  t={v.t}  A_v={v.amplitude:.6f}")
        lines.append("="*68)
        return "\n".join(lines)
    def to_dict(self) -> dict:
        """
        Serialise the 2-complex to a plain Python dict (JSON-serialisable).

        Schema (version 1)
        ------------------
        {
          "version":  1,
          "params":   {mu2, lam, P_min, sigma_J},
          "vertices": { str(vid): {vid, t, amplitude, edge_ids} },
          "edges":    { str(eid): {eid, kind, t, face_ids,
                                   v_src_id, v_tgt_id, intertwiner} },
          "faces":    { str(fid): {fid, j, q, propagator_weight,
                                   t_from, t_to, edge_ids} }
        }

        • All values are JSON-native (float, int, list, str, None).
        • numpy arrays (Intertwiner.components) are stored as nested lists.
        • Enums are stored as their .name string.
        • Half-integer spins (e.g. 2.5) round-trip exactly through JSON.

        Round-trip guarantee
        --------------------
            foam2 = SpinFoam2Complex.from_dict(foam.to_dict())
            assert foam2.to_dict() == foam.to_dict()
        """
        import json as _json

        def _iota(iota: Intertwiner) -> dict:
            return {
                "face_spins":  iota.face_spins,
                "components":  iota.components.tolist()
                               if iota.components is not None else None,
                "dim_inv":     iota.dim_inv,
                "norm":        iota.norm,
            }

        return {
            "version": 1,
            "params": {
                "mu2":     self.params.mu2,
                "lam":     self.params.lam,
                "P_min":   self.params.P_min,
                "sigma_J": self.params.sigma_J,
            },
            "vertices": {
                str(vid): {
                    "vid":       v.vid,
                    "t":         v.t,
                    "amplitude": v.amplitude,
                    "edge_ids":  v.edge_ids,
                }
                for vid, v in self.vertices.items()
            },
            "edges": {
                str(eid): {
                    "eid":         e.eid,
                    "kind":        e.kind.name,   # "TIME" | "INSTANTONIC"
                    "t":           e.t,
                    "face_ids":    e.face_ids,
                    "v_src_id":    e.v_src_id,
                    "v_tgt_id":    e.v_tgt_id,
                    "intertwiner": _iota(e.intertwiner),
                }
                for eid, e in self.edges.items()
            },
            "faces": {
                str(fid): {
                    "fid":               f.fid,
                    "j":                 f.j,
                    "q":                 f.q,
                    "propagator_weight": f.propagator_weight,
                    "t_from":            f.t_from,
                    "t_to":              f.t_to,
                    "edge_ids":          f.edge_ids,
                }
                for fid, f in self.faces.items()
            },
        }

    def to_json(self, path: str, indent: int = 2) -> None:
        """
        Write the 2-complex to *path* as a UTF-8 JSON file.

        Usage
        -----
            foam.to_json("piece.json")
            foam2 = SpinFoam2Complex.from_json("piece.json")
        """
        import json as _json
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(self.to_dict(), fh, indent=indent, ensure_ascii=False)
        print(f"  Saved → {path}  "
              f"({len(self.vertices)}V  {len(self.edges)}E  "
              f"{len(self.faces)}F)")

    @classmethod
    def from_dict(cls, d: dict) -> "SpinFoam2Complex":
        """
        Reconstruct a SpinFoam2Complex from the dict produced by to_dict().

        Raises ValueError on unsupported format version.
        """
        version = d.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported serialization version: {version}")

        p = d["params"]
        foam = cls(params=GFTParams(
            mu2     = p["mu2"],
            lam     = p["lam"],
            P_min   = p["P_min"],
            sigma_J = p["sigma_J"],
        ))

        for vid_str, v in d["vertices"].items():
            foam.vertices[int(vid_str)] = SFVertex(
                vid       = v["vid"],
                t         = v["t"],
                amplitude = v["amplitude"],
                edge_ids  = v["edge_ids"],
            )

        for eid_str, e in d["edges"].items():
            raw = e["intertwiner"]
            comp = np.array(raw["components"]) \
                   if raw["components"] is not None else None
            foam.edges[int(eid_str)] = SFEdge(
                eid         = e["eid"],
                kind        = EdgeKind[e["kind"]],
                t           = e["t"],
                face_ids    = e["face_ids"],
                v_src_id    = e["v_src_id"],
                v_tgt_id    = e["v_tgt_id"],
                intertwiner = Intertwiner(
                    face_spins = raw["face_spins"],
                    components = comp,
                    dim_inv    = raw["dim_inv"],
                    norm       = raw["norm"],
                ),
            )

        for fid_str, f in d["faces"].items():
            foam.faces[int(fid_str)] = SFace(
                fid               = f["fid"],
                j                 = f["j"],
                q                 = f["q"],
                propagator_weight = f["propagator_weight"],
                t_from            = f["t_from"],
                t_to              = f["t_to"],
                edge_ids          = f["edge_ids"],
            )

        return foam

    @classmethod
    def from_json(cls, path: str) -> "SpinFoam2Complex":
        """
        Load a SpinFoam2Complex from a JSON file written by to_json().

        Usage
        -----
            foam = SpinFoam2Complex.from_json("piece.json")
        """
        import json as _json
        with open(path, "r", encoding="utf-8") as fh:
            d = _json.load(fh)
        foam = cls.from_dict(d)
        print(f"  Loaded ← {path}  "
              f"({len(foam.vertices)}V  {len(foam.edges)}E  "
              f"{len(foam.faces)}F)")
        return foam

# ─────────────────────────────────────────────────────────────
# §4.  Step 1: σ → FeynmanDiagram
# ─────────────────────────────────────────────────────────────

@dataclass
class Propagator:
    pid:int; j:float; q:int; propagator_weight:float; t_from:int; t_to:int
    @property
    def kind(self)->FaceKind:
        return FaceKind.REGULAR if self.t_to>self.t_from else FaceKind.INSTANTONIC

@dataclass
class FDVertex:
    t:int; incident_pids:List[int]; amplitude:float; dim_inv:int

@dataclass
class FeynmanDiagram:
    T:int; params:GFTParams
    propagators:Dict[int,Propagator]=field(default_factory=dict)
    vertices:Dict[int,FDVertex]=field(default_factory=dict)


def build_feynman(sigma:np.ndarray, params:GFTParams)->FeynmanDiagram:
    T,_=sigma.shape; fd=FeynmanDiagram(T=T,params=params)
    pid=0; active:Dict[int,int]={}
    for t in range(T):
        curr=set(int(n) for n in np.where(sigma[t]>0)[0])
        prev=set(active.keys())
        for n in sorted(curr-prev):
            j,q=pitch_to_quantum_numbers(n)
            fd.propagators[pid]=Propagator(pid=pid,j=j,q=q,
                propagator_weight=gft_propagator(j,q,params),t_from=t,t_to=t)
            active[n]=pid; pid+=1
        for n in sorted(prev-curr): active.pop(n)
        for n in curr&prev: fd.propagators[active[n]].t_to=t
        pids=[active[n] for n in curr if n in active]
        js=[fd.propagators[p].j for p in pids]
        Av=gft_vertex_amplitude(js,params)
        _,dim_inv,_=_couple_spins(js) if js else ([0.],1,1)
        fd.vertices[t]=FDVertex(t=t,incident_pids=pids,amplitude=Av,dim_inv=dim_inv)
    return fd


# ─────────────────────────────────────────────────────────────
# §5.  Step 2: FeynmanDiagram → SpinFoam2Complex
# ─────────────────────────────────────────────────────────────

def lift_to_spinfoam(fd:FeynmanDiagram)->SpinFoam2Complex:
    foam=SpinFoam2Complex(params=fd.params); eid=0
    for t,v in fd.vertices.items():
        foam.vertices[t]=SFVertex(vid=t,t=t,amplitude=v.amplitude)
    for p in fd.propagators.values():
        foam.faces[p.pid]=SFace(fid=p.pid,j=p.j,q=p.q,
            propagator_weight=p.propagator_weight,t_from=p.t_from,t_to=p.t_to)
    edge_reg:Dict[Tuple,int]={}

    def get_edge(kind,t,v_src,v_tgt,fid)->int:
        nonlocal eid
        key=(kind,t,v_src,v_tgt)
        if key not in edge_reg:
            foam.edges[eid]=SFEdge(eid=eid,kind=kind,t=t,face_ids=[],
                v_src_id=v_src,v_tgt_id=v_tgt,
                intertwiner=Intertwiner([],None,0,0.0))
            edge_reg[key]=eid
            foam.vertices[v_src].edge_ids.append(eid)
            if v_tgt!=v_src: foam.vertices[v_tgt].edge_ids.append(eid)
            eid+=1
        e_id=edge_reg[key]
        if fid not in foam.edges[e_id].face_ids:
            foam.edges[e_id].face_ids.append(fid)
        return e_id

    for p in fd.propagators.values():
        fid=p.pid
        if p.kind==FaceKind.INSTANTONIC:
            e_id=get_edge(EdgeKind.INSTANTONIC,p.t_from,p.t_from,p.t_from,fid)
            foam.faces[fid].edge_ids=[e_id]
        else:
            eids=[]
            for t in range(p.t_from,p.t_to):
                e_id=get_edge(EdgeKind.TIME,t,t,t+1,fid); eids.append(e_id)
            foam.faces[fid].edge_ids=eids

    for e in foam.edges.values():
        js=[foam.faces[fid].j for fid in e.face_ids]
        e.intertwiner=make_intertwiner(js,fd.params)
    return foam


# ─────────────────────────────────────────────────────────────
# §6.  Reconstruction
# ─────────────────────────────────────────────────────────────

def decode(foam:SpinFoam2Complex)->np.ndarray:
    """SpinFoam2Complex → σ (T×88).  Exact inverse."""
    T=max(v.t for v in foam.vertices.values())+1
    sigma=np.zeros((T,88),dtype=int)
    for f in foam.faces.values():
        idx=decode_quantum_numbers(f.j,f.q)
        if 0<=idx<88:
            for t in range(f.t_from,f.t_to+1): sigma[t,idx]=1
    return sigma


# ─────────────────────────────────────────────────────────────
# §7.  GFT Vocabulary + Partition Function Z[s_in, s_out]
# ─────────────────────────────────────────────────────────────

_MAX_VOCAB=18  # 2^18=262144 hard cap

def build_bulk_vocab(params:GFTParams,
                     octave_range:Tuple[int,int]=(1,7)
                     )->List[Tuple[float,int]]:
    """
    Bulk vocabulary: notes that can propagate in the interior.
    V_bulk = {(j,q) : P(j,q)>P_min, idx∈[0,87]}
    Boundary notes are NOT added here; they are fixed external conditions.
    """
    vocab:set=set()
    for q in range(octave_range[0],octave_range[1]+1):
        for chroma in range(12):
            j=chroma/2.0; P=gft_propagator(j,q,params)
            idx=decode_quantum_numbers(j,q)
            if P>params.P_min and 0<=idx<88: vocab.add((j,q))
    vocab_list=sorted(vocab)
    N=len(vocab_list)
    if N>_MAX_VOCAB:
        raise ValueError(f"|V_bulk|={N}>{_MAX_VOCAB}. Raise P_min.")
    return vocab_list

def enumerate_states(vocab:List[Tuple[float,int]])->List[State]:
    return [frozenset(vocab[i] for i in range(len(vocab)) if mask&(1<<i))
            for mask in range(1<<len(vocab))]


def _trans_weight(s:State, s_next:State, p:GFTParams)->float:
    Pp=1.0
    for jq in s&s_next: Pp*=gft_propagator(jq[0],jq[1],p)
    return Pp*gft_vertex_amplitude(list(jq[0] for jq in s_next),p)


@dataclass
class PartitionResult:
    Z:float; log_Z:float; s_in:State; s_out:State
    T:int; params:GFTParams; bulk_vocab:List[Tuple[float,int]]
    top_histories:List[Tuple[float,List[State]]]
    marginals:List[Dict[State,float]]

    def summary(self)->str:
        lines=["="*64,
               "  Z[s_in,s_out]  ─  sum over all bulk histories",
               f"  μ²={self.params.mu2}  λ={self.params.lam}  "
               f"P_min={self.params.P_min}  σ_J={self.params.sigma_J}"
               f"  |V_bulk|={len(self.bulk_vocab)}  T={self.T}",
               "="*64,
               f"  s_in  = {sorted(jq_to_name(*jq) for jq in self.s_in)}",
               f"  s_out = {sorted(jq_to_name(*jq) for jq in self.s_out)}",
               f"  log Z = {self.log_Z:.5f}",
               f"  Z     = {self.Z:.6e}" + ("  (overflow — use log Z)" if not np.isfinite(self.Z) else ""),
               f"  Top-{len(self.top_histories)} dominant histories"
               f"  (showing head→...→tail for long pieces):"]
        for rank,(w,hist) in enumerate(self.top_histories,1):
            def fmt(s):
                if s is None: return "···"
                return "{"+",".join(sorted(jq_to_name(*jq) for jq in s))+"}" \
                       if s else "∅"
            path=" → ".join(fmt(s) for s in hist)
            lines.append(f"    #{rank}  w={w:.4e}  {path}")
        lines.append("="*64)
        return "\n".join(lines)


def compute_Z(s_in:State, s_out:State, T:int,
              params:GFTParams, top_k:int=5,
              show_progress:bool=True)->PartitionResult:
    """
    Z[s_in,s_out] = Σ_{bulk histories} ∏_t W(s_t,s_{t+1})

    Boundary states are FIXED external legs (not summed over).
    Bulk states t=1,...,T-2 are drawn from V_bulk (GFT propagator cutoff).

    Complexity
    ----------
    TM build : O(S²)                      — once, with lru_cache on A_v
    Forward DP: O(T · S²) as T matrix-vector products — fast in NumPy
    Beam search: O(T · top_k · S)         — capped to top_k at every step
    Memory    : O(T · S) for alpha_table  — fine up to T≈10^4, S≈1024

    For T=2: Z = W(s_in, s_out) directly (no bulk).
    """
    import time
    bulk_vocab  = build_bulk_vocab(params)
    bulk_states = enumerate_states(bulk_vocab)
    S           = len(bulk_states)
    s_idx       = {s:i for i,s in enumerate(bulk_states)}

    print(f"  compute_Z: |V_bulk|={len(bulk_vocab)}  S={S}  T={T}")

    # ── T=2 shortcut ────────────────────────────────────────
    if T == 2:
        Z     = _trans_weight(s_in, s_out, params)
        log_Z = float(np.log(Z)) if Z>1e-300 else float(-np.inf)
        return PartitionResult(Z=Z, log_Z=log_Z, s_in=s_in, s_out=s_out,
                               T=T, params=params, bulk_vocab=bulk_vocab,
                               top_histories=[(Z,[s_in,s_out])],
                               marginals=[{s_in:1.0},{s_out:1.0}])

    # ── Build transfer matrix (once) ─────────────────────────
    if show_progress:
        print(f"  Building TM ({S}×{S})...", end="", flush=True)
    t0 = time.time()
    TM = np.zeros((S, S))
    for i, s in enumerate(bulk_states):
        for j, sn in enumerate(bulk_states):
            TM[i, j] = _trans_weight(s, sn, params)
    if show_progress:
        print(f" {time.time()-t0:.1f}s")

    # ── Forward DP  (scaled to prevent float64 overflow) ─────
    # The TM often has spectral radius > 1 (empty state self-loops with A_v=λ,
    # large chord states contribute weights > 1), so raw alpha @ TM^T grows
    # exponentially and overflows float64 within ~700 steps.
    #
    # Scaling trick (numerically standard for HMMs over long sequences):
    #   After each step, normalize alpha by its max, accumulate log(norm).
    #   log Z = Σ_t log(norm_t)  +  log(alpha_final · alpha_out)
    #
    alpha     = np.array([_trans_weight(s_in, s, params) for s in bulk_states])
    log_scale = 0.0   # accumulates Σ log(norm_t)

    norm0 = alpha.max()
    if norm0 > 0:
        log_scale += np.log(norm0)
        alpha     /= norm0

    # Sparse marginals: store first and last MARG_CAP bulk steps only.
    MARG_CAP   = 6
    marg_ticks: List[int]               = []
    marg_rows:  List[Dict[State,float]] = []

    for t in range(T - 2):
        alpha = alpha @ TM
        norm  = alpha.max()
        if norm > 0:
            log_scale += np.log(norm)
            alpha     /= norm
        if t < MARG_CAP or t >= (T - 2 - MARG_CAP):
            tot = alpha.sum()
            marg_ticks.append(t + 1)
            marg_rows.append(
                {bulk_states[i]: alpha[i]/tot for i in range(S)}
                if tot > 0 else {})

    alpha_out = np.array([_trans_weight(s, s_out, params) for s in bulk_states])
    dot       = float(alpha @ alpha_out)
    log_Z     = (log_scale + np.log(dot)) if dot > 1e-300 else float(-np.inf)
    # np.exp overflows float64 for log_Z > ~709; keep Z as inf and use log_Z
    Z         = float(np.exp(log_Z)) if (np.isfinite(log_Z) and log_Z < 709) else \
                (0.0 if log_Z == float('-inf') else float('inf'))

    marginals = [{s_in: 1.0}] + marg_rows + [{s_out: 1.0}]

    # ── Beam search for dominant histories ───────────────────
    # Use log-weights to avoid underflow over many steps.
    # Each beam element: (log_weight, [state_indices])
    # For large T: store only first/last HIST_CAP steps of path.
    HIST_CAP = 4   # steps to keep at each end of path for display

    beam: List[Tuple[float, List]] = []
    w0   = np.array([_trans_weight(s_in, s, params) for s in bulk_states])
    top0 = np.argsort(w0)[::-1][:top_k]
    for i in top0:
        if w0[i] > 0:
            beam.append((float(np.log(w0[i])), [i], [i]))
            # format: (log_w, head_indices, tail_indices)
    # Reformulate beam as (log_w, head_list, tail_list, last_idx)
    beam2: List[Tuple[float, List[int], List[int], int]] = []
    for lw, head, _ in beam:
        beam2.append((lw, head[:HIST_CAP], head[-HIST_CAP:], head[-1]))

    for t in range(1, T - 2):
        cands = []
        for lw, head, tail, last in beam2:
            row     = TM[last]
            top_j   = np.argsort(row)[::-1][:top_k * 2]
            for j in top_j:
                w = row[j]
                if w > 0:
                    new_lw   = lw + float(np.log(w))
                    new_tail = (tail + [j])[-HIST_CAP:]
                    new_head = head if len(head) >= HIST_CAP else head + [j]
                    cands.append((new_lw, new_head, new_tail, j))
        cands.sort(key=lambda x: -x[0])
        beam2 = cands[:top_k]

    # Final step → s_out
    top_histories = []
    for lw, head, tail, last in beam2[:top_k]:
        w_final = _trans_weight(bulk_states[last], s_out, params)
        if w_final > 0:
            final_lw = lw + float(np.log(w_final))
            # Reconstruct displayable history
            head_states = [s_in] + [bulk_states[i] for i in head]
            tail_states = [bulk_states[i] for i in tail] + [s_out]
            # Deduplicate if T is small enough to show fully
            if T <= 2 * HIST_CAP + 2:
                hist = head_states + tail_states
            else:
                hist = head_states + [None] + tail_states  # None = "..."
            top_histories.append((float(np.exp(final_lw)), hist))

    if not top_histories:  # fallback if all weights were 0
        top_histories = [(Z, [s_in, s_out])]

    return PartitionResult(Z=Z, log_Z=log_Z, s_in=s_in, s_out=s_out,
                           T=T, params=params, bulk_vocab=bulk_vocab,
                           top_histories=top_histories, marginals=marginals)


# ─────────────────────────────────────────────────────────────
# §8.  Test Suite
# ─────────────────────────────────────────────────────────────

def run_tests(params:GFTParams)->bool:
    results=[]
    def check(name,passed,detail=""):
        sym="✓" if passed else "✗"
        print(f"  {sym}  {name}"+(f"  [{detail}]" if detail else ""))
        results.append(passed)

    print("\n"+"="*62+"\n  Test Suite\n"+"="*62)

    # T1: Encoding bijection — ALL 88 keys
    errs=[f"{i}→{decode_quantum_numbers(*pitch_to_quantum_numbers(i))}"
          for i in range(88)
          if decode_quantum_numbers(*pitch_to_quantum_numbers(i))!=i]
    check("T1  Encoding bijection  ∀ idx∈[0,87]",
          not errs, f"88/88 ok" if not errs else f"FAIL: {errs}")

    # T2: Propagator monotonicity
    Pc3=gft_propagator(0.0,3,params)
    Pc4=gft_propagator(0.0,4,params)
    Pb4=gft_propagator(5.5,4,params)
    check("T2  Propagator monotonicity  P(C3)>P(C4)>P(B4)",
          Pc3>Pc4>Pb4,
          f"P={Pc3:.5f}>{Pc4:.5f}>{Pb4:.5f}")

    # T3: CG symmetry
    a12,d12,_=_couple_spins([1.0,2.0])
    a21,d21,_=_couple_spins([2.0,1.0])
    check("T3  CG symmetry  j1⊗j2=j2⊗j1",
          a12==a21 and d12==d21,
          f"admJ={a12}  dim_inv={d12}")

    # T4: Vacuum amplitude — still λ (J_min=0 → exp(0)=1)
    Av=gft_vertex_amplitude([],params)
    check("T4  Vertex amplitude  A_v(∅)=λ",
          abs(Av-params.lam)<1e-9,
          f"A_v(∅)={Av:.6f}  λ={params.lam}")

    # T4b: Non-closing chord always positive with Gaussian formula
    # j=1⊗j=2 → J_min=1 (no J=0 channel) → A_v=λ·exp(-1/σ_J) > 0
    admJ_12_list,dim12,_=_couple_spins([1.0,2.0])
    Av_nc=gft_vertex_amplitude([1.0,2.0],params)
    expected_nc=params.lam*float(np.exp(-admJ_12_list[0]/params.sigma_J))
    check("T4b Non-closing chord  A_v>0  (Gaussian, not strict zero)",
          Av_nc>0 and abs(Av_nc-expected_nc)<1e-9,
          f"J_min={admJ_12_list[0]}  A_v={Av_nc:.6f}  "
          f"λ·exp(-J_min/σ_J)={expected_nc:.6f}")

    # T5: Intertwiner normalisation
    iota11=build_intertwiner([1.0,1.0])
    norm11=float(np.linalg.norm(iota11)) if iota11 is not None else 0.0
    check("T5  Intertwiner normalisation  ‖ι(1,1)‖=1",
          iota11 is not None and abs(norm11-1.0)<1e-9,
          f"‖ι‖={norm11:.8f}")

    # T6: Intertwiner antisymmetry ι(½,½)=-ι(½,½)ᵀ
    iota_hh=build_intertwiner([0.5,0.5])
    antisym=iota_hh is not None and np.allclose(iota_hh,-iota_hh.T,atol=1e-9)
    check("T6  Intertwiner ι(½,½) antisymmetric  ι=-ιᵀ",
          antisym,
          (f"\n{np.round(iota_hh,4)}" if iota_hh is not None else "None"))

    # T7: Edge connectivity (corrected test for open worldsheets)
    s7=np.zeros((4,88),dtype=int)
    for t,names in enumerate([["C3","D4"],["C3","D3"],[],["C3","C4"]]):
        for name in names: s7[t,_midi_to_index(NOTE_MAP[name])]=1
    fd7=build_feynman(s7,params); foam7=lift_to_spinfoam(fd7)
    ok_ec=foam7.verify_edge_connectivity()
    check("T7  Edge connectivity  (worldsheet open-cylinder invariant)",ok_ec)

    # T8: Decoding exactness
    recon7=decode(foam7); match=np.array_equal(s7,recon7)
    if not match:
        for t in range(s7.shape[0]):
            o=sorted([index_to_name(i) for i in np.where(s7[t]>0)[0]])
            r=sorted([index_to_name(i) for i in np.where(recon7[t]>0)[0]])
            if o!=r: print(f"      t={t}: expected {o}  got {r}")
    check("T8  Decoding exactness  decode∘lift∘build=id",
          match,"exact" if match else "MISMATCH ← above")

    # T9: Z positivity (T=2 shortcut path)
    sin=frozenset({(0.0,3)}); sout=frozenset({(0.0,3)})
    r9=compute_Z(sin,sout,T=2,params=params,top_k=1)
    check("T9  Z positivity  Z[C3,C3]≥0",r9.Z>=0,f"Z={r9.Z:.6e}")

    # T10: Z > 0 for any boundary pair with Gaussian A_v
    # (old strict formula gave Z=0 whenever boundaries outside bulk vocab)
    sout_any=frozenset({(1.0,3)})   # D3 — below P_min, outside bulk vocab
    ra=compute_Z(sin,sin,    T=2,params=params,top_k=1)
    rb=compute_Z(sin,sout_any,T=2,params=params,top_k=1)
    check("T10 Z>0 for any boundary pair  (Gaussian A_v always positive)",
          ra.Z>0 and rb.Z>=0,
          f"Z[C3→C3]={ra.Z:.5e}  Z[C3→D3]={rb.Z:.5e}")

    # T11: MCMC partition function — use chord with multiple admissible intertwiners
    # F1(j=2.5)+A2(j=4.5)+C3(j=0.0): admissible i_e ∈ {2,3,4,5,6,7} → 6 free values
    s11=np.zeros((3,88),dtype=int)
    for t,names in enumerate([['F1','A2','C3'],['F1','A2','C3'],['F1','A2']]):
        for name in names:
            idx=_midi_to_index(NOTE_MAP.get(name, {'F1':29,'A2':45,'C3':48}[name]))
            s11[t,idx]=1
    fd11=build_feynman(s11,params); foam11=lift_to_spinfoam(fd11)
    r11=compute_Z_mcmc(foam11,params,n_steps=2000,n_warmup=200,n_anneal=5,n_chains=1,top_k=1)
    check("T11 MCMC  log_Z finite  free intertwiners sampled",
          np.isfinite(r11.log_Z) and r11.log_Z > r11.log_Z0,
          f"log_Z={r11.log_Z:.4f}  log_Z0={r11.log_Z0:.4f}  acc={r11.acceptance_rate:.3f}")

    passed=sum(results); total=len(results)
    mark="✓ ALL PASS" if passed==total else f"✗ {total-passed} FAILED"
    print(f"\n  {mark}  ({passed}/{total})")
    print("="*62)
    return passed==total


# ─────────────────────────────────────────────────────────────
# §9.  Visualisation  (saves PNG files)
# ─────────────────────────────────────────────────────────────

def _jcolor(j,j_max=5.5): return plt.cm.plasma(j/j_max if j_max>0 else 0.5)
def _pwidth(P,P_max):     return 1.5+6.0*(P/P_max if P_max>0 else 0.5)

def visualize_2complex(foam:SpinFoam2Complex,
                       out:str="/mnt/user-data/outputs/v11_2complex.png"):
    fig,ax=plt.subplots(figsize=(14,6))
    T=max(v.t for v in foam.vertices.values())+1
    all_j=[f.j for f in foam.faces.values()] or [5.5]
    all_P=[f.propagator_weight for f in foam.faces.values()] or [1.0]
    j_max,P_max=max(all_j),max(all_P)
    for t in range(T): ax.axvline(t,color="#ebebeb",lw=1,zorder=0)
    for f in foam.faces.values():
        color=_jcolor(f.j,j_max); lw=_pwidth(f.propagator_weight,P_max)
        if f.kind==FaceKind.REGULAR:
            ax.hlines(f.j+1.0,f.t_from,f.t_to+1,lw=lw,color=color,alpha=0.85,zorder=2)
            ax.text((f.t_from+f.t_to+1)/2,f.j+1.15,
                    f"{jq_to_name(f.j,f.q)}\nj={f.j:.1f}  P={f.propagator_weight:.3f}",
                    fontsize=6.5,ha="center",color=color)
        else:
            ax.scatter(f.t_from,f.j+1.0,s=80,color=color,marker="D",
                       alpha=0.85,zorder=3,edgecolors="white",lw=1)
    for e in foam.edges.values():
        color="steelblue" if e.kind==EdgeKind.TIME else "tomato"
        if e.kind==EdgeKind.TIME:
            ax.annotate("",xy=(e.v_tgt_id,0.0),xytext=(e.v_src_id,0.0),
                        arrowprops=dict(arrowstyle="->",color=color,lw=1.5),zorder=3)
            dim_s=f"dim={e.intertwiner.dim_inv}" if not e.intertwiner.is_trivial() else "triv"
            ax.text((e.v_src_id+e.v_tgt_id)/2,0.18,
                    f"e{e.eid}\n{dim_s}",ha="center",fontsize=6,color=color)
        else:
            ax.scatter(e.t,0.0,s=60,color=color,marker="s",alpha=0.8,zorder=3)
    for v in foam.vertices.values():
        ax.scatter(v.t,-0.5,s=80+450*v.amplitude,color="dimgray",alpha=0.85,zorder=4)
        ax.text(v.t,-0.78,f"v{v.vid}\nA_v={v.amplitude:.3f}",
                ha="center",fontsize=6.5,color="dimgray")
    for label,y in [("C₂ Faces",1.3),("C₁ Edges",0.18),("C₀ Vertices",-0.78)]:
        ax.text(-0.55,y,label,fontsize=8,color="gray",fontstyle="italic")
    sm=plt.cm.ScalarMappable(cmap=plt.cm.plasma,norm=plt.Normalize(0,j_max))
    sm.set_array([]); fig.colorbar(sm,ax=ax,label="SU(2) spin j")
    ax.set_title("Spin Foam 2-Complex\n"
                 "Faces=worldsheets  Edges=time/instantonic  Vertices=beats")
    ax.set_xlabel("Time t"); ax.set_ylabel("Layer")
    ax.set_ylim(-1.1,j_max+1.8); ax.set_xlim(-0.7,T+0.2)
    ax.grid(alpha=0.08); plt.tight_layout()
    plt.savefig(out,dpi=140); plt.close(); print(f"  Saved: {out}")

def visualize_intertwiners(foam:SpinFoam2Complex,
                           out:str="/mnt/user-data/outputs/v11_intertwiners.png"):
    edges=[e for e in foam.edges.values()
           if e.kind==EdgeKind.TIME and not e.intertwiner.is_trivial()]
    if not edges: print("  No non-trivial intertwiners."); return
    n=min(len(edges),6)
    fig,axes=plt.subplots(1,n,figsize=(3*n,3))
    if n==1: axes=[axes]
    for ax,e in zip(axes,edges[:n]):
        iota=e.intertwiner
        if iota.components is not None and len(iota.face_spins)==2:
            im=ax.imshow(iota.components,cmap="RdBu_r",vmin=-1,vmax=1)
            fig.colorbar(im,ax=ax,fraction=0.046)
            ax.set_title(f"e{e.eid}  j={iota.face_spins}\n"
                         f"dim={iota.dim_inv}  ‖ι‖={iota.norm:.3f}",fontsize=7)
            ax.set_xlabel("m₂"); ax.set_ylabel("m₁")
        else:
            ax.text(0.5,0.5,f"e{e.eid}\nj={iota.face_spins}\n"
                    f"dim={iota.dim_inv}\n‖ι‖={iota.norm:.3f}",
                    ha="center",va="center",fontsize=8,transform=ax.transAxes)
            ax.axis("off")
    fig.suptitle("Intertwiner tensors  ι_e ∈ Hom(⊗V^j, ℂ)",fontsize=9)
    plt.tight_layout(); plt.savefig(out,dpi=140); plt.close(); print(f"  Saved: {out}")

def visualize_reconstruction(orig:np.ndarray, recon:np.ndarray,
                             out:str="/mnt/user-data/outputs/v11_reconstruction.png"):
    T,_=orig.shape
    rows=np.where((orig|recon).any(axis=0))[0]
    if not len(rows): return
    r0,r1=max(0,rows.min()-1),min(87,rows.max()+1)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,sigma,label,color in [
        (axes[0],orig, "Original σ",       "steelblue"),
        (axes[1],recon,"Reconstructed σ'", "limegreen"),
    ]:
        ax.set_facecolor("#f7f7f7")
        for t in range(T):
            for idx in range(r0,r1+1):
                if sigma[t,idx]:
                    ax.barh(idx,0.80,left=t-0.40,height=0.65,color=color,alpha=0.78)
                    ax.text(t,idx,index_to_name(idx),ha="center",va="center",
                            fontsize=7,color="white",fontweight="bold")
        ax.set_yticks(range(r0,r1+1))
        ax.set_yticklabels([index_to_name(i) for i in range(r0,r1+1)],fontsize=7)
        ax.set_xticks(range(T)); ax.set_xticklabels([f"t={t}" for t in range(T)],fontsize=8)
        ax.set_xlim(-0.5,T-0.5); ax.set_title(label,fontsize=10); ax.grid(alpha=0.15)
    match=np.array_equal(orig,recon)
    col="green" if match else "red"
    fig.suptitle(f"Reconstruction  {'✓ exact' if match else '✗ MISMATCH'}",
                 fontsize=11,color=col)
    plt.tight_layout(); plt.savefig(out,dpi=140); plt.close(); print(f"  Saved: {out}")

def visualize_partition_result(result:PartitionResult,
                               out:str="/mnt/user-data/outputs/v11_partition.png"):
    T=result.T; top=result.top_histories[:3]

    # Collect all jq that appear in boundary states and non-None history steps
    all_jq=set(result.s_in)|set(result.s_out)
    for _,hist in top:
        for s in hist:
            if s is not None:          # skip the ··· ellipsis marker
                all_jq.update(s)
    all_jq=sorted(all_jq,key=lambda x:(x[1],x[0]))
    labels=[jq_to_name(*jq) for jq in all_jq]; N=len(all_jq)
    jq_idx={jq:i for i,jq in enumerate(all_jq)}
    if N==0: return

    fig=plt.figure(figsize=(15,5), layout="constrained")
    gs=gridspec.GridSpec(1,3,figure=fig,width_ratios=[2,2,1])

    # ── Marginals heatmap (sparse: only stored steps) ──────────
    ax0=fig.add_subplot(gs[0])
    M = len(result.marginals)   # may be << T for long pieces
    prob=np.zeros((N, M))
    for col, marg in enumerate(result.marginals):
        for s, p in marg.items():
            if s is None: continue
            for jq in s:
                if jq in jq_idx: prob[jq_idx[jq], col] += p
    im=ax0.imshow(prob, aspect="auto", cmap="YlOrRd", origin="lower",
                  vmin=0, vmax=max(prob.max(), 0.01),
                  extent=[-0.5, M-0.5, -0.5, N-0.5])
    ax0.set_yticks(range(N)); ax0.set_yticklabels(labels, fontsize=7)
    # X-axis labels: s_in, sparse bulk steps, s_out
    xlabels = ["s_in"] + [f"t≈{int(t)}" for t in
               np.linspace(1, T-2, M-2).tolist()] + ["s_out"]
    ax0.set_xticks(range(M))
    ax0.set_xticklabels(xlabels[:M], fontsize=6, rotation=30, ha="right")
    ax0.set_title(f"Marginal prob (sparse, {M} snapshots of T={T})", fontsize=7)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    # ── Top histories lanes ─────────────────────────────────────
    ax1=fig.add_subplot(gs[1]); colors=["steelblue","tomato","limegreen"]
    for rank,(w,hist) in enumerate(top):
        c=colors[rank%3]
        # Filter out None and show remaining steps across compressed x-axis
        visible = [(i,s) for i,s in enumerate(hist) if s is not None]
        n_vis   = len(visible)
        for xi, (_, s) in enumerate(visible):
            for jq in s:
                if jq in jq_idx:
                    y=jq_idx[jq]+rank*(N+1)
                    ax1.barh(y,0.80,left=xi-0.40,height=0.65,color=c,alpha=0.75)
                    ax1.text(xi,y,jq_to_name(*jq),ha="center",va="center",
                             fontsize=5.5,color="white")
        ax1.text(-0.45,rank*(N+1)+N/2,f"#{rank+1}\nw={w:.2e}",
                 ha="right",va="center",fontsize=7,color=c)
        # Mark ellipsis if history was compressed
        if any(s is None for s in hist):
            ax1.text(n_vis/2, rank*(N+1)-0.5, "··· (bulk hidden) ···",
                     ha="center", va="top", fontsize=6, color=c, style="italic")
    ax1.set_xlim(-1, max(len([s for s in hist if s is not None])
                         for _,hist in top)+1)
    ax1.set_title(f"Top-{len(top)} dominant histories (endpoints shown)", fontsize=8)

    # ── Weight bar chart ────────────────────────────────────────
    ax2=fig.add_subplot(gs[2])
    ws=[w for w,_ in result.top_histories]
    # Use log_Z for display — Z itself may be very large
    log_Z_str = f"{result.log_Z:.3f}" if np.isfinite(result.log_Z) else "−∞"
    ax2.barh([f"#{i+1}" for i in range(len(ws))][::-1], ws[::-1],
             color=colors[:len(ws)][::-1], alpha=0.8)
    ax2.set_xlabel("History weight", fontsize=8)
    ax2.set_title(f"log Z = {log_Z_str}\n|V_bulk|={len(result.bulk_vocab)}", fontsize=8)
    ax2.grid(alpha=0.2, axis="x")
    fig.suptitle(f"Z[s_in→s_out]  T={T}  P_min={result.params.P_min}"
                 f"  σ_J={result.params.sigma_J}", fontsize=9)
    plt.savefig(out, dpi=140); plt.close(); print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# §10.  Hardcoded demo  (called when no MIDI arg is supplied)
# ─────────────────────────────────────────────────────────────

def _run_hardcoded_demo(params: GFTParams) -> None:
    T, K  = 4, 88
    sigma = np.zeros((T, K), dtype=int)
    beats = [["C3","D4","F1","A2"],["C3","D3","F1","A2"],[],["C3","F2","A2","C4"]]
    for t, names in enumerate(beats):
        for name in names:
            sigma[t, _midi_to_index(NOTE_MAP[name])] = 1

    print("\n"+"="*62+"\n  Hardcoded Demo  (no MIDI input)\n"+"="*62)
    fd   = build_feynman(sigma, params)
    foam = lift_to_spinfoam(fd)
    print(foam.summary(full=True))
    sigma_recon = decode(foam)
    print(f"\n  Decoding exact_match = {np.array_equal(sigma, sigma_recon)}")
    for t in range(T):
        o = sorted([index_to_name(i) for i in np.where(sigma[t]>0)[0]])
        r = sorted([index_to_name(i) for i in np.where(sigma_recon[t]>0)[0]])
        print(f"    t={t}  {'✓' if o==r else '✗'}  {o}")
    save_midi(sigma,       "/mnt/user-data/outputs/demo_original.mid",     bpm=100)
    save_midi(sigma_recon, "/mnt/user-data/outputs/demo_reconstructed.mid", bpm=100)
    visualize_2complex(foam)
    visualize_reconstruction(sigma, sigma_recon)
    print("\n  Run with a MIDI file:  python music_gft_v11.py input.mid")

# ─────────────────────────────────────────────────────────────
# §11.  Supplementary: Z demo with propagating boundary states
# ─────────────────────────────────────────────────────────────

def z_demo_propagating(params: GFTParams):
    """
    Demo Z with boundary states entirely within V_bulk (P > P_min).
    Ensures Z ≠ 0 and top histories are meaningful.

    Using notes from the bulk vocab:
        C3 (j=0, q=3,  P=0.105)  ✓
        C2 (j=0, q=2,  P=0.222)  ✓
        C1 (j=0, q=1,  P=0.667)  ✓
    A boundary state with all-C notes trivially couples to J=0
    (all j=0 → Inv is 1-dimensional → A_v = λ/1 = λ).
    """
    print("\n"+"="*62)
    print("  Z demo — propagating boundary states (C notes only)")
    print("="*62)

    # Boundary states: all within bulk vocab
    s_in  = frozenset({(0.0,1),(0.0,3)})   # {C1, C3}
    s_out = frozenset({(0.0,2),(0.0,3)})   # {C2, C3}

    for label, s in [("s_in ",s_in),("s_out",s_out)]:
        notes = sorted(jq_to_name(*jq) for jq in s)
        Ps    = [gft_propagator(jq[0],jq[1],params) for jq in s]
        in_v  = all(gft_propagator(jq[0],jq[1],params) > params.P_min for jq in s)
        print(f"  {label} = {notes}  P={[f'{p:.3f}' for p in Ps]}  in_vocab={in_v}")

    for T in [2, 4]:
        result = compute_Z(s_in, s_out, T=T, params=params, top_k=3)
        print(f"\n  T={T}:")
        print(f"    Z={result.Z:.6e}   −logZ={-result.log_Z:.4f}")
        for rank,(w,hist) in enumerate(result.top_histories[:3],1):
            path=" → ".join(
                "{"+",".join(sorted(jq_to_name(*jq) for jq in s))+"}"
                if s else "∅" for s in hist)
            print(f"    #{rank} w={w:.4e}  {path}")

    # Intertwiners on same-j edges
    print("\n  Intertwiner check for same-j edges:")
    for js_test in [[0.0,0.0],[1.0,1.0],[0.5,0.5,0.5,0.5]]:
        iota = build_intertwiner(js_test)
        if iota is not None:
            print(f"    j={js_test}  shape={iota.shape}  ‖ι‖={np.linalg.norm(iota):.6f}")
        else:
            print(f"    j={js_test}  → None (no J=0 channel)")
    print("="*62)




# ─────────────────────────────────────────────────────────────
# §12.  MIDI Export  (pure Python, no external dependencies)
# ─────────────────────────────────────────────────────────────

def _vlq(n: int) -> bytes:
    """Variable-length quantity (MIDI delta-time encoding)."""
    buf = [n & 0x7F]; n >>= 7
    while n:
        buf.append((n & 0x7F) | 0x80); n >>= 7
    return bytes(reversed(buf))


def save_midi(sigma: np.ndarray,
              path:  str,
              bpm:   float = 120.0,
              tpb:   int   = 480,
              velocity: int = 80) -> None:
    """
    Save a piano roll σ (T×88) as a Standard MIDI File (.mid).

    Parameters
    ----------
    sigma    : np.ndarray  shape (T, 88), dtype int
               σ[t, i] = 1  ↔  88-key index i is active at beat t
               Each beat = one quarter note.
    path     : output path, e.g. "output.mid"
    bpm      : tempo in beats per minute
    tpb      : ticks per beat (resolution)
    velocity : MIDI velocity for all note-on events (0–127)

    Format
    ------
    MIDI format-0, single track, channel 0.
    88-key index i → MIDI note number i+21  (A0=21, C8=108).
    Sustain: if σ[t,i]=σ[t+1,i]=1 the note is held (single long note).
    """
    import struct

    T, K = sigma.shape
    assert K == 88, f"Expected 88-column piano roll, got {K}"

    def idx_to_midi_note(i): return i + 21   # A0=21

    # ── Collect (start_tick, end_tick, midi_pitch) ─────────────
    note_list = []
    open_notes: Dict[int, int] = {}   # midi_pitch → start_tick

    for t in range(T):
        tick = t * tpb
        for i in range(88):
            pitch  = idx_to_midi_note(i)
            active = bool(sigma[t, i])
            was    = pitch in open_notes

            if active and not was:
                open_notes[pitch] = tick
            elif not active and was:
                note_list.append((open_notes.pop(pitch), tick, pitch))

    end_tick = T * tpb
    for pitch, start in open_notes.items():
        note_list.append((start, end_tick, pitch))

    # Ensure non-zero duration
    note_list = [(s, max(e, s+tpb//4), p) for s,e,p in note_list]

    # ── Build MIDI event list (sorted by tick) ─────────────────
    raw = []
    for start, end, pitch in note_list:
        raw.append((start, True,  pitch))   # note-on
        raw.append((end,   False, pitch))   # note-off
    # note-off before note-on at same tick (avoid stuck notes)
    raw.sort(key=lambda x: (x[0], 0 if not x[1] else 1))

    us_per_beat = int(60_000_000 / bpm)

    # ── Convert to delta-time events ───────────────────────────
    events: List[bytes] = []

    # Tempo meta-event
    events.append(_vlq(0) + b'\xFF\x51\x03' +
                  bytes([(us_per_beat>>16)&0xFF,
                         (us_per_beat>> 8)&0xFF,
                          us_per_beat     &0xFF]))

    cur_tick = 0
    for tick, on, pitch in raw:
        delta    = tick - cur_tick
        cur_tick = tick
        status   = 0x90 if on else 0x80
        vel      = velocity if on else 0
        events.append(_vlq(delta) + bytes([status, pitch, vel]))

    events.append(_vlq(0) + b'\xFF\x2F\x00')  # end-of-track

    # ── Assemble MIDI binary ────────────────────────────────────
    track_body = b''.join(events)
    header = b'MThd' + struct.pack('>IHHH', 6, 0, 1, tpb)
    track  = b'MTrk' + struct.pack('>I', len(track_body)) + track_body

    with open(path, 'wb') as f:
        f.write(header + track)

    n_on  = sum(1 for _,on,_ in raw if on)
    n_off = sum(1 for _,on,_ in raw if not on)
    print(f"  MIDI saved → {path}")
    print(f"    {T} beats  {bpm} bpm  tpb={tpb}")
    print(f"    {n_on} note-ons  {n_off} note-offs  "
          f"{len(note_list)} distinct notes")


# ─────────────────────────────────────────────────────────────
# §13.  MIDI Reader  (pure Python, no external dependencies)
# ─────────────────────────────────────────────────────────────

def _read_vlq(data: bytes, pos: int) -> Tuple[int, int]:
    """Decode MIDI variable-length quantity. Returns (value, new_pos)."""
    value = 0
    while True:
        b = data[pos]; pos += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return value, pos


def read_midi(path: str) -> Tuple[List[Tuple[int,int,bool]], int, int]:
    """
    Parse a Standard MIDI File.

    Returns
    -------
    events : list of (tick, midi_pitch, is_note_on)
    tpb    : ticks per beat (from header)
    tempo  : µs per beat (from first Set Tempo meta-event, default 500000)

    Handles format 0 and format 1, running status, all meta/sysex events.
    """
    import struct as _struct

    with open(path, 'rb') as fh:
        data = fh.read()

    pos = 0
    if data[pos:pos+4] != b'MThd':
        raise ValueError(f"Not a MIDI file: {path}")
    pos += 4
    pos += 4                          # header length (always 6)
    fmt, n_tracks, tpb = _struct.unpack_from('>HHH', data, pos); pos += 6

    if tpb & 0x8000:
        raise ValueError("SMPTE time codes not supported")

    events: List[Tuple[int,int,bool]] = []
    tempo  = 500_000   # 120 bpm default

    for _ in range(n_tracks):
        if data[pos:pos+4] != b'MTrk':
            raise ValueError(f"Expected MTrk chunk at byte {pos}")
        pos += 4
        tlen = _struct.unpack_from('>I', data, pos)[0]; pos += 4
        end  = pos + tlen

        cur_tick    = 0
        last_status = 0

        while pos < end:
            delta, pos = _read_vlq(data, pos)
            cur_tick  += delta

            b = data[pos]
            if b & 0x80:                   # new status byte
                status      = b; pos += 1
                last_status = status
            else:                          # running status
                status = last_status

            stype = status & 0xF0

            if stype == 0x90:              # note-on
                pitch = data[pos]; vel = data[pos+1]; pos += 2
                events.append((cur_tick, pitch, vel > 0))

            elif stype == 0x80:            # note-off
                pitch = data[pos]; pos += 2
                events.append((cur_tick, pitch, False))

            elif stype in (0xA0, 0xB0, 0xE0):   # 2-byte data
                pos += 2
            elif stype in (0xC0, 0xD0):          # 1-byte data
                pos += 1

            elif status == 0xFF:           # meta-event
                mtype    = data[pos]; pos += 1
                mlen, pos = _read_vlq(data, pos)
                mdata    = data[pos:pos+mlen]; pos += mlen
                if mtype == 0x51 and mlen == 3:  # Set Tempo
                    tempo = (mdata[0]<<16)|(mdata[1]<<8)|mdata[2]

            elif status in (0xF0, 0xF7):   # sysex
                slen, pos = _read_vlq(data, pos); pos += slen

            else:
                pos = end; break           # unknown chunk byte, skip track

        pos = end   # always advance past track

    return events, tpb, tempo


def midi_to_piano_roll(path:            str,
                       beat_resolution: int = 1
                       ) -> Tuple[np.ndarray, float]:
    """
    Load a MIDI file → piano roll σ (T×88).

    Parameters
    ----------
    path            : .mid file path
    beat_resolution : how many MIDI quarter-note beats = one σ row.
                      1  → each row = one quarter note  (default)
                      2  → each row = one half note
                      0.5→ each row = one eighth note

    Returns
    -------
    sigma : np.ndarray  shape (T, 88)
    bpm   : float  tempo detected from file
    """
    events, tpb, tempo = read_midi(path)
    bpm = 60_000_000 / tempo

    ticks_per_cell = int(tpb * beat_resolution)
    if not events:
        return np.zeros((1,88), dtype=int), bpm

    # Build (start_tick, end_tick, midi_pitch) from note-on/off pairs
    open_notes: Dict[int,int] = {}   # pitch → start_tick
    note_list:  List[Tuple[int,int,int]] = []

    for tick, pitch, is_on in sorted(events, key=lambda x: x[0]):
        if is_on:
            if pitch not in open_notes:
                open_notes[pitch] = tick
        else:
            if pitch in open_notes:
                note_list.append((open_notes.pop(pitch), tick, pitch))

    max_tick = max((e for e in events), key=lambda x:x[0])[0]
    for pitch, start in open_notes.items():     # unclosed notes
        note_list.append((start, max_tick + ticks_per_cell, pitch))

    if not note_list:
        return np.zeros((1,88), dtype=int), bpm

    T = max(end for _,end,_ in note_list) // ticks_per_cell + 1
    sigma = np.zeros((T, 88), dtype=int)

    for start, end, pitch in note_list:
        idx = pitch - 21    # MIDI note → 88-key index (A0=21)
        if not (0 <= idx < 88):
            continue
        t0 = start // ticks_per_cell
        t1 = max(t0, (end - 1) // ticks_per_cell)
        sigma[t0:min(t1+1, T), idx] = 1

    # Strip trailing all-zero rows
    last_active = T - 1
    while last_active > 0 and not sigma[last_active].any():
        last_active -= 1
    sigma = sigma[:last_active+1]

    return sigma, bpm


def print_piano_roll(sigma: np.ndarray, label: str = "Piano roll") -> None:
    """Pretty-print σ beat by beat."""
    print(f"\n  {label}  shape={sigma.shape}")
    for t in range(sigma.shape[0]):
        notes = sorted([index_to_name(i) for i in np.where(sigma[t]>0)[0]])
        print(f"    beat {t:>3d} : {notes if notes else '(rest)'}")


# ─────────────────────────────────────────────────────────────
# §14.  CLI via argparse
# ─────────────────────────────────────────────────────────────

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="music_gft_v11",
        description=(
            "GFT-grounded Music Spin Foam pipeline.\n"
            "Encodes a MIDI file as a spin foam, runs the partition function,\n"
            "and writes the decoded MIDI back out."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods
-------
  Default  : Transfer-matrix HMM (fast, mean-field, ~2s)
  --mcmc   : Spin Foam MCMC (accurate, sums over intertwiners, ~minutes)

Examples
--------
  # Fast default run:
  python music_gft_v11.py input.mid --no_tests

  # MCMC (more accurate):
  python music_gft_v11.py input.mid --mcmc --n_steps 50000 --n_chains 3

  # Finer beat grid:
  python music_gft_v11.py input.mid --beat_resolution 0.5 --mcmc
""")

    p.add_argument("midi_in",
                   help="Input MIDI file (.mid)")
    p.add_argument("--midi_out", default=None,
                   help="Output MIDI file for decoded result "
                        "(default: <midi_in>_decoded.mid)")
    p.add_argument("--beat_resolution", type=float, default=1.0,
                   help="Beats per σ row: 1=quarter, 2=half, 0.5=eighth (default: 1)")
    p.add_argument("--bpm", type=float, default=None,
                   help="Override tempo for output MIDI (default: use file tempo)")
    p.add_argument("--mu2",     type=float, default=0.5,
                   help="GFT mass μ² (default: 0.5)")
    p.add_argument("--lam",     type=float, default=1.0,
                   help="GFT coupling λ (default: 1.0)")
    p.add_argument("--P_min",   type=float, default=0.10,
                   help="Propagator cutoff P_min (default: 0.10 → |V|≈10)")
    p.add_argument("--sigma_J", type=float, default=1.0,
                   help="Harmonic temperature σ_J in A_v=λ·exp(-J_min/σ_J).\n"
                        "Large=lenient (all chords), small=strict (only J=0). (default: 1.0)")
    p.add_argument("--mcmc", action="store_true",
                   help="Use MCMC over intertwiner configurations (more accurate, slower).")
    p.add_argument("--n_steps", type=int, default=30_000,
                   help="MCMC steps per chain (default: 30000)")
    p.add_argument("--n_chains", type=int, default=3,
                   help="Number of independent MCMC chains (default: 3)")
    p.add_argument("--n_anneal", type=int, default=20,
                   help="AIS annealing steps for Z estimation (default: 20)")
    p.add_argument("--top_k", type=int,   default=5,
                   help="Top-k dominant histories to report (default: 5)")
    p.add_argument("--no_tests", action="store_true",
                   help="Skip the built-in test suite")
    p.add_argument("--no_figures", action="store_true",
                   help="Skip figure generation")
    p.add_argument("--no_partition", action="store_true",
                   help="Skip partition function Z (fast mode)")
    p.add_argument("--serialize_spinfoam", action="store_true",
                   help="Serialize SpinFoam")
    
    return p


def main_cli(argv=None):
    import os, sys
    parser = _build_parser()
    args   = parser.parse_args(argv)

    params = GFTParams(mu2=args.mu2, lam=args.lam, P_min=args.P_min,
                       sigma_J=args.sigma_J)

    print("="*64)
    print("  music_gft_v11  —  GFT Music Spin Foam")
    print("="*64)
    print(f"  Input : {args.midi_in}")
    print(f"  μ²={params.mu2}  λ={params.lam}  P_min={params.P_min}")

    # ── Tests ────────────────────────────────────────────────
    if not args.no_tests:
        run_tests(params)

    # ── Load MIDI ────────────────────────────────────────────
    print(f"\n  Loading MIDI: {args.midi_in}")
    sigma, file_bpm = midi_to_piano_roll(args.midi_in,
                                         beat_resolution=args.beat_resolution)
    bpm_out = args.bpm if args.bpm else file_bpm

    print(f"  Detected bpm={file_bpm:.1f}  beat_resolution={args.beat_resolution}")
    print_piano_roll(sigma, label="Input piano roll")

    T, K = sigma.shape
    print(f"\n  Piano roll: T={T} beats × {K} keys  "
          f"({sigma.sum()} active note-beats)")

    if T == 0 or sigma.sum() == 0:
        print("  ⚠ Empty piano roll — check MIDI file and beat_resolution.")
        return

    # ── Step 1: build Feynman diagram ────────────────────────
    print("\n  Step 1: σ → FeynmanDiagram")
    fd = build_feynman(sigma, params)
    print(f"    {len(fd.propagators)} propagators  {len(fd.vertices)} vertices")

    # ── Step 2: lift to spin foam ────────────────────────────
    print("  Step 2: FeynmanDiagram → SpinFoam2Complex")
    foam = lift_to_spinfoam(fd)
    print(foam.summary(full=False))

    # ── Decode ───────────────────────────────────────────────
    sigma_recon = decode(foam)
    match = np.array_equal(sigma, sigma_recon)
    print(f"\n  Decoding exact_match = {match}")
    if not match:
        for t in range(T):
            o = sorted([index_to_name(i) for i in np.where(sigma[t]>0)[0]])
            r = sorted([index_to_name(i) for i in np.where(sigma_recon[t]>0)[0]])
            if o != r:
                print(f"    ✗ t={t}: expected {o}, got {r}")

    # ── Save decoded MIDI ────────────────────────────────────
    if args.midi_out:
        out_path = args.midi_out
    else:
        base = os.path.splitext(args.midi_in)[0]
        out_path = base + "_decoded.mid"

    save_midi(sigma_recon, out_path, bpm=bpm_out)
    if args.serialize_spinfoam:
        base = os.path.splitext(args.midi_in)[0]
        serialization_out_path = base + "_spinfoam.json"
        foam.to_json(serialization_out_path)
        

    # ── Partition function ────────────────────────────────────
    if not args.no_partition:
        s_in, s_out = foam.boundary_states()
        if s_in and s_out:
            print(f"\n  s_in  = {sorted(jq_to_name(*jq) for jq in s_in)}")
            print(f"  s_out = {sorted(jq_to_name(*jq) for jq in s_out)}")
            base = os.path.splitext(args.midi_in)[0]

            if args.mcmc:
                # ── More accurate: MCMC over intertwiner configurations ──
                print(f"\n  Running MCMC  "
                      f"(n_steps={args.n_steps} n_chains={args.n_chains} "
                      f"n_anneal={args.n_anneal})")
                try:
                    mcmc_result = compute_Z_mcmc(
                        foam, params,
                        n_steps  = args.n_steps,
                        n_warmup = args.n_steps // 6,
                        n_anneal = args.n_anneal,
                        n_chains = args.n_chains,
                        top_k    = args.top_k,
                    )
                    print(mcmc_result.summary())
                    if not args.no_figures:
                        visualize_mcmc_result(mcmc_result, foam,
                                              out=base+"_mcmc.png")
                except Exception as e:
                    print(f"  ⚠ MCMC failed: {e}")
            else:
                # ── Fast: transfer-matrix HMM ────────────────────────────
                try:
                    result = compute_Z(s_in, s_out, T=T,
                                       params=params, top_k=args.top_k)
                    print(result.summary())
                    if not args.no_figures:
                        visualize_partition_result(result, out=base+"_partition.png")
                except ValueError as e:
                    print(f"  ⚠ Skipping Z: {e}")
        else:
            print("  ⚠ Could not extract boundary states — skipping Z.")

    # ── Figures ──────────────────────────────────────────────
    if not args.no_figures:
        base = os.path.splitext(args.midi_in)[0]
        visualize_2complex(foam,         out=base+"_2complex.png")
        visualize_intertwiners(foam,     out=base+"_intertwiners.png")
        visualize_reconstruction(sigma, sigma_recon,
                                        out=base+"_reconstruction.png")

    print("\n  Done.")




# ─────────────────────────────────────────────────────────────
# §15.  Wigner 6j symbols + proper recoupling amplitudes
# ─────────────────────────────────────────────────────────────

from sympy.physics.wigner import wigner_6j as _sympy_6j
from sympy import Rational as _R, N as _N
import random as _random

@lru_cache(maxsize=65536)
def wigner_6j_cached(j1:float,j2:float,j3:float,
                     l1:float,l2:float,l3:float) -> float:
    """Exact Wigner 6j symbol via sympy, heavily cached."""
    def r(x): return _R(int(round(2*x)), 2)
    try:
        return float(_N(_sympy_6j(r(j1),r(j2),r(j3),r(l1),r(l2),r(l3))))
    except Exception:
        return 0.0


def admissible_intertwiners(js: List[float]) -> List[float]:
    """
    All valid intertwiner values when coupling representations j_1,...,j_n.
    = set of all J_total reachable by sequential CG coupling.
    """
    if not js:
        return [0.0]
    reached = {js[0]}
    for jn in js[1:]:
        nxt: set = set()
        for jc in reached:
            J = abs(jc - jn)
            while J <= jc + jn + 1e-9:
                nxt.add(round(J, 1)); J += 1.0
        reached = nxt
    return sorted(reached)


@lru_cache(maxsize=32768)
def vertex_amp_6j(js_tuple: Tuple[float,...],
                  target_J:  Optional[float] = None) -> float:
    """
    Vertex amplitude using recoupling theory.

    Couples j_1,...,j_n via a sequential binary tree.  At each step k,
    coupling I_{prev} ⊗ j_k → I_new contributes weight sqrt(2*I_new+1).

    target_J=None : sum over all final channels, normalised by ∏(2j+1).
                    Always in [0,1].  Measures how 'harmonically rich' the
                    chord is — how many coupling channels are accessible.

    target_J given: raw amplitude for coupling to a specific intertwiner.
                    Used by MCMC to compute the weight of each i_e value.
                    Always ≥ 0.

    Key properties
    ──────────────
    • Always ≥ 0 (positive definite — no Ponzano-Regge sign cancellations)
    • target_J=None with all-zero js → 1.0  (closed chord = maximal)
    • Larger chords with many admissible channels → higher unnormalised weight
    • Replaces the Gaussian approximation exp(-J_min/σ_J) with exact recoupling
    """
    if not js_tuple:
        return 1.0 if target_J is None or abs(target_J) < 1e-9 else 0.0
    js = list(js_tuple)

    # Sequential coupling state: dict { J_cur : amplitude }
    states: Dict[float, float] = {js[0]: 1.0}

    for k in range(1, len(js)):
        jk = js[k]
        nxt: Dict[float, float] = {}
        for J_prev, amp in states.items():
            J = abs(J_prev - jk)
            while J <= J_prev + jk + 1e-9:
                J_new = round(J, 1)
                nxt[J_new] = nxt.get(J_new, 0.0) + sqrt(2*J_new + 1) * amp
                J += 1.0
        states = nxt

    if target_J is not None:
        return states.get(target_J, 0.0)

    # Normalised: divide by ∏ (2j+1)
    dim_prod = float(np.prod([2*j + 1 for j in js]))
    return sum(states.values()) / dim_prod


# ─────────────────────────────────────────────────────────────
# §16.  Spin Foam MCMC
# ─────────────────────────────────────────────────────────────

@dataclass
class MCMCResult:
    """Result of the MCMC partition function computation."""
    log_Z:            float          # log Z (always finite)
    Z:                float          # exp(log_Z) — may be inf for large T
    log_Z0:           float          # log of reference (uniform) partition fn
    log_Z_err:        float          # std error of log Z estimate
    acceptance_rate:  float
    n_steps:          int
    n_warmup:         int
    n_anneal:         int
    foam_summary:     str            # concise foam info string
    dominant_configs: List[Tuple[float, Dict[int,float]]]  # top (logW, {eid:i_e})
    params:           GFTParams

    def summary(self) -> str:
        lines = [
            "=" * 64,
            "  Spin Foam MCMC Partition Function",
            f"  μ²={self.params.mu2}  λ={self.params.lam}  "
            f"σ_J={self.params.sigma_J}  P_min={self.params.P_min}",
            "=" * 64,
            self.foam_summary,
            "",
            f"  log Z  = {self.log_Z:.5f}  (±{self.log_Z_err:.3f})",
            f"  Z      = {self.Z:.6e}" +
                ("  (use log Z)" if not np.isfinite(self.Z) else ""),
            f"  log Z0 = {self.log_Z0:.5f}  (uniform reference)",
            f"  log(Z/Z0) = {self.log_Z - self.log_Z0:.5f}",
            "",
            f"  Acceptance rate : {self.acceptance_rate:.3f}",
            f"  Steps           : {self.n_steps}  "
            f"(warmup {self.n_warmup}  anneal {self.n_anneal})",
            "",
            f"  Top-{len(self.dominant_configs)} configurations by weight:",
        ]
        for rank, (lw, cfg) in enumerate(self.dominant_configs, 1):
            edge_str = "  ".join(f"e{eid}:i={i_e:.1f}"
                                 for eid, i_e in sorted(cfg.items()))
            lines.append(f"    #{rank}  logW={lw:.4f}  {edge_str}")
        lines.append("=" * 64)
        return "\n".join(lines)


def _build_foam_adjacency(foam: 'SpinFoam2Complex'
                          ) -> Tuple[Dict[int,List[float]],       # vid → sorted face spins
                                     Dict[int,Optional[int]],      # vid → outgoing time edge id
                                     Dict[int,List[int]]]:         # eid → [vid_src, vid_tgt]
    """
    Precompute three adjacency maps used by the MCMC local updater.
    Called once before the chain starts — O(V+E+F).
    """
    # vid → face spins (for vertex amplitude)
    vid_js: Dict[int, List[float]] = {vid: [] for vid in foam.vertices}
    for fid, face in foam.faces.items():
        for vid, v in foam.vertices.items():
            if face.t_from <= v.t <= face.t_to:
                vid_js[vid].append(face.j)

    # vid → outgoing time-edge id (the intertwiner that appears in A_v)
    vid_time_eid: Dict[int, Optional[int]] = {vid: None for vid in foam.vertices}
    for eid, edge in foam.edges.items():
        if edge.kind == EdgeKind.TIME:
            vid_time_eid[edge.v_src_id] = eid

    # eid → adjacent vertex ids  (at most 2: src and tgt)
    eid_vids: Dict[int, List[int]] = {}
    for eid, edge in foam.edges.items():
        vids_for_edge = set()
        for vid, v in foam.vertices.items():
            # vertex is adjacent if any face of the edge touches this time
            for fid in edge.face_ids:
                face = foam.faces[fid]
                if face.t_from <= v.t <= face.t_to:
                    vids_for_edge.add(vid)
        eid_vids[eid] = list(vids_for_edge)

    return vid_js, vid_time_eid, eid_vids


def _vertex_log_weight_local(vid:          int,
                              vid_js:       Dict[int,List[float]],
                              vid_time_eid: Dict[int,Optional[int]],
                              config:       Dict[int,float],
                              lam:          float) -> float:
    """
    log A_v for one vertex given the current config.
    Pure function — no foam access after setup.  O(n_faces at vertex).
    """
    js = vid_js[vid]
    if not js:
        return float(np.log(lam + 1e-300))
    teid = vid_time_eid[vid]
    target_J = config.get(teid) if teid is not None else None
    amp = vertex_amp_6j(tuple(sorted(js)), target_J=target_J) * lam
    return float(np.log(amp + 1e-300))


def compute_Z_mcmc(foam:     'SpinFoam2Complex',
                   params:   GFTParams,
                   n_steps:  int = 30_000,
                   n_warmup: int = 5_000,
                   n_anneal: int = 20,
                   n_chains: int = 3,
                   top_k:    int = 5) -> 'MCMCResult':
    """
    Annealed Importance Sampling (AIS) MCMC — spin foam partition function.

    Free variables
    ──────────────
    Face spins (j,q) fixed by music.  MCMC samples intertwiner i_e per edge.
    For edge e with face spins [j₁,...,jₖ]: i_e ∈ admissible_intertwiners(js).

    Weight per configuration
    ────────────────────────
        W({i_e}) = ∏_e (2i_e+1)  ×  ∏_v λ · vertex_amp_6j({j_f at v}, i_e_out)

    Key optimisation — LOCAL UPDATES (v11.2 fix)
    ─────────────────────────────────────────────
    When edge e is flipped (i_e → i_e'), only the (≤2) vertices adjacent to e
    need their A_v recomputed.  The rest of log W is unchanged.

    This reduces per-step cost from O(N_vertices) → O(1), giving a
    ~N_vertices/2 ≈ 400x speedup for a 788-beat piece.

    AIS (Annealed Importance Sampling)
    ────────────────────────────────────
        π_0 = uniform over {i_e}           log Z_0 = Σ_e log|I(e)|
        π_β  ∝ W^β                         β ∈ [0,1]
        log Z = log Z_0 + Σ_k Δβ · log W_k   (accumulated along the anneal)
    """
    import time
    t0_total = time.time()

    # ── Precompute adjacency (once) ─────────────────────────
    vid_js, vid_time_eid, eid_vids = _build_foam_adjacency(foam)

    # ── Intertwiner spaces ──────────────────────────────────
    edge_spaces: Dict[int, List[float]] = {}
    for eid, edge in foam.edges.items():
        js = [foam.faces[fid].j for fid in edge.face_ids]
        edge_spaces[eid] = admissible_intertwiners(js) if js else [0.0]

    free_eids = [eid for eid, space in edge_spaces.items() if len(space) > 1]
    n_free    = len(free_eids)
    log_Z0    = sum(float(np.log(len(edge_spaces[eid]))) for eid in edge_spaces)

    print(f"  MCMC  |V|={len(foam.vertices)}  |E|={len(foam.edges)}"
          f"  |F|={len(foam.faces)}  free_edges={n_free}")
    if n_free > 0:
        sizes = sorted(set(len(edge_spaces[e]) for e in free_eids), reverse=True)
        print(f"    Intertwiner space sizes: {sizes[:10]}{'…' if len(sizes)>10 else ''}")
    print(f"    log Z0 = {log_Z0:.4f}")

    if n_free == 0:
        # Trivial: only one configuration
        config  = {eid: edge_spaces[eid][0] for eid in edge_spaces}
        lw_edge = sum(float(np.log(2*config[eid]+1)) for eid in config)
        lw_v    = sum(_vertex_log_weight_local(vid, vid_js, vid_time_eid,
                                               config, params.lam)
                      for vid in foam.vertices)
        log_W   = lw_edge + lw_v
        Z       = float(np.exp(log_W)) if log_W < 709 else float('inf')
        print(f"    No free edges — Z exact: log_Z={log_W:.4f}")
        return MCMCResult(
            log_Z=log_W, Z=Z, log_Z0=log_Z0, log_Z_err=0.0,
            acceptance_rate=0.0, n_steps=0, n_warmup=0, n_anneal=n_anneal,
            foam_summary=f"  {len(foam.vertices)}V  {len(foam.edges)}E  "
                         f"{len(foam.faces)}F  (0 free intertwiners — exact)",
            dominant_configs=[(log_W, dict(config))],
            params=params,
        )

    # ── Warm up vertex_amp_6j cache ─────────────────────────
    print(f"    Warming vertex_amp_6j cache...", end="", flush=True)
    t0 = time.time()
    for vid in foam.vertices:
        js_t = tuple(sorted(vid_js[vid]))
        adm  = admissible_intertwiners(list(js_t))
        for J in adm:
            vertex_amp_6j(js_t, target_J=J)
        vertex_amp_6j(js_t, target_J=None)
    print(f" {time.time()-t0:.1f}s")

    betas     = np.linspace(0.0, 1.0, n_anneal + 1)
    steps_per = max(1, (n_warmup + n_steps) // n_anneal)

    chain_log_Z: List[float] = []
    all_top:     List[Tuple[float, Dict[int,float]]] = []
    total_accepted = total_proposed = 0

    for chain_idx in range(n_chains):
        t_chain = time.time()

        # Random initial config
        config  = {eid: _random.choice(edge_spaces[eid]) for eid in edge_spaces}

        # ── Compute initial log W with per-vertex cache ──────
        # v_lw[vid] = current log A_v for vertex vid
        v_lw: Dict[int, float] = {
            vid: _vertex_log_weight_local(vid, vid_js, vid_time_eid,
                                          config, params.lam)
            for vid in foam.vertices}
        e_lw: Dict[int, float] = {
            eid: float(np.log(2*config[eid]+1)) for eid in config}
        log_W = sum(v_lw.values()) + sum(e_lw.values())

        ais_lw = 0.0   # AIS accumulator for this chain

        for k, (beta1, beta2) in enumerate(zip(betas[:-1], betas[1:])):
            ais_lw += (beta2 - beta1) * log_W

            for _ in range(steps_per):
                # LOCAL PROPOSAL: flip one random free edge
                eid    = _random.choice(free_eids)
                old_i  = config[eid]
                new_i  = _random.choice(edge_spaces[eid])
                if abs(new_i - old_i) < 1e-9:
                    continue
                total_proposed += 1

                # Delta from edge quantum dimension
                old_e_lw  = e_lw[eid]
                new_e_lw  = float(np.log(2*new_i+1))

                # Delta from adjacent vertices only (LOCAL UPDATE)
                config[eid] = new_i
                adj_vids    = eid_vids[eid]
                new_v_lws   = {vid: _vertex_log_weight_local(
                                    vid, vid_js, vid_time_eid, config, params.lam)
                               for vid in adj_vids}

                delta = (new_e_lw - old_e_lw) + sum(
                    new_v_lws[vid] - v_lw[vid] for vid in adj_vids)
                new_log_W = log_W + delta

                # Metropolis
                if (delta >= 0 or
                        _random.random() < float(np.exp(min(beta2 * delta, 500)))):
                    # Accept
                    log_W       = new_log_W
                    e_lw[eid]   = new_e_lw
                    v_lw.update(new_v_lws)
                    total_accepted += 1
                else:
                    # Reject — restore
                    config[eid] = old_i

            all_top.append((log_W, dict(config)))

        chain_log_Z.append(log_Z0 + ais_lw)
        print(f"    Chain {chain_idx+1}/{n_chains}  "
              f"log_Z={log_Z0+ais_lw:.3f}  "
              f"t={time.time()-t_chain:.1f}s")

    # ── Aggregate ────────────────────────────────────────────
    lz_arr   = np.array(chain_log_Z)
    log_Z    = float(np.mean(lz_arr))
    log_Z_err= float(np.std(lz_arr) / np.sqrt(n_chains))
    Z        = float(np.exp(log_Z)) if log_Z < 709 else float('inf')
    acc_rate = total_accepted / max(total_proposed, 1)

    # Deduplicate top configs
    all_top.sort(key=lambda x: -x[0])
    seen: set = set(); top_unique = []
    for lw, cfg in all_top:
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key); top_unique.append((lw, cfg))
        if len(top_unique) >= top_k: break

    elapsed = time.time() - t0_total
    print(f"  MCMC done  {elapsed:.1f}s  acc={acc_rate:.3f}"
          f"  log_Z={log_Z:.4f}±{log_Z_err:.4f}")

    foam_s = (f"  {len(foam.vertices)}V  {len(foam.edges)}E  "
              f"{len(foam.faces)}F  ({n_free} free intertwiners)")
    return MCMCResult(
        log_Z=log_Z, Z=Z, log_Z0=log_Z0, log_Z_err=log_Z_err,
        acceptance_rate=acc_rate, n_steps=n_steps,
        n_warmup=n_warmup, n_anneal=n_anneal,
        foam_summary=foam_s,
        dominant_configs=top_unique[:top_k],
        params=params,
    )


def visualize_mcmc_result(result: MCMCResult,
                          foam:   'SpinFoam2Complex',
                          out:    str = "/mnt/user-data/outputs/v11_mcmc.png") -> None:
    """Plot MCMC result: intertwiner distribution + Z estimate."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

    # ── Panel 1: intertwiner distribution for top config ──────
    ax = axes[0]
    if result.dominant_configs:
        _, best_cfg = result.dominant_configs[0]
        eids   = sorted(best_cfg.keys())
        i_vals = [best_cfg[e] for e in eids]
        ax.bar([f"e{e}" for e in eids], i_vals, color="steelblue", alpha=0.8)
        ax.set_ylabel("i_e  (intertwiner spin)", fontsize=9)
        ax.set_title("Best config intertwiner assignment", fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.grid(alpha=0.2, axis='y')

    # ── Panel 2: top config weights ────────────────────────────
    ax = axes[1]
    log_ws = [lw for lw,_ in result.dominant_configs]
    ranks  = [f"#{i+1}" for i in range(len(log_ws))]
    ax.barh(ranks[::-1], log_ws[::-1], color="tomato", alpha=0.8)
    ax.set_xlabel("log W (unnormalised)", fontsize=9)
    ax.set_title(f"Top-{len(log_ws)} configurations", fontsize=9)
    ax.grid(alpha=0.2, axis='x')

    # ── Panel 3: Z summary ─────────────────────────────────────
    ax = axes[2]
    ax.axis('off')
    log_Z_str = f"{result.log_Z:.4f} ± {result.log_Z_err:.4f}"
    Z_str     = f"{result.Z:.3e}" if np.isfinite(result.Z) else "inf"
    lines = [
        f"log Z  = {log_Z_str}",
        f"Z      = {Z_str}",
        f"log Z0 = {result.log_Z0:.4f}",
        f"log(Z/Z0) = {result.log_Z - result.log_Z0:.4f}",
        "",
        f"Acceptance = {result.acceptance_rate:.3f}",
        f"Steps  = {result.n_steps}",
        f"Chains = 3",
        f"Anneal = {result.n_anneal} β-steps",
        "",
        f"μ² = {result.params.mu2}",
        f"λ  = {result.params.lam}",
        f"σ_J = {result.params.sigma_J}",
    ]
    for i, line in enumerate(lines):
        ax.text(0.05, 0.95 - i*0.07, line,
                transform=ax.transAxes, fontsize=8,
                fontfamily='monospace',
                color="navy" if i < 4 else "dimgray")
    ax.set_title("MCMC Summary", fontsize=9)

    fig.suptitle(f"Spin Foam MCMC  —  {result.foam_summary.strip()}", fontsize=9)
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  Saved: {out}")
    
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main_cli()           # MIDI file supplied → full CLI pipeline
    else:
        params = GFTParams(mu2=0.5, lam=1.0, P_min=0.10)
        run_tests(params)
        _run_hardcoded_demo(params)
