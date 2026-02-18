

# Spin Foam Representations of Music

This repository implements a background-independent framework for encoding MIDI as spin foam in 2-complexes by various appraoches. By mapping musical events to ontological components of spin foam, we treat music instance as a discrete spacetime structure. This work aims at bridging the quantum field theory and music theories.

## 1. Formalism & Compilation

The compilation process involves mapping a 1D discrete event stream (MIDI) onto a 2D topological complex . Within the `core/compilation/schemes/` module, various algorithms define the embedding of musical "fields" into a vacuum lattice.

The primary scheme, `IsoDirectIndepTopoVertex`, implements a **Dense Lattice Discretization**. In this model, the temporal metric is recovered by a contiguous chain of vertices  and edges , where each unit represents a fundamental Planck-scale "tick" of the musical universe.

```python
from music_spinfoam.core.entities import MIDIEntity
from music_spinfoam.core.compilation.schemes import IsoDirectIndepTopoVertex

# Discretizing the MIDI source into a Relational Entity
midi = MIDIEntity.from_file('assets/first_rabbit.mid')

# Initializing the Topological Mapping Scheme
scheme = IsoDirectIndepTopoVertex()

# Compiling the Spinfoam Complex (K)
# This maps the event history to a causal 2-complex
sf = scheme.compile_to_spinfoam(midi)

print(f"Manifold localized with {len(sf.vertices)} 0-cells.")

```

## 2. Causal Reconstruction (Decoding)

Reconstructing the musical observable requires a **Causal Walk** across the 1-skeleton of the complex. Unlike standard playback, this process recovers the "time-like" evolution by observing the spin-labels  and amplitudes  associated with the faces  bounding each edge.

For the `IsoDirectIndepTopoVertex` scheme, the reconstruction is defined as an accumulation of spin-fields along the primary causal path:

```python
# Causal recovery of the observable state-stream
scheme.decode_to_midi_file(
    sf, 
    init_vertex_id=0, 
    output_path='outputs/reconstructed_rabbit.mid'
)

```

## Quick Start

```bash
python -m experiments.exp_001_iso_direct_indep_topo_vertex \
    --mid_file ./assets/first_rabbit.mid \
    --output_file_path outputs/out.sp

```

The resulting `.sp` file is a serialized **Spin Foam** instance.