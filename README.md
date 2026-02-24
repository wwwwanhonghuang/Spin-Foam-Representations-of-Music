

# Spin Foam Representations of Music

This repository implements a background-independent framework for encoding MIDI as spin foam in 2-complexes by various appraoches. By mapping musical events to ontological components of spin foam, we treat music instance as a discrete spacetime structure. This work aims at bridging the quantum field theory and music theories.


## Quick Start

```bash
python -m dev.music_spinfoam_conversion_experiment <midi-file-path> --n_chains 3 --serialize_spinfoam
```

The resulting `.sp` file is a serialized **Spin Foam** instance.