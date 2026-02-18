"""
exp_03_iso_direct_indep_topo_homological.py
Experiment for scheme 03: Iso-Direct-Indep + Topo-Homological
"""

from __future__ import annotations
import os
import numpy as np

from music.midi.midi_entity import MIDIEntity
from core.compilation.components.spinfoam import SpinfoamComplex
from core.compilation.schemes.s_003_iso_direct_indep_topo_homological import IsoDirectIndepTopoHomological

SCHEME_TITLE = "Iso-Direct-Indep + Topo-Homological"
SCHEME_IDX   = 3


def run(midi_path: str = None, verbose: bool = True) -> SpinfoamComplex:
    """Run experiment 03: Iso-Direct-Indep + Topo-Homological"""

    if midi_path and os.path.exists(midi_path):
        midi = MIDIEntity.from_file(midi_path)
    else:
        pr = np.zeros((88, 4), dtype=np.float32)
        for pitch in [60, 64, 67]:
            pr[pitch - 21, :] = 1.0
        midi = MIDIEntity.from_piano_roll(pr)

    scheme = IsoDirectIndepTopoHomological()
    sf = scheme.compile_to_spinfoam(midi)
    return sf


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run(midi_path=path, verbose=True)
