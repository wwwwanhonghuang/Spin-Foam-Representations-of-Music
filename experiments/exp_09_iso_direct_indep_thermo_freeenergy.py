"""
exp_09_iso_direct_indep_thermo_freeenergy.py
Experiment for scheme 09: Iso-Direct-Indep + Thermo-FreeEnergy
"""

from __future__ import annotations
import os
import numpy as np

from music.midi.midi_entity import MIDIEntity
from core.compilation.components.spinfoam import SpinfoamComplex
from core.compilation.schemes.s_009_iso_direct_indep_thermo_freeenergy import IsoDirectIndepThermoFreeEnergy

SCHEME_TITLE = "Iso-Direct-Indep + Thermo-FreeEnergy"
SCHEME_IDX   = 9


def run(midi_path: str = None, verbose: bool = True) -> SpinfoamComplex:
    """Run experiment 09: Iso-Direct-Indep + Thermo-FreeEnergy"""

    if midi_path and os.path.exists(midi_path):
        midi = MIDIEntity.from_file(midi_path)
    else:
        pr = np.zeros((88, 4), dtype=np.float32)
        for pitch in [60, 64, 67]:
            pr[pitch - 21, :] = 1.0
        midi = MIDIEntity.from_piano_roll(pr)

    scheme = IsoDirectIndepThermoFreeEnergy()
    sf = scheme.compile_to_spinfoam(midi)
    return sf


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run(midi_path=path, verbose=True)
