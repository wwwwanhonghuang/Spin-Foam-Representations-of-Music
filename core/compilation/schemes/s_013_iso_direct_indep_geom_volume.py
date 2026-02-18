"""
013. Iso-Direct-Indep + Geom-VolumeOperator
Classification: S = Iso-Direct-Independent   T = T.5.2 (Volume operator gradient)
"""

from __future__ import annotations
import numpy as np
from music.midi.midi_entity import MIDIEntity
from core.compilation.components.spinfoam import SpinfoamComplex
from core.compilation.compilation_scheme import CompilationScheme


class IsoDirectIndepGeomVolume(CompilationScheme):

    @property
    def scheme_id(self) -> str:
        return "iso_direct_indep_geom_volume"

    @property
    def description(self) -> str:
        return "Iso-Direct-Indep + Geom-VolumeOperator"

    def compile_to_spinfoam(self, midi: MIDIEntity) -> SpinfoamComplex:
        raise NotImplementedError

    def _vertex_amplitude(self, sigma: np.ndarray, t: int, T: int) -> complex:
        raise NotImplementedError

    def _time_index(self, sigma: np.ndarray, t: int, T: int,
                    piano_roll: np.ndarray) -> float:
        raise NotImplementedError
