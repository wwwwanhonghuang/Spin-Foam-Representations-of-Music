"""
005. Iso-Direct-Indep + Causal-SorkinCount
Classification: S = Iso-Direct-Independent   T = T.2.2 (Sorkin counting measure)
"""

from __future__ import annotations
import numpy as np
from music.midi.midi_entity import MIDIEntity
from core.compilation.components.spinfoam import SpinfoamComplex
from core.compilation.compilation_scheme import CompilationScheme


class IsoDirectIndepCausalSorkin(CompilationScheme):

    @property
    def scheme_id(self) -> str:
        return "iso_direct_indep_causal_sorkin"

    @property
    def description(self) -> str:
        return "Iso-Direct-Indep + Causal-SorkinCount"

    def compile_to_spinfoam(self, midi: MIDIEntity) -> SpinfoamComplex:
        raise NotImplementedError

    def _vertex_amplitude(self, sigma: np.ndarray, t: int, T: int) -> complex:
        raise NotImplementedError

    def _time_index(self, sigma: np.ndarray, t: int, T: int,
                    piano_roll: np.ndarray) -> float:
        raise NotImplementedError
