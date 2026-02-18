"""
compilation/base.py
Abstract interface that every compilation scheme must implement.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from music.midi.midi_entity import MIDIEntity
from core.compilation.components.spinfoam import SpinfoamComplex


class CompilationScheme(ABC):
    """
    Unified interface for all schemes in the classification space S × T.

    Subclasses must implement:
        compile_to_spinfoam(midi: MIDIEntity) -> SpinfoamComplex
        scheme_id                              (str property)
        description                            (str property)
    """

    @property
    @abstractmethod
    def scheme_id(self) -> str:
        """Unique identifier, e.g. 'iso_direct_indep_topo_vertex'"""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the scheme."""

    @abstractmethod
    def compile_to_spinfoam(self, midi: MIDIEntity) -> SpinfoamComplex:
        """
        Encode a MIDIEntity into a SpinfoamComplex according to this scheme.

        Args:
            midi: a MIDIEntity loaded from file or constructed programmatically

        Returns:
            SpinfoamComplex with faces, edges, vertices, and time structure
            populated according to the scheme's S-dimension and T-dimension rules.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scheme_id={self.scheme_id!r})"