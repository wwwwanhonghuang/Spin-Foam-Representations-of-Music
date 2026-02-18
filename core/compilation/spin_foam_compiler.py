from abc import ABC, abstractmethod
from music.midi.midi_entity import MIDIEntity

class SpinfoamCompiler(ABC):
    def __init__(self, scheme_id: str):
        self.scheme_id = scheme_id

    @abstractmethod
    def compile_to_spinfoam(self, midi: MIDIEntity):
        """
        返回一个包含顶点 (Vertices)、边 (Edges) 和 面 (Faces) 的数据结构
        """
        pass