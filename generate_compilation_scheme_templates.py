"""
generate_files.py  —  generates all 14 compilation + experiment files
"""
import os

SCHEMES = [
    (1,  "iso_direct_indep_topo_vertex",     "IsoDirectIndepTopoVertex",     "Iso-Direct-Indep + Topo-Vertex-PartialOrder",    "T.1.1", "Vertex partial order"),
    (2,  "iso_direct_indep_topo_face",       "IsoDirectIndepTopoFace",        "Iso-Direct-Indep + Topo-Face-Stratification",   "T.1.2", "Face stratification"),
    (3,  "iso_direct_indep_topo_homological","IsoDirectIndepTopoHomological", "Iso-Direct-Indep + Topo-Homological",           "T.1.3", "Homological order"),
    (4,  "iso_direct_indep_causal_pure",     "IsoDirectIndepCausalPure",      "Iso-Direct-Indep + Causal-PureOrder",           "T.2.1", "Pure causal partial order"),
    (5,  "iso_direct_indep_causal_sorkin",   "IsoDirectIndepCausalSorkin",    "Iso-Direct-Indep + Causal-SorkinCount",         "T.2.2", "Sorkin counting measure"),
    (6,  "iso_direct_indep_thermo_vonneumann","IsoDirectIndepThermoVonNeumann","Iso-Direct-Indep + Thermo-Entropy-vonNeumann", "T.3.1.1", "von Neumann entropy gradient"),
    (7,  "iso_direct_indep_thermo_shannon",  "IsoDirectIndepThermoShannon",   "Iso-Direct-Indep + Thermo-Entropy-Shannon",     "T.3.1.2", "Shannon entropy gradient"),
    (8,  "iso_direct_indep_thermo_thermal",  "IsoDirectIndepThermoThermal",   "Iso-Direct-Indep + Thermo-ThermalTime",         "T.3.2", "Thermal time KMS flow"),
    (9,  "iso_direct_indep_thermo_freeenergy","IsoDirectIndepThermoFreeEnergy","Iso-Direct-Indep + Thermo-FreeEnergy",         "T.3.3", "Free energy gradient"),
    (10, "iso_direct_indep_amp_maxpath",     "IsoDirectIndepAmpMaxPath",      "Iso-Direct-Indep + Amp-MaxPath",                "T.4.1", "Maximum amplitude path"),
    (11, "iso_direct_indep_amp_weightedwalk","IsoDirectIndepAmpWeightedWalk", "Iso-Direct-Indep + Amp-WeightedWalk",           "T.4.2", "Amplitude-weighted random walk"),
    (12, "iso_direct_indep_geom_area",       "IsoDirectIndepGeomArea",        "Iso-Direct-Indep + Geom-AreaOperator",          "T.5.1", "Area operator gradient"),
    (13, "iso_direct_indep_geom_volume",     "IsoDirectIndepGeomVolume",      "Iso-Direct-Indep + Geom-VolumeOperator",        "T.5.2", "Volume operator gradient"),
    (14, "iso_direct_indep_geom_geodesic",   "IsoDirectIndepGeomGeodesic",    "Iso-Direct-Indep + Geom-GeodesicDistance",      "T.5.3", "Geodesic distance order"),
]


def write_compilation(idx, module, cls, title, t_class, t_desc):
    lines = [
        f'"""',
        f'{idx:03d}. {title}',
        f'Classification: S = Iso-Direct-Independent   T = {t_class} ({t_desc})',
        f'"""',
        f'',
        f'from __future__ import annotations',
        f'import numpy as np',
        f'from music.midi.midi_entity import MIDIEntity',
        f'from core.compilation.components.spinfoam import SpinfoamComplex',
        f'from core.compilation.compilation_scheme import CompilationScheme',
        f'',
        f'',
        f'class {cls}(CompilationScheme):',
        f'',
        f'    @property',
        f'    def scheme_id(self) -> str:',
        f'        return "{module}"',
        f'',
        f'    @property',
        f'    def description(self) -> str:',
        f'        return "{title}"',
        f'',
        f'    def compile_to_spinfoam(self, midi: MIDIEntity) -> SpinfoamComplex:',
        f'        raise NotImplementedError',
        f'',
        f'    def _vertex_amplitude(self, sigma: np.ndarray, t: int, T: int) -> complex:',
        f'        raise NotImplementedError',
        f'',
        f'    def _time_index(self, sigma: np.ndarray, t: int, T: int,',
        f'                    piano_roll: np.ndarray) -> float:',
        f'        raise NotImplementedError',
    ]
    path = f"core/compilation/schemes/s_{idx:03d}_{module}.py"
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  {path}")


def write_experiment(idx, module, cls, title):
    idx3 = f"{idx:03d}"
    lines = [
        f'"""',
        f'exp_{idx3}_{module}.py',
        f'Experiment for scheme {idx:02d}: {title}',
        f'"""',
        f'',
        f'from __future__ import annotations',
        f'import os',
        f'import numpy as np',
        f'',
        f'from music.midi.midi_entity import MIDIEntity',
        f'from core.compilation.components.spinfoam import SpinfoamComplex',
        f'from core.compilation.schemes.s_{idx:03d}_{module} import {cls}',
        f'',
        f'SCHEME_TITLE = "{title}"',
        f'SCHEME_IDX   = {idx}',
        f'',
        f'',
        f'def run(midi_path: str = None, verbose: bool = True) -> SpinfoamComplex:',
        f'    """Run experiment {idx:02d}: {title}"""',
        f'',
        f'    if midi_path and os.path.exists(midi_path):',
        f'        midi = MIDIEntity.from_file(midi_path)',
        f'    else:',
        f'        pr = np.zeros((88, 4), dtype=np.float32)',
        f'        for pitch in [60, 64, 67]:',
        f'            pr[pitch - 21, :] = 1.0',
        f'        midi = MIDIEntity.from_piano_roll(pr)',
        f'',
        f'    scheme = {cls}()',
        f'    sf = scheme.compile_to_spinfoam(midi)',
        f'    return sf',
        f'',
        f'',
        f'if __name__ == "__main__":',
        f'    import sys',
        f'    path = sys.argv[1] if len(sys.argv) > 1 else None',
        f'    run(midi_path=path, verbose=True)',
    ]
    path = f"experiments/exp_{idx3}_{module}.py"
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  {path}")


if __name__ == "__main__":
    os.makedirs("core/compilation/schems", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    # Remove old prefixed files
    for f in os.listdir("compilation"):
        if f.startswith("c") and f[1:2].isdigit():
            os.remove(f"compilation/{f}")

    print("Generating compilation schemes...")
    for idx, module, cls, title, t_class, t_desc in SCHEMES:
        write_compilation(idx, module, cls, title, t_class, t_desc)

    print("\nGenerating experiment files...")
    for idx, module, cls, title, t_class, t_desc in SCHEMES:
        write_experiment(idx, module, cls, title)

    print("\nDone.")