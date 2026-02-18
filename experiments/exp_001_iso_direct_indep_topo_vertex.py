"""
exp_001_iso_direct_indep_topo_vertex.py
Experiment for scheme 01: Iso-Direct-Indep + Topo-Vertex-PartialOrder
"""

from __future__ import annotations
import os
import numpy as np

from music.midi.midi_entity import MIDIEntity
from core.compilation.components.spinfoam import SpinfoamComplex
from core.compilation.schemes.s_001_iso_direct_indep_topo_vertex import IsoDirectIndepTopoVertex

SCHEME_TITLE = "Iso-Direct-Indep + Topo-Vertex-PartialOrder"
SCHEME_IDX   = 1


def run(midi_path: str = 'assets/first_rabbit.mid', verbose: bool = True) -> SpinfoamComplex:
    """Run experiment 01: Iso-Direct-Indep + Topo-Vertex-PartialOrder"""

    if midi_path and os.path.exists(midi_path):
        print(f'load midi from file {midi_path}.')
        midi = MIDIEntity.from_file(midi_path)
        print(midi)
        midi.print_notes_by_tick(limit=100)
    else:
        pr = np.zeros((88, 4), dtype=np.float32)
        for pitch in [60, 64, 67]:
            pr[pitch - 21, :] = 1.0
        midi = MIDIEntity.from_piano_roll(pr)

    scheme = IsoDirectIndepTopoVertex()
    sf = scheme.compile_to_spinfoam(midi)

    return sf


import os
from pathlib import Path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Spinfoam MIDI Compiler (Preorder Version)")
    parser.add_argument("--mid_file", default='assets/first_rabbit.mid')
    parser.add_argument("--spinfoam_zip_file", default='outputs/out.sp')

    parser.add_argument("--output_file_path", default='outputs/out.sp')
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--skip_check", action="store_true")

    args = parser.parse_args()
    
    # 1. Directory Integrity Check
    # We use Pathlib for modern, cross-platform directory handling
    output_path = Path(args.output_file_path)
    if not output_path.parent.exists():
        print(f"Creating directory: {output_path.parent}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    
    if args.decode:
        print(f"Loading Spinfoam from: {args.spinfoam_zip_file}")
        sf = SpinfoamComplex.deserialize(args.spinfoam_zip_file)
        
        if not args.skip_check:
            from core.compilation.spinfoam_integrity_checker import SpinfoamIntegrityChecker
            checker = SpinfoamIntegrityChecker(sf)
        if not checker.run_all_checks()["causal_directionality"]:
            print("Error: Spinfoam fails causal integrity. Aborting decode.")
            exit(1)
        
        scheme = IsoDirectIndepTopoVertex()
        scheme.decode_to_midi_file(sf=sf, init_vertex_id=0, output_path=args.output_file_path)
        # 3. 初始化解码器
        # 我们假设第一个顶点的 ID 是 0 (这是 PreorderCompiler 的初始锚点)

        # 4. 执行解码并保存 MIDI
        # 输出路径使用 args.output_file_path，通常应该是一个 .mid 文件
    else:
        # 2. Compilation (Preorder / Log-Space stable)
        # Assuming 'run' is your wrapper that calls PreorderCompiler
        sf = run(midi_path=args.mid_file, verbose=True)

        # 3. Serialization
        # This will now save the topological skeleton and constant log-amplitudes
        sf.serialize(str(output_path))
        from core.compilation.spinfoam_integrity_checker import SpinfoamIntegrityChecker

        # 4. Final Integrity Check (Physics Guardrail)
        print(f'Run integrity checker...')

        # checker = SpinfoamIntegrityChecker(sf)
        # check_results = checker.run_all_checks()
        # print(check_results)
        scheme = IsoDirectIndepTopoVertex()

        scheme.decode_to_midi_file(sf=sf, init_vertex_id=0, output_path="test.mid")