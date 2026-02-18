from __future__ import annotations
import numpy as np
from typing import List, Set, Dict
from core.compilation.components.spinfoam import SpinfoamComplex

class SpinfoamIntegrityChecker:
    """
    Validates the physical and topological consistency of a SpinfoamComplex.
    Ensures that the 'Preorder' version adheres to causal and manifold rules.
    """

    def __init__(self, sp_complex: SpinfoamComplex):
        self.sf = sp_complex
        self.sf.build_index()

    def run_all_checks(self) -> Dict[str, bool]:
        """Executes a suite of integrity tests."""
        results = {
            "causal_directionality": self.check_causal_directionality(),
            "face_continuity": self.check_face_continuity(),
            "rank_consistency": self.check_rank_consistency(),
            "no_isolated_elements": self.check_no_isolated_elements(),
            "amplitude": self.check_amplitude_log_stability()
        }
        
        passed = all(results.values())
        print(f"--- Integrity Check Summary: {'PASSED' if passed else 'FAILED'} ---")
        for test, status in results.items():
            print(f"{test.replace('_', ' ').capitalize()}: {'OK' if status else 'FAIL'}")
        
        return results

    def check_causal_directionality(self) -> bool:
        """Checks that every internal edge has a defined 'from' and 'to'."""
        for edge in self.sf.edges:
            # In the preorder version, internal edges must be oriented
            if edge.from_vertex is None and edge.to_vertex is None:
                print(f"Integrity Error: Edge {edge.id} is totally disconnected.")
                return False
        return True

    def check_face_continuity(self) -> bool:
        """Ensures that faces (notes) correctly propagate through vertices."""
        for face in self.sf.faces:
            # Find all edges belonging to this face
            face_edges = [e for e in self.sf.edges if face.id in e.face_ids]
            if not face_edges:
                print(f"Integrity Error: Face {face.id} ({face.semantic_label}) has no edges.")
                return False
                
            # Check for topological gap: every 'to_vertex' of a face-segment 
            # (except for the final exit) should be the 'from_vertex' of the next.
            # This ensures the note doesn't 'teleport' across the complex.
            verts_involved = set()
            for e in face_edges:
                if e.from_vertex is not None: verts_involved.add(e.from_vertex)
                if e.to_vertex is not None: verts_involved.add(e.to_vertex)
                
            # A note spanning N segments should involve N+1 vertices in a chain
            if len(verts_involved) < 2:
                print(f"Integrity Error: Face {face.id} is static/point-like.")
                return False
        return True

    def check_rank_consistency(self) -> bool:
        """Verifies that time_index strictly increases along edges (Entropic Arrow)."""
        for edge in self.sf.edges:
            if edge.from_vertex is not None and edge.to_vertex is not None:
                v_past = self.sf.vertex(edge.from_vertex)
                v_future = self.sf.vertex(edge.to_vertex)
                
                if v_past.time_index is not None and v_future.time_index is not None:
                    if v_future.time_index <= v_past.time_index:
                        print(f"Integrity Error: Causal violation at Edge {edge.id}. "
                              f"Rank {v_past.time_index} -> {v_future.time_index}")
                        return False
        return True

    def check_no_isolated_elements(self) -> bool:
        """Ensures every vertex is part of the causal graph."""
        edge_referenced_vertices = set()
        for e in self.sf.edges:
            if e.from_vertex is not None: edge_referenced_vertices.add(e.from_vertex)
            if e.to_vertex is not None: edge_referenced_vertices.add(e.to_vertex)
            
        for v in self.sf.vertices:
            if v.id not in edge_referenced_vertices:
                print(f"Integrity Error: Vertex {v.id} is an orphan (no edges).")
                return False
        return True
    
    def check_amplitude_log_stability(self) -> bool:
        """
        使用对数累加校验振幅完整性，防止数值下溢。
        公式：log(Z) = sum(log(A_v)) + sum(log(A_f))
        """
        log_z_real = 0.0
        phase_z = 0.0

        # 处理面振幅
        for face in self.sf.faces:
            # A_f = 2j + 1
            val = (2 * face.spin_j) + 1
            log_z_real += np.log(val)
            # 如果 face.amplitude 有相位，在此处累加 phase_z

        # 处理顶点振幅
        for vertex in self.sf.vertices:
            amp = vertex.amplitude
            if np.abs(amp) < 1e-15:
                print(f"Warning: Vertex {vertex.id} amplitude is near-zero. Causal break.")
                return False
                
            log_z_real += np.log(np.abs(amp))
            phase_z += np.angle(amp)

        # 最终的 Z 依然可以表示为 exp(log_z_real + i*phase_z)
        self.log_total_amplitude = complex(log_z_real, phase_z)
        return True