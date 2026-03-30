import argparse
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *


# ── 参数解析 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="用 OpenMM 模拟两分子相互作用")
parser.add_argument("--mol1", default="pdb_folder/il_lipid.pdb", help="第一个分子的 PDB 文件")
parser.add_argument("--mol2", default="data/2L7Bfixed.pdb", help="第二个分子的 PDB 文件")
parser.add_argument("--forcefield", default="amber14-all.xml",
                    help="力场文件 (默认: amber14-all.xml)")
parser.add_argument("--water_model", default="amber14/tip3pfb.xml",
                    help="水模型文件 (默认: amber14/tip3pfb.xml)")
parser.add_argument("--steps", type=int, default=10000, help="MD 步数 (默认: 10000)")
parser.add_argument("--step_size", type=float, default=2.0, help="步长 fs (默认: 2.0)")
parser.add_argument("--temperature", type=float, default=300.0, help="温度 K (默认: 300)")
parser.add_argument("--output", default="trajectory.dcd", help="轨迹输出文件")
parser.add_argument("--report_interval", type=int, default=500, help="报告间隔步数")
parser.add_argument("--no_solvent", action="store_true", help="不添加溶剂（真空模拟）")
args = parser.parse_args()

pdb1 = PDBFile(args.mol1)
pdb2 = PDBFile(args.mol2)

modeller = Modeller(pdb1.topology, pdb1.positions)
modeller.add(pdb2.topology, pdb2.positions)

for res in modeller.topology.residues():
    if res.index < 2:
        print("Residue:", res.index, res.name)
        for atom in res.atoms():
            print("  ", atom.name, atom.element)
