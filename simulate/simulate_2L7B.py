"""
两分子相互作用力与能量模拟 - OpenMM
用法: python simulate_interaction.py --mol1 molecule1.pdb --mol2 molecule2.pdb
"""

# PDB修复
'''from pdbfixer import PDBFixer
from openmm.app import PDBFile

fixer = PDBFixer(filename='data/2L7B.pdb')
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(pH=7.0)

PDBFile.writeFile(fixer.topology, fixer.positions, open('data/2L7Bfixed.pdb', 'w'))'''


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

from openff.toolkit import Molecule
from openmm.app import ForceField
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from rdkit import Chem
from openmm import NonbondedForce, CustomNonbondedForce

smiles = "CCCCCCCCCCCCNC(=O)C(CCCCCOC(=O)CCCCCCCC)NCCN(C)C"

# 1. 从 SMILES 建分子
mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

# 2. 生成 3D 构象
#mol.generate_conformers(n_conformers=10)
mol.generate_conformers(n_conformers=1)

# 3. 分配部分电荷
#mol.assign_partial_charges("am1bcc")
mol.assign_partial_charges("gasteiger")

# 4.导出pdb文件
rdmol = mol.to_rdkit()
Chem.MolToPDBFile(rdmol, "pdb_folder/il_lipid.pdb")

# 4. 注册小分子模板+力场构建
smirnoff = SMIRNOFFTemplateGenerator(
    molecules=mol,
    forcefield="openff-2.3.0"
)

forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')
forcefield.registerTemplateGenerator(smirnoff.generator)


# ── 1. 读取并合并两个 PDB ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  两分子相互作用模拟  (OpenMM)")
print(f"{'='*60}")
print(f"  分子1 : {args.mol1}")
print(f"  分子2 : {args.mol2}")
print(f"  力场  : {args.forcefield}")
print(f"{'='*60}\n")

pdb1 = PDBFile(args.mol1)
pdb2 = PDBFile(args.mol2)

# 合并两个分子到同一 Modeller
modeller = Modeller(pdb1.topology, pdb1.positions)
modeller.add(pdb2.topology, pdb2.positions)
print(f"[1/6] PDB 合并完成: {modeller.topology.getNumAtoms()} 个原子")


# ── 2. 建立力场 ────────────────────────────────────────────────────────────────

'''try:
    forcefield = ForceField(args.forcefield, args.water_model)
except Exception:
    print("      未找到水模型，尝试仅加载蛋白力场...")
    forcefield = ForceField(args.forcefield)
print(f"[2/6] 力场加载完成: {args.forcefield}")'''


# ── 3. 可选: 添加氢与溶剂 ─────────────────────────────────────────────────────
try:
    modeller.addHydrogens(forcefield)
    print(f"[3/6] 氢原子添加完成，共 {modeller.topology.getNumAtoms()} 个原子")
except Exception as e:
    print(f"[3/6] 氢原子添加跳过 ({e})")

if not args.no_solvent:
    try:
        modeller.addSolvent(forcefield, model="tip3p", padding=1.0*nanometer)
        print(f"[3/6] 溶剂添加完成，共 {modeller.topology.getNumAtoms()} 个原子")
    except Exception as e:
        print(f"[3/6] 溶剂添加失败，使用真空模拟 ({e})")

# 检查残基原子
"""for res in modeller.topology.residues():
    if res.index < 2:
        print("Residue:", res.index, res.name)
        for atom in res.atoms():
            print("  ", atom.name, atom.element)"""


# ── 4. 创建系统 ────────────────────────────────────────────────────────────────
system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=PME if not args.no_solvent else NoCutoff,
    nonbondedCutoff=1.0*nanometer,
    constraints=HBonds
)
print(f"[4/6] 系统创建完成，共 {system.getNumForces()} 个力项")




# 关键部分
# ── 5. 添加自定义相互作用力分析 (分组能量) ─────────────────────────────────────
# 通过力组 (force group) 分离两分子之间的非键相互作用
n_atoms_mol1 = pdb1.topology.getNumAtoms()  # 第一个分子的原子数

for i, force in enumerate(system.getForces()):
    force.setForceGroup(0)   # 所有力默认组 0

# 克隆 NonbondedForce 用于仅计算分子间相互作用（组 1）
for force in system.getForces():
    if isinstance(force, NonbondedForce):
        nbforce = force
        break

# 创建自定义非键力: 仅计算跨分子原子对
custom_nb = CustomNonbondedForce(
    "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + 138.935456*charge1*charge2/r;"
    "sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)"
)
custom_nb.addPerParticleParameter("charge")
custom_nb.addPerParticleParameter("sigma")
custom_nb.addPerParticleParameter("epsilon")
custom_nb.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
custom_nb.setCutoffDistance(2.0*nanometer)


# 设置原子参数并定义两个相互作用组
mol1_indices = list(range(n_atoms_mol1))
mol2_indices = list(range(n_atoms_mol1, system.getNumParticles()))

for i in range(system.getNumParticles()):
    charge, sigma, epsilon = nbforce.getParticleParameters(i)
    custom_nb.addParticle([charge, sigma, epsilon])

for i in range(nbforce.getNumExceptions()):
    p1, p2, _, _, _ = nbforce.getExceptionParameters(i)
    custom_nb.addExclusion(p1, p2)

custom_nb.addInteractionGroup(set(mol1_indices), set(mol2_indices))
custom_nb.setForceGroup(1)   # 组 1 = 分子间相互作用
system.addForce(custom_nb)
print(f"[4/6] 跨分子非键力 (组1) 添加完成")
print(f"       分子1: {n_atoms_mol1} 原子 | 分子2: {len(mol2_indices)} 原子")




# ── 6. 积分器与模拟 ────────────────────────────────────────────────────────────
integrator = LangevinMiddleIntegrator(
    args.temperature * kelvin,
    1.0 / picosecond,
    args.step_size * femtoseconds
)
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
print(f"\n[5/6] 能量最小化中...")
simulation.minimizeEnergy(maxIterations=500)
print("       能量最小化完成")


# ── 7. 报告器 ──────────────────────────────────────────────────────────────────
simulation.reporters.append(DCDReporter(args.output, args.report_interval))
simulation.reporters.append(
    StateDataReporter(
        "energy_log.csv",
        args.report_interval,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        separator=","
    )
)
simulation.reporters.append(
    StateDataReporter(
        None,          # None → 打印到 stdout
        args.report_interval,
        step=True,
        potentialEnergy=True,
        temperature=True
    )
)


# ── 8. 运行 MD 并每隔 report_interval 步记录分子间能量 ─────────────────────────
print(f"\n[6/6] 开始 MD 模拟 ({args.steps} 步, 步长 {args.step_size} fs)...\n")

interaction_log = []

n_blocks = args.steps // args.report_interval
for block in range(n_blocks):
    simulation.step(args.report_interval)

    # 仅使用力组 1 计算分子间相互作用能
    state_inter = simulation.context.getState(
        getEnergy=True, getForces=True, groups={1}
    )
    inter_energy = state_inter.getPotentialEnergy().value_in_unit(kilocalories_per_mole)

    # 分子1 质心受力 (来自分子间相互作用)
    forces = state_inter.getForces(asNumpy=True).value_in_unit(
        kilocalories_per_mole / angstrom
    )
    f_mol1 = forces[mol1_indices]
    f_mol2 = forces[mol2_indices]
    net_f1 = np.linalg.norm(f_mol1.sum(axis=0))
    net_f2 = np.linalg.norm(f_mol2.sum(axis=0))

    step_num = (block + 1) * args.report_interval
    interaction_log.append({
        "step": step_num,
        "inter_energy_kcal": inter_energy,
        "net_force_mol1_kcal_per_mol_A": net_f1,
        "net_force_mol2_kcal_per_mol_A": net_f2,
    })
    print(f"  Step {step_num:>7d} | 相互作用能: {inter_energy:>12.4f} kcal/mol"
          f" | |F_mol1|: {net_f1:.4f}  |F_mol2|: {net_f2:.4f} kcal/(mol·Å)")

# 剩余步数
remainder = args.steps % args.report_interval
if remainder:
    simulation.step(remainder)


# ── 9. 保存相互作用能 CSV ──────────────────────────────────────────────────────
import csv
with open("interaction_energy.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=interaction_log[0].keys())
    writer.writeheader()
    writer.writerows(interaction_log)

print(f"\n{'='*60}")
print("  模拟完成！")
print(f"  轨迹文件        : {args.output}")
print(f"  总能量日志      : energy_log.csv")
print(f"  分子间相互作用  : interaction_energy.csv")
print(f"{'='*60}\n")


# ── 10. 最终状态摘要 ──────────────────────────────────────────────────────────
final_state = simulation.context.getState(getEnergy=True, getPositions=True)
print("最终系统状态:")
print(f"  总势能  : {final_state.getPotentialEnergy().value_in_unit(kilocalories_per_mole):.4f} kcal/mol")
print(f"  总动能  : {final_state.getKineticEnergy().value_in_unit(kilocalories_per_mole):.4f} kcal/mol")

final_inter = simulation.context.getState(getEnergy=True, groups={1})
print(f"  分子间能: {final_inter.getPotentialEnergy().value_in_unit(kilocalories_per_mole):.4f} kcal/mol")

# 最终质心距离
pos = final_state.getPositions(asNumpy=True).value_in_unit(angstrom)
com1 = pos[mol1_indices].mean(axis=0)
com2 = pos[mol2_indices].mean(axis=0)
dist = np.linalg.norm(com1 - com2)
print(f"  质心距离: {dist:.3f} Å")