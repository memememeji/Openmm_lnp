"""
OpenMM LNP builder and MD driver.

Notes
-----
- The input PDBs for A/B/C/D should already represent the protonation states that
  correspond to the target pH. OpenMM/OpenFF will not automatically retitrate
  arbitrary custom lipids from pH alone.
- The pH argument is therefore treated as simulation metadata plus the condition
  used when adding any standard hydrogens and aqueous ions.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openff.toolkit import Molecule
from openmm import MonteCarloBarostat, LangevinMiddleIntegrator, Platform
from openmm.app import DCDReporter, ForceField, Modeller, PDBFile, Simulation, StateDataReporter, PME, HBonds
from openmm.unit import atmosphere, kelvin, molar, nanometer, picosecond, femtosecond
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from rdkit import Chem


RATIO_MAP = {
    "ionizable": 35.0,
    "helper": 16.0,
    "peg": 2.5,
    "cholesterol": 46.5,
}


@dataclass
class MoleculeSpec:
    key: str
    pdb_path: Path
    residue_name: str
    off_molecule: Molecule
    topology: object
    positions_nm: object
    radius_nm: float


def parse_args():
    """
    解析命令行参数，用于配置LNP(脂质纳米粒)的构建和模拟参数。
    Returns:
        argparse.Namespace: 包含所有命令行参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(
        description="Build and simulate an LNP-like nanoparticle with OpenMM."
    )

    # 脂质相关参数
    parser.add_argument("--lipid-a", default="data/A1B1C1.pdb", help="Ionizable lipid PDB.")  # 可电离脂质的PDB文件路径
    parser.add_argument("--lipid-b", default="data/DOPE.pdb", help="Helper lipid PDB.")  # 辅助脂质的PDB文件路径
    parser.add_argument("--lipid-c", default="data/PEG_smiles.pdb", help="PEG lipid PDB.")  # PEG脂质的PDB文件路径
    parser.add_argument("--lipid-d", default="data/cholesterol.pdb", help="Cholesterol PDB.")  # 胆固醇的PDB文件路径

    
    # LNP组成参数
    parser.add_argument("--total-molecules", type=int, default=200, help="Total lipid count in the LNP.")  # LNP中脂质总数

    
    # 环境条件参数
    parser.add_argument("--ph", type=float, default=7.4, help="Target aqueous pH metadata.")  # 目标水相pH值
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K.")  # 温度(开尔文)
    parser.add_argument("--pressure", type=float, default=1.0, help="Pressure in atm.")  # 压力(大气压)
    parser.add_argument("--ionic-strength", type=float, default=0.15, help="Salt concentration in mol/L.")  # 盐浓度(摩尔/升)
    parser.add_argument("--positive-ion", default="Na+", help="Positive ion name for addSolvent.")  # 溶剂中正离子名称
    parser.add_argument("--negative-ion", default="Cl-", help="Negative ion name for addSolvent.")  # 溶剂中负离子名称
    parser.add_argument("--water-model", default="tip3p", help="Water model name passed to addSolvent.")  # 水模型名称

    
    # 结构参数
    parser.add_argument("--sphere-radius-nm", type=float, default=None, help="Initial LNP packing radius.")  # LNP初始堆积半径(纳米)
    parser.add_argument("--solvent-padding-nm", type=float, default=1.5, help="Water box padding beyond solute.")  # 溶质周围的水盒子填充(纳米)
    parser.add_argument("--min-center-gap-nm", type=float, default=0.18, help="Extra gap between molecular centers.")  # 分子中心之间的额外间隙(纳米)

    
    # 模拟参数
    parser.add_argument("--minimize-iters", type=int, default=1000, help="Energy minimization iterations.")  # 能量最小化迭代次数
    parser.add_argument("--equil-steps", type=int, default=25000, help="Equilibration MD steps.")  # 平衡MD步数
    parser.add_argument("--prod-steps", type=int, default=100000, help="Production MD steps.")  # 生产MD步数
    parser.add_argument("--report-interval", type=int, default=1000, help="Reporter interval in steps.")  # 报告间隔(步数)
    parser.add_argument("--step-size-fs", type=float, default=2.0, help="Integrator step size in fs.")  # 积分器步长(飞秒)
    parser.add_argument("--friction-per-ps", type=float, default=1.0, help="Langevin friction in ps^-1.")  # Langevin摩擦系数(每皮秒)

    
    # 其他参数
    parser.add_argument("--seed", type=int, default=20260321, help="Random seed.")  # 随机种子
    parser.add_argument("--platform", default=None, help="Optional OpenMM platform name.")  # 可选的OpenMM平台名称
    parser.add_argument("--output-prefix", default="lnp_simulation", help="Prefix for output files.")  # 输出文件前缀
    return parser.parse_args()


def allocate_counts(total_molecules: int) -> dict:
    raw = {
        key: total_molecules * value / 100.0
        for key, value in RATIO_MAP.items()
    }
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total_molecules - sum(counts.values())
    ranked = sorted(raw.items(), key=lambda item: item[1] - math.floor(item[1]), reverse=True)
    for key, _ in ranked[:remainder]:
        counts[key] += 1
    return counts


def _infer_residue_name(key: str) -> str:
    return {
        "ionizable": "LIA",
        "helper": "LHB",
        "peg": "LPC",
        "cholesterol": "CHL",
    }[key]


def load_openff_molecule(pdb_path: Path, residue_name: str) -> Molecule:
    rdkit_mol = Chem.MolFromPDBFile(
        str(pdb_path),
        removeHs=False,
        sanitize=True,
        proximityBonding=False,
    )
    if rdkit_mol is None:
        raise ValueError(f"Failed to parse {pdb_path}. Ensure CONECT records exist in the PDB.")
    mol = Molecule.from_rdkit(
        rdkit_mol,
        allow_undefined_stereo=True,
        hydrogens_are_explicit=True,
    )
    mol.name = residue_name
    for atom_index, atom in enumerate(mol.atoms, start=1):
        atom.name = f"{atom.symbol}{atom_index}"
    return mol


def prepare_spec(key: str, pdb_path: Path) -> MoleculeSpec:
    residue_name = _infer_residue_name(key)
    off_molecule = load_openff_molecule(pdb_path, residue_name)
    topology = off_molecule.to_topology().to_openmm()
    positions_nm = off_molecule.conformers[0].to("nanometer")
    coords = positions_nm.magnitude
    center = coords.mean(axis=0)
    radius_nm = float(np.linalg.norm(coords - center, axis=1).max()) + 0.05
    return MoleculeSpec(
        key=key,
        pdb_path=pdb_path,
        residue_name=residue_name,
        off_molecule=off_molecule,
        topology=topology,
        positions_nm=positions_nm,
        radius_nm=radius_nm,
    )


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return np.array(
        [
            [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)],
        ]
    )


def sample_center(
    spec: MoleculeSpec,
    outer_radius_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if spec.key == "peg":
        radius = outer_radius_nm * rng.uniform(0.72, 0.95)
    elif spec.key == "helper":
        radius = outer_radius_nm * (rng.random() ** (1.0 / 3.0)) * 0.95
    elif spec.key == "cholesterol":
        radius = outer_radius_nm * (rng.random() ** (1.0 / 3.0)) * 0.85
    else:
        radius = outer_radius_nm * (rng.random() ** (1.0 / 3.0))

    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    return direction * radius


def place_molecules(specs: dict, counts: dict, sphere_radius_nm: float, extra_gap_nm: float, seed: int):
    rng = np.random.default_rng(seed)
    placements = []
    occupied = []

    placement_order = ["peg", "helper", "cholesterol", "ionizable"]
    for key in placement_order:
        spec = specs[key]
        for copy_index in range(counts[key]):
            placed = False
            for _ in range(2000):
                center = sample_center(spec, sphere_radius_nm - spec.radius_nm, rng)
                clash = False
                for old_center, old_radius in occupied:
                    min_allowed = spec.radius_nm + old_radius + extra_gap_nm
                    if np.linalg.norm(center - old_center) < min_allowed:
                        clash = True
                        break
                if not clash:
                    placements.append((spec, copy_index + 1, center, random_rotation_matrix(rng)))
                    occupied.append((center, spec.radius_nm))
                    placed = True
                    break
            if not placed:
                raise RuntimeError(
                    f"Could not place {key} molecule {copy_index + 1}. Increase --sphere-radius-nm or lower --total-molecules."
                )
    return placements


def transform_positions(positions_nm, rotation: np.ndarray, center_nm: np.ndarray):
    coords = positions_nm.magnitude
    centered = coords - coords.mean(axis=0)
    transformed = centered @ rotation.T + center_nm
    return transformed * nanometer


def build_initial_modeller(placements):
    first_spec, first_index, first_center, first_rotation = placements[0]
    modeller = Modeller(
        first_spec.topology,
        transform_positions(first_spec.positions_nm, first_rotation, first_center),
    )
    for spec, _, center, rotation in placements[1:]:
        modeller.add(spec.topology, transform_positions(spec.positions_nm, rotation, center))
    return modeller


def choose_sphere_radius(specs: dict, counts: dict) -> float:
    molecular_volume = 0.0
    for key, count in counts.items():
        molecular_volume += count * (4.0 / 3.0) * math.pi * (specs[key].radius_nm ** 3)
    return max(2.5, (molecular_volume / 0.58 / ((4.0 / 3.0) * math.pi)) ** (1.0 / 3.0))


def write_composition(prefix: str, counts: dict, args):
    composition_path = Path(f"{prefix}_composition.txt")
    lines = [
        f"Target pH metadata: {args.ph}",
        f"Total molecules: {sum(counts.values())}",
        f"Ionizable lipid (A): {counts['ionizable']}",
        f"Helper lipid (B): {counts['helper']}",
        f"PEG lipid (C): {counts['peg']}",
        f"Cholesterol (D): {counts['cholesterol']}",
        f"Ionic strength (M): {args.ionic_strength}",
        f"Positive ion: {args.positive_ion}",
        f"Negative ion: {args.negative_ion}",
    ]
    composition_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    input_paths = {
        "ionizable": Path(args.lipid_a).expanduser().resolve(),
        "helper": Path(args.lipid_b).expanduser().resolve(),
        "peg": Path(args.lipid_c).expanduser().resolve(),
        "cholesterol": Path(args.lipid_d).expanduser().resolve(),
    }

    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input PDB(s): " + ", ".join(missing))

    counts = allocate_counts(args.total_molecules)
    specs = {key: prepare_spec(key, path) for key, path in input_paths.items()}

    sphere_radius_nm = args.sphere_radius_nm or choose_sphere_radius(specs, counts)
    placements = place_molecules(
        specs=specs,
        counts=counts,
        sphere_radius_nm=sphere_radius_nm,
        extra_gap_nm=args.min_center_gap_nm,
        seed=args.seed,
    )

    modeller = build_initial_modeller(placements)
    with open(f"{args.output_prefix}_initial_lnp.pdb", "w") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle)

    molecules = [spec.off_molecule for spec in specs.values()]
    smirnoff = SMIRNOFFTemplateGenerator(molecules=molecules, forcefield="openff-2.3.0")
    forcefield = ForceField("amber/tip3p_standard.xml", "amber/tip3p_HFE_multivalent.xml")
    forcefield.registerTemplateGenerator(smirnoff.generator)

    try:
        modeller.addHydrogens(forcefield, pH=args.ph)
    except Exception as exc:
        print(f"Skipping addHydrogens for custom lipids: {exc}")

    modeller.addSolvent(
        forcefield,
        model=args.water_model,
        padding=args.solvent_padding_nm * nanometer,
        positiveIon=args.positive_ion,
        negativeIon=args.negative_ion,
        ionicStrength=args.ionic_strength * molar,
        neutralize=True,
    )

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
    )
    system.addForce(MonteCarloBarostat(args.pressure * atmosphere, args.temperature * kelvin, 25))

    integrator = LangevinMiddleIntegrator(
        args.temperature * kelvin,
        args.friction_per_ps / picosecond,
        args.step_size_fs * femtosecond,
    )

    if args.platform:
        platform = Platform.getPlatformByName(args.platform)
        simulation = Simulation(modeller.topology, system, integrator, platform)
    else:
        simulation = Simulation(modeller.topology, system, integrator)

    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=args.minimize_iters)
    simulation.context.setVelocitiesToTemperature(args.temperature * kelvin, args.seed)

    simulation.reporters.append(DCDReporter(f"{args.output_prefix}.dcd", args.report_interval))
    simulation.reporters.append(
        StateDataReporter(
            f"{args.output_prefix}_state.csv",
            args.report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            density=True,
            volume=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=args.equil_steps + args.prod_steps,
            separator=",",
        )
    )

    print("LNP composition counts:", counts)
    print(f"Initial packing radius: {sphere_radius_nm:.3f} nm")
    print(f"Target pH metadata: {args.ph}")
    print("Input A/B/C/D protonation states must already match the intended pH.")
    print("Running equilibration...")
    simulation.step(args.equil_steps)
    print("Running production...")
    simulation.step(args.prod_steps)

    final_state = simulation.context.getState(getPositions=True, getEnergy=True)
    with open(f"{args.output_prefix}_final.pdb", "w") as handle:
        PDBFile.writeFile(modeller.topology, final_state.getPositions(), handle)

    write_composition(args.output_prefix, counts, args)
    print("Finished.")
    print(f"Final potential energy: {final_state.getPotentialEnergy()}")


if __name__ == "__main__":
    main()
