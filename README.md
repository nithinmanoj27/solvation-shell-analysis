# Automated Solvation Shell Analysis for Polymer Electrolytes 

A Python-based scientific computing pipeline that automates Molecular Dynamics (MD) simulation analysis for polymer electrolyte systems.

The project directly reads GROMOS `.g96` trajectory files and computes structural properties such as Radial Distribution Functions (RDF), neighbor counts, and solvation shell populations without requiring manual preprocessing.

Developed as a Design Credit Project at IIT Jodhpur.

---

#  Features

- Automated parsing of GROMOS `.g96` molecular dynamics trajectory files.
- Multi-frame trajectory analysis.
- Periodic Boundary Condition (PBC) handling.
- Neighbor counting for polymer-ion interactions.
- Radial Distribution Function (RDF) computation.
- Solvation shell population analysis.
- Automatic generation of plots and CSV outputs.
- Reproducible scientific analysis pipeline.

---

#  Tech Stack

- Python
- NumPy
- SciPy
- Pandas
- Matplotlib

---

#  Project Workflow

The pipeline performs the following steps:

1. Read polymer, cation, and anion `.g96` trajectory files.

2. Extract atomic coordinates from all simulation frames.

3. Apply Periodic Boundary Conditions (PBC).

4. Compute local neighbor counts.

5. Calculate Radial Distribution Functions (RDF).

6. Identify solvation shell boundaries.

7. Generate probability distributions for ion populations.

8. Export results as plots and CSV files.

---

#  Generated Outputs

The project automatically generates:

### RDF Analysis

- Polymer-Cation RDF
- Polymer-Anion RDF

### Solvation Shell Analysis

- First shell population distribution
- Ion occupancy probability P(n)

### CSV Files

- Neighbor count summaries
- RDF numerical values
- Solvation shell statistics

### Visualizations

- RDF plots
- Solvation shell population plots

---

#  Example Input Files

The pipeline accepts GROMOS `.g96` files such as:

```text
polymer.g96
cation.g96
anion.g96
```

---

#  How to Run

## 1. Clone the repository

```bash
git clone https://github.com/nithinmanoj27/solvation-shell-analysis.git

cd solvation-shell-analysis
```

## 2. Install dependencies

```bash
pip install numpy scipy pandas matplotlib
```

## 3. Place input files

Add the following files inside the project directory:

```text
polymer.g96
cation.g96
anion.g96
```

## 4. Run the analysis

```bash
python main.py
```

(Replace `main.py` with your actual Python file name.)

---

#  Core Concepts Used

- Molecular Dynamics (MD)
- Polymer Electrolytes
- Solvation Shells
- Radial Distribution Functions (RDF)
- Periodic Boundary Conditions (PBC)
- KDTree-based nearest neighbor search

---

#  Applications

This project can be used in:

- Battery material research
- Polymer electrolyte analysis
- Scientific computing
- Molecular dynamics data analysis
- Computational chemistry research

---

#  Project Information

**Project:** Design Credit Project

**Institute:** Indian Institute of Technology Jodhpur

**Duration:** Aug 2024 – Nov 2024

---

# 👥 Contributors

- Yerra Nithin Manoj (B22CS066)

