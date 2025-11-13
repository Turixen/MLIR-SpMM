# MLIR Sparse Matrix Multiplication Research

This repository contains tools and code for research on sparse matrix multiplication using MLIR (Multi-Level Intermediate Representation). The project focuses on generating MLIR code for various sparse matrix operations and benchmarking them with real-world matrices from the SuiteSparse Matrix Collection.

## Project Overview

The project aims to:
- Generate MLIR code for sparse matrix operations (multiplication, elementwise operations)
- Work with real-world sparse matrices from the SuiteSparse Collection
- Compare performance between different sparse matrix operations
- Provide tools for matrix exploration and analysis

## Repository Structure

```
SPMM_MLIR/
├── python/                         # Python tools and utilities
│   ├── new_code.py                 
│   └── real_matrices_2.py          
├── matrices/                       # Generated/processed matrices
├── matrices_downloaded/            # SuiteSparse matrices download directory
├── mlir_files/                     # Generated MLIR files for matrix multiplication
├── mlir_files_elementwise/         # Generated MLIR files for elementwise operations
├── mlir_files_sparse/              # Generated MLIR files for sparse-sparse operations
├── matrixmul/                      # Matrix multiplication results
├── elementwise/                    # Elementwise operation results
└── sparse_sparse/                  # Sparse-sparse operation results
```

## Getting Started

### Prerequisites

```bash
pip install numpy scipy ssgetpy argparse pathlib
```

### Required Dependencies

- **NumPy**: For dense matrix operations
- **SciPy**: For sparse matrix handling
- **ssgetpy**: For accessing SuiteSparse Matrix Collection
- **pathlib**: For file path management

## 🔧 Main Tools

### 1. MLIR Code Generator (`new_code.py`)

The main tool for generating MLIR code from SuiteSparse matrices.

#### Usage:
```bash
python python/new_code.py --matrix1 path/to/matrix1.mtx --matrix2 path/to/matrix2.mtx [options]
# or
python python/new_code.py -m1 matrix1.mtx -m2 matrix2.mtx --dense-cols 50
```

#### Options:
- `--matrix1, -m1`: Path to first sparse matrix (.mtx file) [required]
- `--matrix2, -m2`: Path to second sparse matrix (.mtx file) [required]
- `--dense-cols`: Number of columns for generated dense matrices (default: 50)
- `--output-dir`: Output directory for generated matrices (default: ../matrices)

#### What it does:
1. **Reads** MatrixMarket (.mtx) files from SuiteSparse
2. **Generates** dense matrices for various operations
3. **Creates** MLIR code for:
   - Sparse-dense matrix multiplication
   - Elementwise sparse-dense multiplication
   - Sparse-sparse matrix multiplication
4. **Saves** expected computation results for validation

### 2. Matrix Explorer (`real_matrices_2.py`)

Interactive tool for exploring and downloading matrices from SuiteSparse Collection.

#### Usage:
```bash
# Interactive mode
python python/real_matrices_2.py

# Find multiplicable matrices
python python/real_matrices_2.py --multiplicable [options]
```

#### Options for multiplicable search:
- `--min-sparsity`: Minimum sparsity required (default: 0.5)
- `--min-size`: Minimum matrix dimension (default: 200)
- `--max-size`: Maximum matrix dimension (default: 300)
- `--max-results`: Maximum number of pairs to show (default: 50)

#### Features:
- 🔍 Search matrices by ID, name, or group
- 📊 Display matrix properties (dimensions, sparsity, type)
- 🎯 Find compatible matrix pairs for multiplication
- 📥 Download matrices directly from SuiteSparse
- 📋 List available matrix types and groups

## Interactive Menu Options

When running `real_matrices_2.py` in interactive mode:

1. **Search matrix by ID** - Get detailed information about a specific matrix
2. **Search matrices by name** - Find matrices containing specific text
3. **Search matrices by group** - Browse matrices from specific collections
4. **List matrix types** - See all available matrix categories
5. **List matrix groups** - Browse all available matrix collections
6. **Find multiplicable matrices** - Discover compatible matrix pairs
7. **Exit** - Close the application

## Generated Files

### MLIR Files
- **Matrix Multiplication**: `mlir_files/matmul_<matrix>_dense.mlir`
- **Elementwise Operations**: `mlir_files_elementwise/elementwise_<matrix>_dense.mlir`
- **Sparse-Sparse Operations**: `mlir_files_sparse/sparse_sparse_<matrix1>_<matrix2>.mlir`

### Expected Results
- **Matrix Multiplication Sums**: `matrixmul/matmul_<matrix>_dense_sum.txt`
- **Elementwise Sums**: `elementwise/elementwise_<matrix>_dense_sum.txt`
- **Sparse-Sparse Sums**: `sparse_sparse/sparse_sparse_<matrix1>_<matrix2>_sum.txt`

### Matrix Files
- **Sparse Matrices**: `matrices/<matrix>.npz` (NumPy compressed format)
- **Dense Matrices**: `matrices/dense_for_<operation>_<matrix>.npy`

## 🔬 Example Workflow

1. **Explore available matrices**:
   ```bash
   python python/real_matrices_2.py --multiplicable --min-sparsity 0.7 --max-size 500
   ```

2. **Download a compatible pair**:
   Select a pair from the list and download them to `matrices_downloaded/`

3. **Generate MLIR code**:
   ```bash
   python python/new_code.py -m1 matrices_downloaded/matrix1/matrix1.mtx -m2 matrices_downloaded/matrix2/matrix2.mtx
   ```

4. **Results**: Check generated MLIR files and expected computation results in respective directories.

## Matrix Information

The tools provide detailed information about matrices:
- **Dimensions** (rows × columns)
- **Non-zero elements** (nnz)
- **Sparsity ratio** (percentage of zero elements)
- **Matrix type** (real/complex, symmetric/general, etc.)
- **Storage format** (coordinate, pattern, etc.)

## Matrix Compatibility

For matrix multiplication A × B:
- A.columns must equal B.rows
- The tool automatically attempts to transpose B if dimensions don't match
- Compatible pairs are automatically identified in the multiplicable search

## Advanced Features

### Matrix Market Format Support
- Reads standard MatrixMarket (.mtx) files
- Handles different value types (real, complex, pattern)
- Supports symmetric matrices (automatically expands)
- Converts to 0-based indexing for processing


## References

- [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [Matrix Market Format](https://math.nist.gov/MatrixMarket/formats.html)
