import numpy as np
import scipy.sparse as sp
import os
import random
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import math

class MatrixMarketReader:
    """Class for reading SuiteSparse MatrixMarket files."""
    
    def __init__(self):
        pass
    
    def read_matrix_market(self, filepath: str) -> sp.csr_matrix:
        """
        Read a MatrixMarket file and return a CSR sparse matrix.
        
        Args:
            filepath: Path to the .mtx file
            
        Returns:
            A scipy CSR sparse matrix
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Matrix file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            # Read header
            header = f.readline().strip()
            if not header.startswith('%%MatrixMarket'):
                raise ValueError(f"Invalid MatrixMarket header: {header}")
            
            # Parse header components
            header_parts = header.split()
            if len(header_parts) < 4:
                raise ValueError(f"Invalid MatrixMarket header format: {header}")
            
            matrix_type = header_parts[2]  # coordinate
            value_type = header_parts[3] if len(header_parts) > 3 else 'real'
            symmetry = header_parts[4] if len(header_parts) > 4 else 'general'
            
            # Skip comment lines
            while True:
                line = f.readline().strip()
                if not line.startswith('%'):
                    break
            
            # Parse dimensions
            dims = line.split()
            if len(dims) < 3:
                raise ValueError(f"Invalid dimension line: {line}")
            
            rows, cols, nnz = map(int, dims[:3])
            
            # Read matrix data
            row_indices = []
            col_indices = []
            values = []
            
            for i in range(nnz):
                line = f.readline().strip()
                if not line:
                    break
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                row = int(parts[0]) - 1  # Convert to 0-based indexing
                col = int(parts[1]) - 1  # Convert to 0-based indexing
                
                # Handle different value types
                if value_type.lower() == 'pattern':
                    value = 1.0  # Pattern matrices have implicit 1.0 values
                elif len(parts) > 2:
                    value = float(parts[2])
                else:
                    value = 1.0
                
                row_indices.append(row)
                col_indices.append(col)
                values.append(value)
                
                # Handle symmetric matrices
                if symmetry.lower() == 'symmetric' and row != col:
                    row_indices.append(col)
                    col_indices.append(row)
                    values.append(value)
        
        # Create and return CSR matrix
        coo_matrix = sp.coo_matrix((values, (row_indices, col_indices)), 
                                   shape=(rows, cols))
        return coo_matrix.tocsr()
    
    def get_matrix_info(self, matrix: sp.csr_matrix) -> dict:
        """Get information about a sparse matrix."""
        return {
            'shape': matrix.shape,
            'nnz': matrix.nnz,
            'sparsity': 1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
            'density': matrix.nnz / (matrix.shape[0] * matrix.shape[1])
        }

class MatrixGenerator:
    """Class for generating and handling sparse and dense matrices."""

    def __init__(self, output_dir: str = "../matrices"):
        """
        Initialize the matrix generator.
        Args:
            output_dir: Directory to store generated matrices
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dense_matrix(self, rows: int, cols: int) -> np.ndarray:
        """
        Generate a dense matrix with random values.
        Args:
            rows: Number of rows
            cols: Number of columns
        Returns:
            A NumPy dense matrix
        """
        return np.random.rand(rows, cols) * 10

    def save_sparse_matrix(self, matrix: sp.csr_matrix, filename: str) -> None:
        """
        Save a sparse matrix to file.
        Args:
            matrix: Sparse matrix to save
            filename: Target filename
        """
        filepath = self.output_dir / filename
        np.savez(filepath,
                 matrix_data=matrix.data,
                 matrix_indices=matrix.indices,
                 matrix_indptr=matrix.indptr,
                 matrix_shape=matrix.shape)

    def save_dense_matrix(self, matrix: np.ndarray, filename: str) -> None:
        """
        Save a dense matrix to file.
        Args:
            matrix: Dense matrix to file
            filename: Target filename
        """
        filepath = self.output_dir / filename
        np.save(filepath, matrix)

    @staticmethod
    def load_sparse_matrix(filepath: str) -> sp.csr_matrix:
        """
        Load a sparse matrix from file.
        Args:
            filepath: Path to the saved matrix
        Returns:
            The loaded sparse matrix
        """
        loaded = np.load(filepath)
        return sp.csr_matrix((loaded['matrix_data'], loaded['matrix_indices'],
                            loaded['matrix_indptr']), shape=loaded['matrix_shape'])

class MlirGenerator:
    """Class for generating MLIR code for various matrix operations."""

    def __init__(self):
        """
        Initialize the MLIR generator.
        """
        self.output_dirs = {
            'matmul': Path("../mlir_files"),
            'elementwise': Path("../mlir_files_elementwise"),
            'sparse': Path("../mlir_files_sparse")
        }
        for dir in self.output_dirs.values():
            dir.mkdir(parents=True, exist_ok=True)

    def _format_float_literal(self, val: float) -> str:
        """
        Format a float number as a string literal suitable for MLIR.
        """
        if math.isnan(val):
            return '0.0 / 0.0'
        elif math.isinf(val):
            return '1.0 / 0.0' if val > 0 else '-1.0 / 0.0'
        else:
            if abs(val - round(val)) < 1e-9:
                return f"{int(round(val))}.0"
            else:
                return f"{val:.6f}"

    def _format_dense_matrix(self, matrix: np.ndarray) -> str:
        """
        Format a dense matrix to a string suitable for MLIR.
        """
        rows = []
        for row in matrix.tolist():
            formatted_row = [self._format_float_literal(val) for val in row]
            rows.append(f"[{', '.join(formatted_row)}]")
        return f"[{', '.join(rows)}]"
    
    def _get_csr_components(self, sparse_matrix: sp.csr_matrix):
        """
        Extract CSR components with correct naming and explicit row indices.
        """
        values = [self._format_float_literal(v) for v in sparse_matrix.data.tolist()]
        col_indices = sparse_matrix.indices.tolist()  # CORRECT: these are column indices
        row_pointers = sparse_matrix.indptr.tolist()  # CORRECT: these are row pointers
        
        # Generate explicit row indices for each non-zero element
        row_indices = []
        for row_idx in range(len(row_pointers) - 1):
            start = row_pointers[row_idx]
            end = row_pointers[row_idx + 1]
            row_indices.extend([row_idx] * (end - start))
        
        return values, col_indices, row_pointers, row_indices
    
    def generate_matmul_mlir(self, sparse_matrix: sp.csr_matrix, 
                       dense_matrix: np.ndarray, matrix_name1: str, matrix_name2: str) -> str:
        """Generate MLIR for sparse-dense matrix multiplication."""
        m, k = sparse_matrix.shape
        k2, n = dense_matrix.shape

        if k != k2:
            raise ValueError(f"Matrix dimensions don't match for multiplication: sparse({m}x{k}) * dense({k2}x{n})")

        values, col_indices, row_pointers, row_indices = self._get_csr_components(sparse_matrix)

        expected_result = sparse_matrix.toarray() @ dense_matrix
        expected_sum = np.sum(expected_result)
        dense_data_str = self._format_dense_matrix(dense_matrix)
        expected_sum_str = self._format_float_literal(expected_sum)

        return f"""// Sparse-Dense Matrix Multiplication: {matrix_name1} * {matrix_name2}
#CSR = #sparse_tensor.encoding<{{
    map = (d0, d1) -> (d0: dense, d1: compressed)
}}>

module {{
    func.func @compute_sum(%tensor: tensor<{m}x{n}xf64>) -> f64 {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %rows = arith.constant {m} : index
        %cols = arith.constant {n} : index
        %init = arith.constant 0.0 : f64

        %sum = scf.for %i = %c0 to %rows step %c1 iter_args(%sum_iter = %init) -> (f64) {{
            %inner_sum = scf.for %j = %c0 to %cols step %c1 iter_args(%inner_sum_iter = %sum_iter) -> (f64) {{
                %elem = tensor.extract %tensor[%i, %j] : tensor<{m}x{n}xf64>
                %new_sum = arith.addf %inner_sum_iter, %elem : f64
                scf.yield %new_sum : f64
            }}
            scf.yield %inner_sum : f64
        }}
        return %sum : f64
    }}

    func.func @sparse_dense_matmul(
        %sparse : tensor<{m}x{k}xf64, #CSR>,
        %dense : tensor<{k}x{n}xf64>,
        %init : tensor<{m}x{n}xf64>
    ) -> tensor<{m}x{n}xf64> {{
        %result = linalg.matmul
            ins(%sparse, %dense: tensor<{m}x{k}xf64, #CSR>, tensor<{k}x{n}xf64>)
            outs(%init: tensor<{m}x{n}xf64>) -> tensor<{m}x{n}xf64>
        return %result : tensor<{m}x{n}xf64>
    }}

    func.func @main() -> i32 {{
        %1 = arith.constant 0 : i32  
        %2 = arith.constant 1 : i32
        %output = tensor.empty() : tensor<{m}x{n}xf64>
        %sparse_tensor = call @assemble_sparse_tensor() : () -> tensor<{m}x{k}xf64, #CSR>
        %dense_tensor = arith.constant dense<{dense_data_str}> : tensor<{k}x{n}xf64>

        %computed_result = call @sparse_dense_matmul(%sparse_tensor, %dense_tensor, %output) : 
            (tensor<{m}x{k}xf64, #CSR>, tensor<{k}x{n}xf64>, tensor<{m}x{n}xf64>) -> tensor<{m}x{n}xf64>

        %expected_sum = arith.constant {expected_sum_str} : f64
        %sum = call @compute_sum(%computed_result) : (tensor<{m}x{n}xf64>) -> f64

        %is_equal = arith.cmpf oeq, %sum, %expected_sum : f64

        %result = arith.select %is_equal, %1, %2 : i32            
        return %result : i32
    }}

    func.func private @assemble_sparse_tensor() -> tensor<{m}x{k}xf64, #CSR> {{
        %values = arith.constant dense<[{', '.join(values)}]> : tensor<{len(values)}xf64>
        %col_indices = arith.constant dense<[{', '.join(map(str, col_indices))}]> : tensor<{len(col_indices)}xindex>
        %row_pointers = arith.constant dense<[{', '.join(map(str, row_pointers))}]> : tensor<{len(row_pointers)}xindex>

        %sparse_tensor = sparse_tensor.assemble (%row_pointers, %col_indices), %values
            : (tensor<{len(row_pointers)}xindex>, tensor<{len(col_indices)}xindex>), tensor<{len(values)}xf64> 
            to tensor<{m}x{k}xf64, #CSR>
        return %sparse_tensor : tensor<{m}x{k}xf64, #CSR>
    }}
}}"""

    def generate_elementwise_mlir(self, sparse_matrix: sp.csr_matrix, 
                            dense_matrix: np.ndarray, matrix_name1: str, matrix_name2: str) -> str:
        """Generate MLIR for elementwise multiplication."""
        m, k = sparse_matrix.shape
        k2, n = dense_matrix.shape

        if sparse_matrix.shape != dense_matrix.shape:
            raise ValueError(f"Matrix dimensions don't match for elementwise operation: sparse{sparse_matrix.shape} != dense{dense_matrix.shape}")

        values, col_indices, row_pointers, row_indices = self._get_csr_components(sparse_matrix)

        expected_result = sparse_matrix.toarray() * dense_matrix
        expected_sum = np.sum(expected_result)
        dense_data_str = self._format_dense_matrix(dense_matrix)
        expected_sum_str = self._format_float_literal(expected_sum)

        return f"""// Elementwise Multiplication: {matrix_name1} .* {matrix_name2}
#CSR = #sparse_tensor.encoding<{{
    map = (d0, d1) -> (d0: dense, d1: compressed)
}}>

#trait_mul_ds = {{
    indexing_maps = [
        affine_map<(i,j) -> (i,j)>,  // A
        affine_map<(i,j) -> (i,j)>,  // B
        affine_map<(i,j) -> (i,j)>   // C (out)
    ],
    iterator_types = ["parallel", "parallel"],
    doc = "C(i,j) = A(i,j) * B(i,j)"
}}

module {{
    func.func @compute_sum(%tensor: tensor<{m}x{n}xf64>) -> f64 {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %rows = arith.constant {m} : index
        %cols = arith.constant {n} : index
        %init = arith.constant 0.0 : f64

        %sum = scf.for %i = %c0 to %rows step %c1 iter_args(%sum_iter = %init) -> (f64) {{
            %inner_sum = scf.for %j = %c0 to %cols step %c1 iter_args(%inner_sum_iter = %sum_iter) -> (f64) {{
                %elem = tensor.extract %tensor[%i, %j] : tensor<{m}x{n}xf64>
                %new_sum = arith.addf %inner_sum_iter, %elem : f64
                scf.yield %new_sum : f64
            }}
            scf.yield %inner_sum : f64
        }}
        return %sum : f64
    }}

    func.func @mul_ds(
        %sparse : tensor<{m}x{k}xf64, #CSR>,
        %dense : tensor<{k}x{n}xf64>,
        %init : tensor<{m}x{n}xf64>
    ) -> tensor<{m}x{n}xf64> {{
        %0 = linalg.generic #trait_mul_ds
            ins(%sparse, %dense: tensor<{m}x{k}xf64, #CSR>, tensor<{k}x{n}xf64>)
            outs(%init: tensor<{m}x{n}xf64>) {{
            ^bb0(%a: f64, %b: f64, %x: f64):
                %0 = arith.mulf %a, %b : f64
                linalg.yield %0 : f64
        }} -> tensor<{m}x{n}xf64>
        return %0 : tensor<{m}x{n}xf64>
    }}

    func.func @main() -> i32 {{
        %1 = arith.constant 0 : i32  
        %2 = arith.constant 1 : i32
        %output = tensor.empty() : tensor<{m}x{n}xf64>
        %sparse_tensor = call @assemble_sparse_tensor() : () -> tensor<{m}x{k}xf64, #CSR>
        %dense_tensor = arith.constant dense<{dense_data_str}> : tensor<{k}x{n}xf64>

        %computed_result = call @mul_ds(%sparse_tensor, %dense_tensor, %output) : 
            (tensor<{m}x{k}xf64, #CSR>, tensor<{k}x{n}xf64>, tensor<{m}x{n}xf64>) -> tensor<{m}x{n}xf64>

        %expected_sum = arith.constant {expected_sum_str} : f64
        %sum = call @compute_sum(%computed_result) : (tensor<{m}x{n}xf64>) -> f64

        %is_equal = arith.cmpf oeq, %sum, %expected_sum : f64
        %result = arith.select %is_equal, %1, %2 : i32            
        return %result : i32
    }}

    func.func private @assemble_sparse_tensor() -> tensor<{m}x{k}xf64, #CSR> {{
        %values = arith.constant dense<[{', '.join(values)}]> : tensor<{len(values)}xf64>
        %col_indices = arith.constant dense<[{', '.join(map(str, col_indices))}]> : tensor<{len(col_indices)}xindex>
        %row_pointers = arith.constant dense<[{', '.join(map(str, row_pointers))}]> : tensor<{len(row_pointers)}xindex>

        %sparse_tensor = sparse_tensor.assemble (%row_pointers, %col_indices), %values
            : (tensor<{len(row_pointers)}xindex>, tensor<{len(col_indices)}xindex>), tensor<{len(values)}xf64> 
            to tensor<{m}x{k}xf64, #CSR>
        return %sparse_tensor : tensor<{m}x{k}xf64, #CSR>
    }}
}}"""

    def generate_sparse_sparse_mlir(self, sparse_matrix1: sp.csr_matrix, 
                                sparse_matrix2: sp.csr_matrix, 
                                matrix_name1: str, matrix_name2: str) -> str:
        """Generate MLIR for sparse-sparse matrix multiplication."""
        m, k = sparse_matrix1.shape
        k2, n = sparse_matrix2.shape

        if k != k2:
            raise ValueError(f"Matrix dimensions don't match for multiplication: sparse1({m}x{k}) * sparse2({k2}x{n})")

        # Format both sparse matrices
        values1, col_indices1, row_pointers1, row_indices1 = self._get_csr_components(sparse_matrix1)
        values2, col_indices2, row_pointers2, row_indices2 = self._get_csr_components(sparse_matrix2)

        expected_result = sparse_matrix1.toarray() @ sparse_matrix2.toarray()
        expected_sum = np.sum(expected_result)
        expected_sum_str = self._format_float_literal(expected_sum)

        return f"""// Sparse-Sparse Matrix Multiplication: {matrix_name1} * {matrix_name2}
#CSR = #sparse_tensor.encoding<{{
    map = (d0, d1) -> (d0: dense, d1: compressed)
}}>

module {{
    func.func @compute_sum(%tensor: tensor<{m}x{n}xf64>) -> f64 {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %rows = arith.constant {m} : index
        %cols = arith.constant {n} : index
        %init = arith.constant 0.0 : f64

        %sum = scf.for %i = %c0 to %rows step %c1 iter_args(%sum_iter = %init) -> (f64) {{
            %inner_sum = scf.for %j = %c0 to %cols step %c1 iter_args(%inner_sum_iter = %sum_iter) -> (f64) {{
                %elem = tensor.extract %tensor[%i, %j] : tensor<{m}x{n}xf64>
                %new_sum = arith.addf %inner_sum_iter, %elem : f64
                scf.yield %new_sum : f64
            }}
            scf.yield %inner_sum : f64
        }}
        return %sum : f64
    }}

    func.func @sparse_sparse_matmul(
        %sparse1 : tensor<{m}x{k}xf64, #CSR>,
        %sparse2 : tensor<{k}x{n}xf64, #CSR>,
        %init : tensor<{m}x{n}xf64>
    ) -> tensor<{m}x{n}xf64> {{
        %result = linalg.matmul
            ins(%sparse1, %sparse2: tensor<{m}x{k}xf64, #CSR>, tensor<{k}x{n}xf64, #CSR>)
            outs(%init: tensor<{m}x{n}xf64>) -> tensor<{m}x{n}xf64>
        return %result : tensor<{m}x{n}xf64>
    }}

    func.func @main() -> i32 {{
        %1 = arith.constant 0 : i32  
        %2 = arith.constant 1 : i32
        %output = tensor.empty() : tensor<{m}x{n}xf64>
        %sparse_tensor1 = call @assemble_sparse_tensor1() : () -> tensor<{m}x{k}xf64, #CSR>
        %sparse_tensor2 = call @assemble_sparse_tensor2() : () -> tensor<{k}x{n}xf64, #CSR>

        %computed_result = call @sparse_sparse_matmul(%sparse_tensor1, %sparse_tensor2, %output) : 
            (tensor<{m}x{k}xf64, #CSR>, tensor<{k}x{n}xf64, #CSR>, tensor<{m}x{n}xf64>) -> tensor<{m}x{n}xf64>

        %expected_sum = arith.constant {expected_sum_str} : f64
        %sum = call @compute_sum(%computed_result) : (tensor<{m}x{n}xf64>) -> f64

        %is_equal = arith.cmpf oeq, %sum, %expected_sum : f64
        %result = arith.select %is_equal, %1, %2 : i32            
        return %result : i32
    }}

    func.func private @assemble_sparse_tensor1() -> tensor<{m}x{k}xf64, #CSR> {{
        %values = arith.constant dense<[{', '.join(values1)}]> : tensor<{len(values1)}xf64>
        %col_indices = arith.constant dense<[{', '.join(map(str, col_indices1))}]> : tensor<{len(col_indices1)}xindex>
        %row_pointers = arith.constant dense<[{', '.join(map(str, row_pointers1))}]> : tensor<{len(row_pointers1)}xindex>

        %sparse_tensor = sparse_tensor.assemble (%row_pointers, %col_indices), %values
            : (tensor<{len(row_pointers1)}xindex>, tensor<{len(col_indices1)}xindex>), tensor<{len(values1)}xf64> 
            to tensor<{m}x{k}xf64, #CSR>
        return %sparse_tensor : tensor<{m}x{k}xf64, #CSR>
    }}

    func.func private @assemble_sparse_tensor2() -> tensor<{k}x{n}xf64, #CSR> {{
        %values = arith.constant dense<[{', '.join(values2)}]> : tensor<{len(values2)}xf64>
        %col_indices = arith.constant dense<[{', '.join(map(str, col_indices2))}]> : tensor<{len(col_indices2)}xindex>
        %row_pointers = arith.constant dense<[{', '.join(map(str, row_pointers2))}]> : tensor<{len(row_pointers2)}xindex>

        %sparse_tensor = sparse_tensor.assemble (%row_pointers, %col_indices), %values
            : (tensor<{len(row_pointers2)}xindex>, tensor<{len(col_indices2)}xindex>), tensor<{len(values2)}xf64> 
            to tensor<{k}x{n}xf64, #CSR>
        return %sparse_tensor : tensor<{k}x{n}xf64, #CSR>
    }}
}}"""

    def save_mlir(self, mlir_content: str, filename: str, operation_type: str) -> None:
        """
        Save MLIR code to file in the appropriate directory.
        Args:
            mlir_content: MLIR code as string
            filename: Target filename
            operation_type: Type of operation ('matmul', 'elementwise', 'sparse')
        """
        filepath = self.output_dirs[operation_type] / filename
        with open(filepath, 'w') as f:
            f.write(mlir_content)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate MLIR code for SuiteSparse matrix operations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python script.py --matrix1 path/to/matrix1.mtx --matrix2 path/to/matrix2.mtx
  python script.py -m1 matrix1.mtx -m2 matrix2.mtx --dense-cols 50
        '''
    )
    
    parser.add_argument('--matrix1', '-m1', required=True,
                        help='Path to first sparse matrix (.mtx file)')
    parser.add_argument('--matrix2', '-m2', required=True,
                        help='Path to second sparse matrix (.mtx file)')
    parser.add_argument('--dense-cols', type=int, default=50,
                        help='Number of columns for generated dense matrices (default: 50)')
    parser.add_argument('--output-dir', default='../matrices',
                        help='Output directory for generated matrices (default: ../matrices)')
    
    return parser.parse_args()

def get_matrix_name(filepath: str) -> str:
    """Extract matrix name from filepath."""
    return Path(filepath).stem

def main():
    """Main function to generate matrices and MLIR code from SuiteSparse files."""
    args = parse_arguments()
    
    # Create output directories
    Path("../matrixmul").mkdir(parents=True, exist_ok=True)
    Path("../elementwise").mkdir(parents=True, exist_ok=True)
    Path("../sparse_sparse").mkdir(parents=True, exist_ok=True)

    # Initialize classes
    matrix_reader = MatrixMarketReader()
    matrix_generator = MatrixGenerator(args.output_dir)
    mlir_generator = MlirGenerator()
    
    try:
        print(f"Reading matrix 1 from: {args.matrix1}")
        sparse_matrix1 = matrix_reader.read_matrix_market(args.matrix1)
        matrix1_info = matrix_reader.get_matrix_info(sparse_matrix1)
        matrix1_name = get_matrix_name(args.matrix1)
        
        print(f"Matrix 1 ({matrix1_name}): {matrix1_info['shape']}, "
              f"nnz={matrix1_info['nnz']}, sparsity={matrix1_info['sparsity']:.3f}")
        
        print(f"Reading matrix 2 from: {args.matrix2}")
        sparse_matrix2 = matrix_reader.read_matrix_market(args.matrix2)
        matrix2_info = matrix_reader.get_matrix_info(sparse_matrix2)
        matrix2_name = get_matrix_name(args.matrix2)
        
        print(f"Matrix 2 ({matrix2_name}): {matrix2_info['shape']}, "
              f"nnz={matrix2_info['nnz']}, sparsity={matrix2_info['sparsity']:.3f}")
        
        # Save the sparse matrices
        matrix_generator.save_sparse_matrix(sparse_matrix1, f"{matrix1_name}.npz")
        matrix_generator.save_sparse_matrix(sparse_matrix2, f"{matrix2_name}.npz")
        
        # Generate dense matrices for different operations
        m1, k1 = sparse_matrix1.shape
        m2, k2 = sparse_matrix2.shape
        
        # For sparse-dense matmul: matrix1 (m1 x k1) * dense (k1 x dense_cols)
        dense_matrix_matmul = matrix_generator.generate_dense_matrix(k1, args.dense_cols)
        matrix_generator.save_dense_matrix(dense_matrix_matmul, f"dense_for_matmul_{matrix1_name}.npy")
        
        # For elementwise: dense matrix with same shape as matrix1
        dense_matrix_elementwise = matrix_generator.generate_dense_matrix(m1, k1)
        matrix_generator.save_dense_matrix(dense_matrix_elementwise, f"dense_for_elementwise_{matrix1_name}.npy")
        
        print("\n=== Generating MLIR Files ===")
        
        # 1. Sparse-Dense Matrix Multiplication
        print(f"Generating sparse-dense matmul: {matrix1_name} * dense...")
        try:
            matmul_mlir = mlir_generator.generate_matmul_mlir(
                sparse_matrix1, dense_matrix_matmul, matrix1_name, "dense")
            mlir_filename = f"matmul_{matrix1_name}_dense.mlir"
            mlir_generator.save_mlir(matmul_mlir, mlir_filename, 'matmul')
            
            # Save expected result sum
            expected_matmul = sparse_matrix1.toarray() @ dense_matrix_matmul
            expected_matmul_sum = np.sum(expected_matmul)
            with open(f"../matrixmul/matmul_{matrix1_name}_dense_sum.txt", 'w') as f:
                f.write(str(expected_matmul_sum))
            
            print(f"  ✓ Generated {mlir_filename}")
            print(f"  ✓ Expected sum: {expected_matmul_sum}")
            
        except Exception as e:
            print(f"  ✗ Error in sparse-dense matmul: {e}")
        
        # 2. Elementwise Multiplication
        print(f"Generating elementwise multiplication: {matrix1_name} .* dense...")
        try:
            elementwise_mlir = mlir_generator.generate_elementwise_mlir(
                sparse_matrix1, dense_matrix_elementwise, matrix1_name, "dense")
            mlir_filename = f"elementwise_{matrix1_name}_dense.mlir"
            mlir_generator.save_mlir(elementwise_mlir, mlir_filename, 'elementwise')
            
            # Save expected result sum
            expected_elementwise = sparse_matrix1.toarray() * dense_matrix_elementwise
            expected_elementwise_sum = np.sum(expected_elementwise)
            with open(f"../elementwise/elementwise_{matrix1_name}_dense_sum.txt", 'w') as f:
                f.write(str(expected_elementwise_sum))
            
            print(f"  ✓ Generated {mlir_filename}")
            print(f"  ✓ Expected sum: {expected_elementwise_sum}")
            
        except Exception as e:
            print(f"  ✗ Error in elementwise multiplication: {e}")
        
        # 3. Sparse-Sparse Matrix Multiplication
        print(f"Generating sparse-sparse matmul: {matrix1_name} * {matrix2_name}...")
        try:
            # Check dimension compatibility
            if k1 != m2:
                print(f"  ! Matrices not compatible for multiplication: {matrix1_name}({m1}x{k1}) * {matrix2_name}({m2}x{k2})")
                print("  ! Attempting to transpose matrix2...")
                sparse_matrix2_T = sparse_matrix2.T
                m2_T, k2_T = sparse_matrix2_T.shape
                if k1 == m2_T:
                    print(f"  ✓ Using transposed matrix2: {matrix1_name}({m1}x{k1}) * {matrix2_name}^T({m2_T}x{k2_T})")
                    sparse_matrix2 = sparse_matrix2_T
                    matrix2_name = f"{matrix2_name}_T"
                else:
                    raise ValueError(f"Even after transpose, matrices are not compatible: {k1} != {m2_T}")
            
            sparse_sparse_mlir = mlir_generator.generate_sparse_sparse_mlir(
                sparse_matrix1, sparse_matrix2, matrix1_name, matrix2_name)
            mlir_filename = f"sparse_sparse_{matrix1_name}_{matrix2_name}.mlir"
            mlir_generator.save_mlir(sparse_sparse_mlir, mlir_filename, 'sparse')
            
            # Save expected result sum
            expected_sparse_sparse = sparse_matrix1.toarray() @ sparse_matrix2.toarray()
            expected_sparse_sparse_sum = np.sum(expected_sparse_sparse)
            with open(f"../sparse_sparse/sparse_sparse_{matrix1_name}_{matrix2_name}_sum.txt", 'w') as f:
                f.write(str(expected_sparse_sparse_sum))
            
            print(f"  ✓ Generated {mlir_filename}")
            print(f"  ✓ Expected sum: {expected_sparse_sparse_sum}")
            
        except Exception as e:
            print(f"  ✗ Error in sparse-sparse matmul: {e}")
        
        print("\n=== Summary ===")
        print(f"Matrix 1: {matrix1_name} {matrix1_info['shape']} (sparsity: {matrix1_info['sparsity']:.3f})")
        print(f"Matrix 2: {matrix2_name} {matrix2_info['shape']} (sparsity: {matrix2_info['sparsity']:.3f})")
        print(f"Dense matrix for matmul: ({k1}, {args.dense_cols})")
        print(f"Dense matrix for elementwise: ({m1}, {k1})")
        print("Generated MLIR files in:")
        print(f"  - Matrix multiplication: ../mlir_files/")
        print(f"  - Elementwise operations: ../mlir_files_elementwise/")
        print(f"  - Sparse-sparse operations: ../mlir_files_sparse/")
        print("Generated matrices saved in:", args.output_dir)
        print("Expected computation results saved in respective directories.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the matrix files exist and paths are correct.")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    print("\nGeneration complete!")
    return 0

if __name__ == "__main__":
    exit(main())