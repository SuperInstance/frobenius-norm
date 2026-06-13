# frobenius-norm: Matrix Norms and Inner Products

A linear algebra crate providing the **Frobenius norm**, **Frobenius inner product**, **Frobenius distance**, matrix multiplication, and transpose — all operating on row-major flat-storage matrices. Zero dependencies, fully tested.

## Why It Matters

The Frobenius norm is the most commonly used matrix norm in machine learning, signal processing, and numerical analysis. It measures the "size" of a matrix the same way the Euclidean (L2) norm measures the size of a vector. Applications include:

- **Loss functions**: Matrix factorization (PCA, NMF) minimizes ‖A − WH‖_F
- **Compression quality**: ‖A − A_k‖_F for rank-k SVD truncation
- **Convergence testing**: ‖X_{n+1} − X_n‖_F < ε
- **Differential privacy**: Sensitivity is measured in Frobenius norm
- **Quantum information**: The Hilbert–Schmidt norm equals the Frobenius norm for density matrices

## How It Works

### Frobenius Norm

For an m×n matrix A with entries a_{ij}:

$$\|A\|_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n} |a_{ij}|^2} = \sqrt{\text{trace}(A^* A)}$$

**Implementation**: Single pass over flat data, no intermediate allocation:

```rust
self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
```

**Complexity**: O(mn) time, O(1) extra space.

### Frobenius Inner Product

$$\langle A, B \rangle_F = \sum_{i,j} a_{ij} \cdot b_{ij} = \text{trace}(A^T B)$$

This reduces to a dot product on the flattened arrays. **Complexity**: O(mn).

### Frobenius Distance

$$d_F(A, B) = \|A - B\|_F = \sqrt{\sum_{i,j} (a_{ij} - b_{ij})^2}$$

**Complexity**: O(mn).

### Matrix Multiplication

Standard triple-nested loop: C = A × B where A is m×k, B is k×n, C is m×n.

$$c_{ij} = \sum_{l=1}^{k} a_{il} \cdot b_{lj}$$

**Complexity**: O(mkn) time, O(mn) space.

### Properties

The Frobenius norm is **sub-multiplicative**: ‖AB‖_F ≤ ‖A‖_F · ‖B‖_F. It is also invariant under orthogonal transformations: ‖UAV‖_F = ‖A‖_F for orthogonal U, V.

### Summary Table

| Operation | Time | Space |
|-----------|------|-------|
| `frobenius_norm()` | O(mn) | O(1) |
| `frobenius_norm_sq()` | O(mn) | O(1) |
| `frobenius_inner(&B)` | O(mn) | O(1) |
| `frobenius_distance(&B)` | O(mn) | O(1) |
| `matmul(&B)` | O(mkn) | O(mn) |
| `transpose()` | O(mn) | O(mn) |

## Quick Start

```rust
use frobenius_norm::Matrix;

let a = Matrix::from_2d(&[&[1.0, 2.0], &[3.0, 4.0]]);
let b = Matrix::zeros(2, 2);

// ||A||_F = sqrt(1 + 4 + 9 + 16) = sqrt(30)
assert!((a.frobenius_norm() - 30.0_f64.sqrt()).abs() < 1e-10);

// Distance from zero = ||A||_F
assert!((a.frobenius_distance(&b).unwrap() - 30.0_f64.sqrt()).abs() < 1e-10);

// Matrix multiply
let t = a.matmul(&a).unwrap();
// [[1,2],[3,4]]² = [[7,10],[15,22]]
assert!((t.get(0, 0) - 7.0).abs() < 1e-10);
```

## API

### `Matrix`

| Method | Signature | Description |
|--------|-----------|-------------|
| `zeros(rows, cols)` | `() -> Self` | Zero-initialized matrix |
| `from_2d(&[&[f64]])` | `() -> Self` | Construct from 2D slice |
| `get(i, j)` / `set(i, j, v)` | `(usize, usize) -> f64` | Element access |
| `frobenius_norm()` | `() -> f64` | ‖A‖_F |
| `frobenius_norm_sq()` | `() -> f64` | ‖A‖_F² (no sqrt) |
| `frobenius_inner(&B)` | `(&Matrix) -> Result<f64>` | ⟨A, B⟩_F |
| `frobenius_distance(&B)` | `(&Matrix) -> Result<f64>` | ‖A − B‖_F |
| `matmul(&B)` | `(&Matrix) -> Result<Matrix>` | A × B |
| `transpose()` | `() -> Matrix` | Aᵀ |

## Architecture Notes

This is a **γ (gamma)** module — pure mathematical operations on a flat `Vec<f64>`. The flat storage layout (row-major) is cache-friendly and interoperates with BLAS/LAPACK conventions. In the γ + η = C framework, this provides the normative measurement layer; **η** orchestration can build optimization loops (gradient descent on Frobenius loss, alternating minimization) on top.

## References

- Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins. §2.3.
- Horn, R. A. & Johnson, C. R. (2012). *Matrix Analysis* (2nd ed.). Cambridge. Chapter 5.
- Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.

## License

MIT
