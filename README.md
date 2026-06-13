# Frobenius Norm & Matrix Operations

**A Rust library for matrix computations centered on the Frobenius norm** — `||A||_F = √(Σ|aᵢⱼ|²)` — with additional operations including matrix multiplication, transpose, Frobenius inner product, and Frobenius distance.

## Why It Matters

The Frobenius norm is the most common matrix norm in numerical computing. It measures the "size" of a matrix as if it were a flat vector, making it ideal for: measuring approximation error in low-rank decompositions (SVD, PCA), computing convergence criteria in iterative solvers, defining matrix regularization in machine learning (weight decay), and quantifying distance between weight matrices in neural network training. The Frobenius inner product `<A, B> = Σ aᵢⱼbᵢⱼ` generalizes the dot product to matrices and underlies matrix factorization objective functions.

## How It Works

Matrices are stored in row-major order as a flat `Vec<f64>` alongside `rows` and `cols` dimensions. The Frobenius norm computes `√(Σ x²)` over all elements — **O(mn)** for an `m×n` matrix. The squared variant `frobenius_norm_sq()` skips the `sqrt` call, which is useful when comparing norms (the ordering is preserved) or when the squared value appears in objective functions.

Matrix multiplication uses the standard triple-nested loop (ikj order) — **O(mnp)** for `m×n` times `n×p`. The Frobenius inner product takes the element-wise product and sums — **O(mn)**. Distance between two matrices `||A - B||_F` computes `(a-b)²` per element then takes the square root — **O(mn)**.

## Quick Start

```rust
use frobenius_norm::Matrix;

fn main() {
    // Create a matrix
    let a = Matrix::from_2d(&[
        &[1.0, 2.0],
        &[3.0, 4.0],
    ]);

    // Frobenius norm: sqrt(1 + 4 + 9 + 16) = sqrt(30)
    println!("||A||_F = {:.4}", a.frobenius_norm()); // 5.4772

    // Matrix multiplication
    let b = Matrix::from_2d(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let c = a.matmul(&b).unwrap();
    println!("A · I = A, norm = {:.4}", c.frobenius_norm());

    // Frobenius distance between two matrices
    let zeros = Matrix::zeros(2, 2);
    let dist = a.frobenius_distance(&zeros).unwrap();
    println!("||A - 0||_F = {:.4}", dist); // Same as ||A||_F

    // Frobenius inner product
    let ip = a.frobenius_inner(&b).unwrap();
    println!("<A, I> = {:.4}", ip); // trace = 1 + 4 = 5
}
```

## API

| Method | Complexity | Description |
|---|---|---|
| `Matrix::zeros(rows, cols)` | **O(mn)** | Create zero matrix |
| `Matrix::from_2d(&[&[f64]])` | **O(mn)** | Create from 2D slice |
| `frobenius_norm()` | **O(mn)** | `√(Σ|aᵢⱼ|²)` |
| `frobenius_norm_sq()` | **O(mn)** | `Σ|aᵢⱼ|²` (no sqrt) |
| `matmul(other)` | **O(mnp)** | Standard matrix multiplication |
| `transpose()` | **O(mn)** | Transpose |
| `frobenius_inner(other)` | **O(mn)** | Element-wise product sum |
| `frobenius_distance(other)` | **O(mn)** | `||A - B||_F` |

## Architecture Notes

Part of the SuperInstance linear algebra foundation. Companion crates include `grassmannian` (subspace geometry) and `fisher-information` (statistical matrices). See the [Architecture Guide](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## License

MIT
