//! Frobenius norm computation for matrices.
//!
//! Provides efficient Frobenius norm: ||A||_F = sqrt(Σ|aᵢⱼ|²)

/// A simple row-major matrix backed by a flat vector.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Create a new matrix with given dimensions, filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create from a 2D slice.
    pub fn from_2d(data: &[ &[f64]]) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        let flat: Vec<f64> = data.iter().flat_map(|r| r.iter().copied()).collect();
        Self { rows, cols, data: flat }
    }

    /// Get element at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    /// Set element at (i, j).
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.data[i * self.cols + j] = val;
    }

    /// Compute the Frobenius norm: sqrt(sum of squares of all elements).
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Compute the squared Frobenius norm (avoids sqrt).
    pub fn frobenius_norm_sq(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum()
    }

    /// Matrix transpose.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Matrix multiplication C = A * B.
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        if self.cols != other.rows {
            return Err("Dimension mismatch for matrix multiplication");
        }
        let mut result = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        Ok(result)
    }

    /// Frobenius inner product: <A, B> = Σ aᵢⱼ·bᵢⱼ
    pub fn frobenius_inner(&self, other: &Matrix) -> Result<f64, &'static str> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Dimension mismatch for Frobenius inner product");
        }
        Ok(self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum())
    }

    /// Frobenius distance between two matrices: ||A - B||_F
    pub fn frobenius_distance(&self, other: &Matrix) -> Result<f64, &'static str> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Dimension mismatch for Frobenius distance");
        }
        let sq_dist: f64 = self.data.iter().zip(&other.data)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        Ok(sq_dist.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_norm() {
        let mut m = Matrix::zeros(3, 3);
        for i in 0..3 { m.set(i, i, 1.0); }
        assert!((m.frobenius_norm() - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_known_matrix() {
        let m = Matrix::from_2d(&[&[1.0, 2.0], &[3.0, 4.0]]);
        // ||A||_F = sqrt(1+4+9+16) = sqrt(30)
        assert!((m.frobenius_norm() - 30.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_distance() {
        let a = Matrix::from_2d(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let b = Matrix::zeros(2, 2);
        assert!((a.frobenius_distance(&b).unwrap() - 2.0_f64.sqrt()).abs() < 1e-10);
    }
}
