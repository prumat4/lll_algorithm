extern crate nalgebra as na;
use na::{DMatrix, DVector, MatrixXx6};
use std::f64;

fn gram_schmidt(basis: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
    let m = basis.ncols();
    let n = basis.nrows();
    let mut ortho = DMatrix::zeros(n, m);
    let mut mu = DMatrix::zeros(m, m);

    for i in 0..m {
        ortho.set_column(i, &basis.column(i));

        for j in 0..i {
            let proj = ortho.column(j).dot(&basis.column(i)) / ortho.column(j).dot(&ortho.column(j));
            mu[(i, j)] = proj;
            ortho.set_column(i, &(ortho.column(i) - proj * ortho.column(j)));
        }
    }

    (ortho, mu)
}

fn lll(basis: &DMatrix<f64>, delta: f64) -> DMatrix<f64> {
    let (m, n) = (basis.nrows(), basis.ncols());
    let mut b = basis.clone();
    let mut k = 1;

    while k < n {
        for j in (0..k).rev() {
            let mu = b.column(k).dot(&b.column(j)) / b.column(j).dot(&b.column(j));
            if mu.abs() > 0.5 {
                b.set_column(k, &(b.column(k) - mu.round() * b.column(j)));
            }
        }

        let (ortho, mu) = gram_schmidt(&b);

        if ortho.column(k).norm_squared() >= (delta - mu[(k, k - 1)].powi(2)) * ortho.column(k - 1).norm_squared() {
            k += 1;
        } else {
            b.swap_columns(k, k - 1);
            k = std::cmp::max(k - 1, 1);
        }
    }

    b
}

fn hadamard_ratio(basis: &DMatrix<f64>) -> f64 {
    let (ortho, _) = gram_schmidt(basis);
    let n = basis.ncols();
    let det = ortho.determinant();
    let norm_product: f64 = (0..n).map(|i| basis.column(i).norm()).product();
    (det.abs() / norm_product).powf(1.0 / n as f64)
}

fn main() {
    let b = DMatrix::from_row_slice(6, 6, &[
        19.0, 2.0, 32.0, 46.0, 3.0, 33.0,
        15.0, 42.0, 10.0, 43.0, 2.0, 23.0,
        12.0, 35.0, 22.0, 18.0, 4.0, 45.0,
        43.0, 3.0, 48.0, 11.0, 10.0, 36.0,
        11.0, 36.0, 11.0, 33.0, 29.0, 11.0,
        40.0, 38.0, 31.0, 3.0, 41.0, 29.0,
    ]);

    let b_reduced = lll(&b, 0.75);
    let b_reduced_099 = lll(&b, 0.99);

    println!("Original Basis:\n{}", b);
    println!("LLL Reduced Basis (delta=0.75):\n{}", b_reduced);
    println!("LLL Reduced Basis (delta=0.99):\n{}", b_reduced_099);

    let hadamard_original = hadamard_ratio(&b);
    let hadamard_reduced_075 = hadamard_ratio(&b_reduced);
    let hadamard_reduced_099 = hadamard_ratio(&b_reduced_099);

    println!("Hadamard Ratio (Original): {}", hadamard_original);
    println!("Hadamard Ratio (Reduced, delta=0.75): {}", hadamard_reduced_075);
    println!("Hadamard Ratio (Reduced, delta=0.99): {}", hadamard_reduced_099);
}

