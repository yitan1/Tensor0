use crate::error::{Result, Tensor0Error};
use crate::sector::Sector;
use crate::space::ProductSpace;

pub fn product_axes<I: Sector>(
    product: &ProductSpace<I>,
    sectors: &[I],
) -> Result<Vec<(usize, usize, usize, usize)>> {
    if product.factors().len() != sectors.len() {
        return Err(Tensor0Error::Message(
            "product axes sector arity mismatch".to_string(),
        ));
    }

    product
        .factors()
        .iter()
        .zip(sectors)
        .map(|(factor, sector)| factor_dense_axis(factor, sector))
        .collect()
}

fn factor_dense_axis<I: Sector>(
    factor: &crate::space::GradedSpace<I>,
    target: &I,
) -> Result<(usize, usize, usize, usize)> {
    let mut start = 0usize;
    for (sector, degeneracy_dim) in factor.sectors() {
        let quantum_dim = sector.quantum_dim();
        let stop = start + degeneracy_dim * quantum_dim;
        if &sector == target {
            return Ok((start, stop, degeneracy_dim, quantum_dim));
        }
        start = stop;
    }

    Err(Tensor0Error::Message(
        "sector is not present in product factor".to_string(),
    ))
}
