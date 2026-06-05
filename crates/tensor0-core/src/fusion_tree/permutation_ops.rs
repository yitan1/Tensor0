use crate::error::{Result, Tensor0Error};

pub(crate) fn linearize_permutation(
    p_codomain: &[usize],
    p_domain: &[usize],
    source_numout: usize,
    source_numin: usize,
) -> Result<Vec<usize>> {
    if p_codomain.len() + p_domain.len() != source_numout + source_numin {
        return Err(invalid_permutation_error());
    }

    p_codomain
        .iter()
        .copied()
        .chain(p_domain.iter().rev().copied())
        .map(|p| linearize_index(p, source_numout, source_numin))
        .collect()
}

pub(crate) fn permutation_to_swaps(permutation: &[usize]) -> Vec<usize> {
    let mut p = permutation.to_vec();
    let mut swaps = Vec::new();
    for k in 0..p.len().saturating_sub(1) {
        for swap in (k..p[k]).rev() {
            swaps.push(swap);
        }
        for index in k + 1..p.len() {
            if p[index] < p[k] {
                p[index] += 1;
            }
        }
        p[k] = k;
    }
    swaps
}

pub(crate) fn is_cyclic_permutation(permutation: &[usize]) -> bool {
    if permutation.is_empty() {
        return true;
    }

    permutation.iter().enumerate().all(|(index, value)| {
        permutation[(index + 1) % permutation.len()] == (value + 1) % permutation.len()
    })
}

fn linearize_index(p: usize, n1: usize, n2: usize) -> Result<usize> {
    if p < n1 {
        return Ok(p);
    }

    let domain_index = p - n1;
    if domain_index < n2 {
        return Ok(n1 + n2 - 1 - domain_index);
    }

    Err(invalid_permutation_error())
}

fn invalid_permutation_error() -> Tensor0Error {
    Tensor0Error::Message(
        "transform permutation must contain each visible index exactly once".to_string(),
    )
}
