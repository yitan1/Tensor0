//! Shared fusion-tree tensor and permutation helpers.

use ndarray::{ArrayD, IxDyn};

use crate::error::{Result, Tensor0Error};

pub(crate) fn tensordot(
    left: &ArrayD<f64>,
    right: &ArrayD<f64>,
    left_axes: &[usize],
    right_axes: &[usize],
) -> Result<ArrayD<f64>> {
    if left_axes.len() != right_axes.len() {
        return Err(Tensor0Error::Message(
            "tensordot axis count mismatch".to_string(),
        ));
    }

    validate_axes(left.ndim(), left_axes, "left")?;
    validate_axes(right.ndim(), right_axes, "right")?;

    for (&left_axis, &right_axis) in left_axes.iter().zip(right_axes) {
        let left_dim = left.shape()[left_axis];
        let right_dim = right.shape()[right_axis];
        if left_dim != right_dim {
            return Err(Tensor0Error::Message(format!(
                "tensordot contracted dimension mismatch: left axis {left_axis} has {left_dim}, right axis {right_axis} has {right_dim}",
            )));
        }
    }

    let left_free = (0..left.ndim())
        .filter(|axis| !left_axes.contains(axis))
        .collect::<Vec<_>>();
    let right_free = (0..right.ndim())
        .filter(|axis| !right_axes.contains(axis))
        .collect::<Vec<_>>();
    let mut left_permutation = left_free.clone();
    left_permutation.extend_from_slice(left_axes);
    let mut right_permutation = right_axes.to_vec();
    right_permutation.extend_from_slice(&right_free);

    let contracted_dim = left_axes
        .iter()
        .map(|&axis| left.shape()[axis])
        .product::<usize>();
    let left_free_dim = left_free
        .iter()
        .map(|&axis| left.shape()[axis])
        .product::<usize>();
    let right_free_dim = right_free
        .iter()
        .map(|&axis| right.shape()[axis])
        .product::<usize>();

    let left_matrix = left
        .clone()
        .permuted_axes(IxDyn(&left_permutation))
        .as_standard_layout()
        .to_owned()
        .into_shape_with_order((left_free_dim, contracted_dim))
        .map_err(|err| Tensor0Error::Message(format!("tensordot left reshape failed: {err}")))?;
    let right_matrix = right
        .clone()
        .permuted_axes(IxDyn(&right_permutation))
        .as_standard_layout()
        .to_owned()
        .into_shape_with_order((contracted_dim, right_free_dim))
        .map_err(|err| Tensor0Error::Message(format!("tensordot right reshape failed: {err}")))?;
    let product = left_matrix.dot(&right_matrix);

    let mut output_shape = left_free
        .iter()
        .map(|&axis| left.shape()[axis])
        .collect::<Vec<_>>();
    output_shape.extend(right_free.iter().map(|&axis| right.shape()[axis]));
    product
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|err| Tensor0Error::Message(format!("tensordot output reshape failed: {err}")))
}

pub(crate) fn linearize_permutation(
    p_codomain: &[usize],
    p_domain: &[usize],
    source_numout: usize,
    source_numin: usize,
) -> Result<Vec<usize>> {
    let visible_len = source_numout + source_numin;
    if p_codomain.len() + p_domain.len() != visible_len {
        return Err(invalid_permutation_error());
    }

    let mut seen = vec![false; visible_len];
    let mut permutation = Vec::with_capacity(visible_len);
    for index in p_codomain.iter().chain(p_domain.iter().rev()) {
        if *index >= visible_len || seen[*index] {
            return Err(invalid_permutation_error());
        }
        seen[*index] = true;
        permutation.push(linearize_index(*index, source_numout, source_numin));
    }

    Ok(permutation)
}

pub(crate) fn permutation_to_swaps(permutation: &[usize]) -> Vec<usize> {
    debug_assert!(is_valid_permutation(permutation));

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

fn linearize_index(index: usize, source_numout: usize, source_numin: usize) -> usize {
    if index < source_numout {
        index
    } else {
        source_numout + source_numin - 1 - (index - source_numout)
    }
}

fn invalid_permutation_error() -> Tensor0Error {
    Tensor0Error::Message(
        "transform permutation must contain each visible index exactly once".to_string(),
    )
}

fn is_valid_permutation(permutation: &[usize]) -> bool {
    let mut seen = vec![false; permutation.len()];
    for &index in permutation {
        if index >= permutation.len() || seen[index] {
            return false;
        }
        seen[index] = true;
    }
    true
}

fn validate_axes(rank: usize, axes: &[usize], label: &str) -> Result<()> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(Tensor0Error::Message(format!(
                "tensordot {label} axis out of range",
            )));
        }
        if seen[axis] {
            return Err(Tensor0Error::Message(format!(
                "tensordot {label} axes contain duplicates",
            )));
        }
        seen[axis] = true;
    }
    Ok(())
}
