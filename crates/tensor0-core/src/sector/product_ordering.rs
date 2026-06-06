use crate::error::{Result, Tensor0Error};

// Dense rank/unrank for the same Manhattan order represented by ProductSector::sort_key().
// A cap is Some(n) for finite component indices 0..n, and None for an infinite
// component. Product values are ordered by sum(indices), then by component order
// inside each equal-sum layer.
pub(super) fn product_sort_index(indices: &[u128], caps: &[Option<u128>]) -> Result<u128> {
    let total = checked_sum(indices)?;
    let lower_layers = if total == 0 {
        0
    } else {
        bounded_count_le(total - 1, caps)?
    };

    let mut same_layer_before = 0u128;
    let mut prefix_sum = 0u128;
    for (position, current) in indices.iter().copied().enumerate() {
        if current > 0 {
            let mut suffix_caps = Vec::with_capacity(caps.len() - position);
            suffix_caps.push(Some(current));
            suffix_caps.extend_from_slice(&caps[position + 1..]);
            let remaining_total = total - prefix_sum;
            same_layer_before = same_layer_before
                .checked_add(bounded_count_exact(remaining_total, &suffix_caps)?)
                .ok_or(Tensor0Error::SectorIndexOverflow)?;
        }
        prefix_sum += current;
    }

    lower_layers
        .checked_add(same_layer_before)
        .ok_or(Tensor0Error::SectorIndexOverflow)
}

pub(super) fn product_indices_at(index: u128, caps: &[Option<u128>]) -> Result<Vec<u128>> {
    if let Some(size) = finite_cardinality(caps)? {
        if index >= size {
            return Err(Tensor0Error::SectorIndexOverflow);
        }
    }

    let layer = layer_for_product_index(index, caps)?;
    let lower_layers = if layer == 0 {
        0
    } else {
        bounded_count_le(layer - 1, caps)?
    };
    let mut offset = index - lower_layers;
    let mut prefix_sum = 0u128;
    let mut indices = Vec::with_capacity(caps.len());

    // Unrank the point inside its Manhattan layer, one component at a time.
    for position in 0..caps.len() {
        let remaining_total = layer - prefix_sum;
        let candidate_count = candidate_count(remaining_total, caps[position])?;
        let current = component_index_at_offset(
            remaining_total,
            offset,
            candidate_count,
            &caps[position + 1..],
        )?;
        let previous_count = if current == 0 {
            0
        } else {
            count_with_component_limit(remaining_total, current, &caps[position + 1..])?
        };
        offset -= previous_count;
        prefix_sum += current;
        indices.push(current);
    }

    if prefix_sum == layer && offset == 0 {
        Ok(indices)
    } else {
        Err(Tensor0Error::SectorIndexOverflow)
    }
}

fn checked_sum(values: &[u128]) -> Result<u128> {
    values.iter().try_fold(0u128, |total, value| {
        total
            .checked_add(*value)
            .ok_or(Tensor0Error::SectorIndexOverflow)
    })
}

fn finite_cardinality(caps: &[Option<u128>]) -> Result<Option<u128>> {
    let mut size = 1u128;
    for cap in caps {
        match cap {
            Some(cap) => {
                size = size
                    .checked_mul(*cap)
                    .ok_or(Tensor0Error::SectorIndexOverflow)?;
            }
            None => return Ok(None),
        }
    }
    Ok(Some(size))
}

fn layer_for_product_index(index: u128, caps: &[Option<u128>]) -> Result<u128> {
    let mut high = 0u128;
    while bounded_count_le(high, caps)? <= index {
        high = high
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(Tensor0Error::SectorIndexOverflow)?;
    }

    let mut low = 0u128;
    while low < high {
        let mid = low + (high - low) / 2;
        if bounded_count_le(mid, caps)? > index {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Ok(low)
}

fn candidate_count(remaining_total: u128, cap: Option<u128>) -> Result<u128> {
    let layer_count = remaining_total
        .checked_add(1)
        .ok_or(Tensor0Error::SectorIndexOverflow)?;
    Ok(cap.map_or(layer_count, |cap| cap.min(layer_count)))
}

fn component_index_at_offset(
    remaining_total: u128,
    offset: u128,
    candidate_count: u128,
    suffix_caps: &[Option<u128>],
) -> Result<u128> {
    if candidate_count == 0 {
        return Err(Tensor0Error::SectorIndexOverflow);
    }
    let total_count = count_with_component_limit(remaining_total, candidate_count, suffix_caps)?;
    if offset >= total_count {
        return Err(Tensor0Error::SectorIndexOverflow);
    }

    let mut low = 1u128;
    let mut high = candidate_count;
    while low < high {
        let mid = low + (high - low) / 2;
        if count_with_component_limit(remaining_total, mid, suffix_caps)? > offset {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Ok(low - 1)
}

fn count_with_component_limit(
    total: u128,
    limit: u128,
    suffix_caps: &[Option<u128>],
) -> Result<u128> {
    let mut caps = Vec::with_capacity(suffix_caps.len() + 1);
    caps.push(Some(limit));
    caps.extend_from_slice(suffix_caps);
    bounded_count_exact(total, &caps)
}

fn bounded_count_le(total: u128, caps: &[Option<u128>]) -> Result<u128> {
    bounded_count(total, caps, caps.len(), true)
}

fn bounded_count_exact(total: u128, caps: &[Option<u128>]) -> Result<u128> {
    bounded_count(total, caps, caps.len().saturating_sub(1), false)
}

fn bounded_count(
    total: u128,
    caps: &[Option<u128>],
    binomial_k: usize,
    cumulative: bool,
) -> Result<u128> {
    if caps.iter().any(|cap| matches!(cap, Some(0))) {
        return Ok(0);
    }
    if caps.is_empty() {
        return Ok(u128::from(total == 0 || cumulative));
    }

    let finite_caps = caps.iter().copied().flatten().collect::<Vec<_>>();
    let subset_count = 1usize
        .checked_shl(finite_caps.len() as u32)
        .ok_or(Tensor0Error::SectorIndexOverflow)?;
    let mut positive = 0u128;
    let mut negative = 0u128;

    for mask in 0..subset_count {
        let mut shift = 0u128;
        let mut bits = 0u32;
        for (index, cap) in finite_caps.iter().copied().enumerate() {
            if (mask & (1usize << index)) != 0 {
                shift = shift
                    .checked_add(cap)
                    .ok_or(Tensor0Error::SectorIndexOverflow)?;
                bits += 1;
            }
        }
        if shift > total {
            continue;
        }
        let n = if cumulative {
            total
                .checked_sub(shift)
                .and_then(|value| value.checked_add(caps.len() as u128))
                .ok_or(Tensor0Error::SectorIndexOverflow)?
        } else {
            total
                .checked_sub(shift)
                .and_then(|value| value.checked_add(caps.len().saturating_sub(1) as u128))
                .ok_or(Tensor0Error::SectorIndexOverflow)?
        };
        let term = checked_binomial(n, binomial_k)?;
        if bits.is_multiple_of(2) {
            positive = positive
                .checked_add(term)
                .ok_or(Tensor0Error::SectorIndexOverflow)?;
        } else {
            negative = negative
                .checked_add(term)
                .ok_or(Tensor0Error::SectorIndexOverflow)?;
        }
    }

    positive
        .checked_sub(negative)
        .ok_or(Tensor0Error::SectorIndexOverflow)
}

fn checked_binomial(n: u128, k: usize) -> Result<u128> {
    if k == 0 {
        return Ok(1);
    }
    let k_u128 = k as u128;
    if n < k_u128 {
        return Ok(0);
    }
    let k = k_u128.min(n - k_u128) as usize;
    let mut result = 1u128;
    for i in 1..=k {
        let mut numerator = n - k as u128 + i as u128;
        let mut denominator = i as u128;

        let common = gcd(numerator, denominator);
        numerator /= common;
        denominator /= common;

        let common = gcd(result, denominator);
        result /= common;
        denominator /= common;

        if denominator != 1 {
            return Err(Tensor0Error::SectorIndexOverflow);
        }
        result = result
            .checked_mul(numerator)
            .ok_or(Tensor0Error::SectorIndexOverflow)?;
    }
    Ok(result)
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::{product_indices_at, product_sort_index};

    #[test]
    fn rank_and_unrank_are_inverse_for_mixed_cardinality() {
        let caps = [None, Some(2), None];
        for rank in 0..50 {
            let indices = product_indices_at(rank, &caps).unwrap();
            assert!(indices[1] < 2);
            assert_eq!(product_sort_index(&indices, &caps).unwrap(), rank);
        }
    }

    #[test]
    fn rank_past_finite_product_cardinality_is_rejected() {
        let caps = [Some(2), Some(3)];
        for rank in 0..6 {
            let indices = product_indices_at(rank, &caps).unwrap();
            assert_eq!(product_sort_index(&indices, &caps).unwrap(), rank);
        }
        assert!(product_indices_at(6, &caps).is_err());
    }
}
