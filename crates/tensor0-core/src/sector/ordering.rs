use crate::error::{Result, Tensor0Error};

// Rank the same Manhattan order represented by Product::sort_key().
pub(super) fn product_sort_index(indices: &[u128], caps: &[Option<u128>]) -> Result<u128> {
    let total = checked_sum(indices)?;
    let lower_layers = if total == 0 {
        0
    } else {
        bounded_count_le(
            total
                .checked_sub(1)
                .ok_or(Tensor0Error::SectorIndexOverflow)?,
            caps,
        )?
    };

    let mut same_layer_before = 0u128;
    let mut prefix_sum = 0u128;
    for (position, current) in indices.iter().copied().enumerate() {
        if current > 0 {
            let mut suffix_caps = Vec::with_capacity(caps.len() - position);
            suffix_caps.push(Some(current));
            suffix_caps.extend_from_slice(&caps[position + 1..]);
            let remaining_total = total
                .checked_sub(prefix_sum)
                .ok_or(Tensor0Error::SectorIndexOverflow)?;
            same_layer_before = same_layer_before
                .checked_add(bounded_count_exact(remaining_total, &suffix_caps)?)
                .ok_or(Tensor0Error::SectorIndexOverflow)?;
        }
        prefix_sum = prefix_sum
            .checked_add(current)
            .ok_or(Tensor0Error::SectorIndexOverflow)?;
    }

    lower_layers
        .checked_add(same_layer_before)
        .ok_or(Tensor0Error::SectorIndexOverflow)
}

fn checked_sum(values: &[u128]) -> Result<u128> {
    values.iter().try_fold(0u128, |total, value| {
        total
            .checked_add(*value)
            .ok_or(Tensor0Error::SectorIndexOverflow)
    })
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
        let mut numerator = n
            .checked_sub(k as u128)
            .and_then(|value| value.checked_add(i as u128))
            .ok_or(Tensor0Error::SectorIndexOverflow)?;
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
