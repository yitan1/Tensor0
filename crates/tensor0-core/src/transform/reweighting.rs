use crate::error::{Result, Tensor0Error};
use crate::layout::SectorStructure;
use crate::sector::{BraidingStyle, Sector};
use crate::space::HomSpace;

/// Returns whether a twist is semantically trivial without constructing layout.
///
/// This classifies empty, Bosonic, trivial Fermionic, and unit NoBraiding
/// selections. Unsupported Anyonic and non-unit NoBraiding selections return an
/// error. A `false` result may still cancel to identity on the actual fusion-tree
/// pairs; [`twist_subblock_factors`] detects that layout-dependent case.
pub fn twist_is_trivial<I: Sector>(space: &HomSpace<I>, indices: &[usize]) -> Result<bool> {
    validate_twist_indices(space, indices)?;

    if indices.is_empty() {
        return Ok(true);
    }

    match I::braiding_style() {
        BraidingStyle::Bosonic => Ok(true),
        BraidingStyle::Fermionic => Ok(indices.iter().all(|&index| {
            space
                .visible_leg(index)
                .expect("twist index was validated")
                .sectors()
                .into_iter()
                .all(|(sector, _)| sector.twist() == 1.0)
        })),
        BraidingStyle::Anyonic => Err(Tensor0Error::Message(
            "twist does not support Anyonic sector families".to_string(),
        )),
        BraidingStyle::NoBraiding => {
            let selected_legs_are_unit = indices.iter().all(|&index| {
                space
                    .visible_leg(index)
                    .expect("twist index was validated")
                    .sectors()
                    .into_iter()
                    .all(|(sector, _)| sector == I::unit())
            });
            if selected_legs_are_unit {
                Ok(true)
            } else {
                Err(Tensor0Error::Message(
                    "twist does not support non-unit NoBraiding sectors".to_string(),
                ))
            }
        }
    }
}

/// Returns twist factors in canonical fusion-tree-pair/subblock order.
///
/// `Some` contains one factor per canonical subblock. `None` means the combined
/// factors cancel to identity on all fusion-tree pairs. When `inv` is true, each
/// combined factor is replaced by its reciprocal. This operation currently
/// supports only real symmetric twists. `structure` must match `space`;
/// supplying the cached sector structure keeps factor order identical to the
/// layout used for degeneracy subblocks.
pub fn twist_subblock_factors<I: Sector>(
    space: &HomSpace<I>,
    structure: &SectorStructure<I>,
    indices: &[usize],
    inv: bool,
) -> Result<Option<Vec<f64>>> {
    validate_twist_indices(space, indices)?;

    if !structure.matches_space(space) {
        return Err(Tensor0Error::Message(
            "sectorstructure does not match HomSpace sector structure".to_string(),
        ));
    }

    let factors = structure
        .fusiontree_pairs()
        .map(|pair| {
            let mut factor = 1.0;
            for &index in indices {
                let sector = if index < space.numout() {
                    &pair.row.uncoupled()[index]
                } else {
                    &pair.col.uncoupled()[index - space.numout()]
                };
                factor *= sector.twist();
            }
            if inv {
                factor = factor.recip();
            }
            Ok(factor)
        })
        .collect::<Result<Vec<_>>>()?;

    if factors.iter().all(|&factor| factor == 1.0) {
        Ok(None)
    } else {
        Ok(Some(factors))
    }
}

fn validate_twist_indices<I: Sector>(space: &HomSpace<I>, indices: &[usize]) -> Result<()> {
    let mut seen = vec![false; space.numind()];
    for &index in indices {
        if index >= space.numind() {
            return Err(Tensor0Error::Message(format!(
                "twist visible index {index} is out of range for rank {}",
                space.numind(),
            )));
        }
        if seen[index] {
            return Err(Tensor0Error::Message(
                "twist visible indices must be unique".to_string(),
            ));
        }
        seen[index] = true;
    }
    Ok(())
}
