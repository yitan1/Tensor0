use crate::error::{Result, Tensor0Error};

use super::{EncodedSectorValue, FermionParity, GroupSpec, SU2Irrep, Sector, SectorSpec, U1Irrep};

impl SectorSpec {
    pub fn canonicalize_value(&self, value: &[i64]) -> Result<EncodedSectorValue> {
        let spec = self.clone().canonicalize()?;
        let mut cursor = ProductValueCursor::new(value);
        let value = spec.canonicalize_from_cursor(&mut cursor)?;
        cursor.finish()?;
        Ok(value)
    }

    pub fn quantum_dim(&self, value: &[i64]) -> Result<usize> {
        let spec = self.clone().canonicalize()?;
        let mut cursor = ProductValueCursor::new(value);
        let dim = spec.quantum_dim_from_cursor(&mut cursor)?;
        cursor.finish()?;
        Ok(dim)
    }

    fn canonicalize_from_cursor(
        &self,
        cursor: &mut ProductValueCursor<'_>,
    ) -> Result<EncodedSectorValue> {
        match self {
            SectorSpec::Irrep {
                group: GroupSpec::U1,
            } => decode_sector::<U1Irrep>(cursor).map(|value| value.encode_value()),
            SectorSpec::Irrep {
                group: GroupSpec::SU2,
            } => decode_sector::<SU2Irrep>(cursor).map(|value| value.encode_value()),
            SectorSpec::FermionParity => {
                decode_sector::<FermionParity>(cursor).map(|value| value.encode_value())
            }
            SectorSpec::Irrep {
                group: GroupSpec::ZN { n },
            } => {
                let value = cursor.take_component_value(1)?;
                Ok(EncodedSectorValue::from_slice(&[value[0].rem_euclid(*n)]))
            }
            SectorSpec::Product { components } => {
                let mut value = EncodedSectorValue::new();
                for component in components {
                    value.extend_from_slice(&component.canonicalize_from_cursor(cursor)?);
                }
                Ok(value)
            }
        }
    }

    fn quantum_dim_from_cursor(&self, cursor: &mut ProductValueCursor<'_>) -> Result<usize> {
        match self {
            SectorSpec::Irrep {
                group: GroupSpec::U1,
            } => decode_sector::<U1Irrep>(cursor).map(|value| value.quantum_dim()),
            SectorSpec::Irrep {
                group: GroupSpec::SU2,
            } => decode_sector::<SU2Irrep>(cursor).map(|value| value.quantum_dim()),
            SectorSpec::FermionParity => {
                decode_sector::<FermionParity>(cursor).map(|value| value.quantum_dim())
            }
            SectorSpec::Irrep {
                group: GroupSpec::ZN { .. },
            } => {
                cursor.take_component_value(1)?;
                Ok(1)
            }
            SectorSpec::Product { components } => {
                components.iter().try_fold(1usize, |dim, component| {
                    dim.checked_mul(component.quantum_dim_from_cursor(cursor)?)
                        .ok_or_else(|| {
                            Tensor0Error::Message("product sector dimension overflowed".to_string())
                        })
                })
            }
        }
    }
}

struct ProductValueCursor<'a> {
    value: &'a [i64],
    offset: usize,
}

impl<'a> ProductValueCursor<'a> {
    fn new(value: &'a [i64]) -> Self {
        ProductValueCursor { value, offset: 0 }
    }

    fn take_component_value(&mut self, width: usize) -> Result<&'a [i64]> {
        let end = self
            .offset
            .checked_add(width)
            .ok_or(Tensor0Error::SectorIndexOverflow)?;
        if end > self.value.len() {
            return Err(Tensor0Error::BadSectorWidth {
                expected: width,
                actual: self.value.len().saturating_sub(self.offset),
            });
        }
        let value = &self.value[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn finish(&self) -> Result<()> {
        if self.offset == self.value.len() {
            Ok(())
        } else {
            Err(Tensor0Error::BadSectorWidth {
                expected: self.offset,
                actual: self.value.len(),
            })
        }
    }
}

fn decode_sector<I: Sector>(cursor: &mut ProductValueCursor<'_>) -> Result<I> {
    let width = I::encoded_width();
    I::decode_value(cursor.take_component_value(width)?)
}
