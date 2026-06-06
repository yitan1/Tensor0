use serde::{Deserialize, Serialize};

use crate::error::{Result, Tensor0Error};

use super::{EncodedSectorValue, FermionParity, SU2Irrep, Sector, U1Irrep};

/// Built-in group families supported by v0 irreducible representations.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GroupSpec {
    #[serde(rename = "u1")]
    U1,
    #[serde(rename = "zn")]
    ZN { n: i64 },
    #[serde(rename = "su2")]
    SU2,
}

/// Serde metadata representation for sector families.
///
/// Direct enum construction is a raw representation. Use constructors or
/// `canonicalize()` before hashing, fingerprinting, or crossing Python/JAX
/// metadata boundaries.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SectorSpec {
    Irrep { group: GroupSpec },
    FermionParity,
    Product { components: Vec<SectorSpec> },
}

struct EncodedValueCursor<'a> {
    value: &'a [i64],
    offset: usize,
}

impl<'a> EncodedValueCursor<'a> {
    fn new(value: &'a [i64]) -> Self {
        EncodedValueCursor { value, offset: 0 }
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

impl SectorSpec {
    pub fn u1() -> SectorSpec {
        SectorSpec::Irrep {
            group: GroupSpec::U1,
        }
    }

    pub fn zn(n: i64) -> Result<SectorSpec> {
        ensure_zn_modulus(n)?;
        Ok(SectorSpec::Irrep {
            group: GroupSpec::ZN { n },
        })
    }

    pub fn su2() -> SectorSpec {
        SectorSpec::Irrep {
            group: GroupSpec::SU2,
        }
    }

    pub fn fermion_parity() -> SectorSpec {
        SectorSpec::FermionParity
    }

    pub fn product(components: Vec<SectorSpec>) -> Result<SectorSpec> {
        SectorSpec::Product { components }.canonicalize()
    }

    pub fn canonicalize(self) -> Result<SectorSpec> {
        match self {
            SectorSpec::Irrep {
                group: GroupSpec::ZN { n },
            } => {
                ensure_zn_modulus(n)?;
                Ok(SectorSpec::Irrep {
                    group: GroupSpec::ZN { n },
                })
            }
            SectorSpec::Irrep { group } => Ok(SectorSpec::Irrep { group }),
            SectorSpec::FermionParity => Ok(SectorSpec::FermionParity),
            SectorSpec::Product { components } => canonicalize_product(components),
        }
    }
}

impl SectorSpec {
    pub fn canonicalize_value(&self, value: &[i64]) -> Result<EncodedSectorValue> {
        let spec = self.clone().canonicalize()?;
        let mut cursor = EncodedValueCursor::new(value);
        let value = spec.canonicalize_from_cursor(&mut cursor)?;
        cursor.finish()?;
        Ok(value)
    }

    pub fn quantum_dim(&self, value: &[i64]) -> Result<usize> {
        let spec = self.clone().canonicalize()?;
        let mut cursor = EncodedValueCursor::new(value);
        let dim = spec.quantum_dim_from_cursor(&mut cursor)?;
        cursor.finish()?;
        Ok(dim)
    }

    fn canonicalize_from_cursor(
        &self,
        cursor: &mut EncodedValueCursor<'_>,
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

    fn quantum_dim_from_cursor(&self, cursor: &mut EncodedValueCursor<'_>) -> Result<usize> {
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

fn canonicalize_product(components: Vec<SectorSpec>) -> Result<SectorSpec> {
    let mut canonical_components = Vec::new();
    for component in components {
        match component.canonicalize()? {
            SectorSpec::Product { components } => {
                canonical_components.extend(components);
            }
            component => canonical_components.push(component),
        }
    }
    if canonical_components.len() < 2 {
        return Err(Tensor0Error::BadProductSectorArity {
            actual: canonical_components.len(),
        });
    }
    Ok(SectorSpec::Product {
        components: canonical_components,
    })
}

fn ensure_zn_modulus(n: i64) -> Result<()> {
    if n > 0 {
        Ok(())
    } else {
        Err(Tensor0Error::BadZnModulus { n })
    }
}

fn decode_sector<I: Sector>(cursor: &mut EncodedValueCursor<'_>) -> Result<I> {
    let width = I::encoded_width();
    I::decode_value(cursor.take_component_value(width)?)
}
