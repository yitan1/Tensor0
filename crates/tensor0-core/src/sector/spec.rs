use serde::{Deserialize, Serialize};

use crate::error::{Result, Tensor0Error};

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

    pub fn validate(&self) -> Result<()> {
        let canonical = self.clone().canonicalize()?;
        if &canonical == self {
            Ok(())
        } else {
            Err(Tensor0Error::Message(
                "sector spec is not canonical".to_string(),
            ))
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
