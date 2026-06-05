use serde::{Deserialize, Serialize};

use crate::sector::SectorSpec;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SectorDimSpec {
    pub sector: Vec<i64>,
    pub dim: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ElementarySpaceSpec {
    pub sector_spec: SectorSpec,
    pub sectors: Vec<SectorDimSpec>,
    pub is_dual: bool,
    pub fingerprint: u128,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductSpaceSpec {
    pub sector_spec: SectorSpec,
    pub factors: Vec<ElementarySpaceSpec>,
    pub fingerprint: u128,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HomSpaceSpec {
    pub codomain: ProductSpaceSpec,
    pub domain: ProductSpaceSpec,
    pub fingerprint: u128,
}
