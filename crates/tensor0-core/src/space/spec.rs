use serde::{Deserialize, Serialize};

use crate::sector::SectorSpec;

/// Serialized degeneracy dimension for one encoded sector value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SectorDimSpec {
    /// Encoded sector value, not a typed sector instance.
    pub sector: Vec<i64>,
    pub dim: usize,
}

/// Serialized elementary graded space.
///
/// Validation and typed sector decoding happen in `GradedSpace::from_spec`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ElementarySpaceSpec {
    pub sector_spec: SectorSpec,
    pub sectors: Vec<SectorDimSpec>,
    pub is_dual: bool,
}

/// Serialized tensor product of elementary graded spaces.
///
/// The top-level `sector_spec` preserves the sector family for empty product
/// spaces and is checked against every factor during `ProductSpace::from_spec`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductSpaceSpec {
    pub sector_spec: SectorSpec,
    pub factors: Vec<ElementarySpaceSpec>,
}

/// Serialized hom space, represented by codomain and domain product spaces.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HomSpaceSpec {
    pub codomain: ProductSpaceSpec,
    pub domain: ProductSpaceSpec,
}
