use thiserror::Error;

#[derive(Debug, Error)]
pub enum Tensor0Error {
    #[error("expected sector value width {expected}, got {actual}")]
    BadSectorWidth { expected: usize, actual: usize },
    #[error("ZN modulus must be positive, got {n}")]
    BadZnModulus { n: i64 },
    #[error(
        "product sector must contain at least two components after canonicalization, got {actual}"
    )]
    BadProductSectorArity { actual: usize },
    #[error("sector cardinality or sort index overflowed")]
    SectorIndexOverflow,
    #[error("sector spec mismatch: expected {expected}, got {actual}")]
    SectorSpecMismatch { expected: String, actual: String },
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, Tensor0Error>;
