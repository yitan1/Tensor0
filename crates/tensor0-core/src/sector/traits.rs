use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::error::Result;
use ndarray::Array4;

use super::SectorSpec;

/// Primitive sector value payload used in canonical metadata.
pub type EncodedSectorValue = SmallVec<[i64; 4]>;

/// Infallible ordering key used by typed sector values.
pub type SortKey = SmallVec<[u128; 8]>;

/// Fusion category supported by a sector family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionStyle {
    UniqueFusion,
    SimpleFusion,
    GenericFusion,
}

/// Braiding behavior used by transform planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BraidingStyle {
    Bosonic,
    Fermionic,
    Anyonic,
    NoBraiding,
}

/// Whether a sector family has finitely enumerable values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SectorCardinality {
    Finite(u128),
    Infinite,
}

/// TensorKit-like behavior interface for a concrete typed sector family.
pub trait Sector: Clone + Eq + Ord + std::hash::Hash + Sized + 'static {
    fn sector_spec() -> SectorSpec;
    fn encoded_width() -> usize;
    fn decode_value(value: &[i64]) -> Result<Self>;
    fn encode_value(&self) -> EncodedSectorValue;

    fn unit() -> Self;
    fn dual(&self) -> Self;
    fn quantum_dim(&self) -> usize;

    fn fusion_style() -> FusionStyle;
    fn braiding_style() -> BraidingStyle;
    fn cardinality() -> Result<SectorCardinality>;

    fn fusion_outputs(&self, rhs: &Self) -> Vec<Self>;
    fn n_symbol(a: &Self, b: &Self, c: &Self) -> usize;
    fn f_symbol(a: &Self, b: &Self, c: &Self, d: &Self, e: &Self, f: &Self) -> Result<f64> {
        unique_fusion_f_symbol::<Self>(a, b, c, d, e, f)
    }
    fn a_symbol(a: &Self, b: &Self, c: &Self) -> Result<f64> {
        a_symbol_from_f_symbol(a, b, c)
    }
    fn b_symbol(a: &Self, b: &Self, c: &Self) -> Result<f64> {
        b_symbol_from_f_symbol(a, b, c)
    }
    fn frobenius_schur_phase(a: &Self) -> Result<f64> {
        frobenius_schur_phase_from_f_symbol(a)
    }
    fn r_symbol(a: &Self, b: &Self, c: &Self) -> f64;
    fn fusion_tensor(a: &Self, b: &Self, c: &Self) -> Result<Array4<f64>> {
        unique_fusion_tensor::<Self>(a, b, c)
    }
    fn sort_key(&self) -> SortKey;
    fn sort_index(&self) -> Result<u128>;
}

/// Delegation layer for typed product sector component tuples.
pub trait SectorTuple: Clone + Eq + Ord + std::hash::Hash + Sized + 'static {
    fn sector_spec() -> SectorSpec;
    fn encoded_width() -> usize;
    fn decode_value(value: &[i64]) -> Result<Self>;
    fn encode_value(&self) -> EncodedSectorValue;
    fn unit() -> Self;
    fn dual(&self) -> Self;
    fn quantum_dim(&self) -> usize;

    fn fusion_style() -> FusionStyle;
    fn braiding_style() -> BraidingStyle;
    fn cardinality() -> Result<SectorCardinality>;
    fn fusion_outputs(&self, rhs: &Self) -> Vec<Self>;
    fn n_symbol(a: &Self, b: &Self, c: &Self) -> usize;
    fn f_symbol(a: &Self, b: &Self, c: &Self, d: &Self, e: &Self, f: &Self) -> Result<f64>;
    fn r_symbol(a: &Self, b: &Self, c: &Self) -> f64;
    fn fusion_tensor(a: &Self, b: &Self, c: &Self) -> Result<Array4<f64>>;
    fn sort_key(&self) -> SortKey;
    fn sort_index(&self) -> Result<u128>;
}

fn unique_fusion_f_symbol<I: Sector>(a: &I, b: &I, c: &I, d: &I, e: &I, f: &I) -> Result<f64> {
    Ok(
        (I::n_symbol(a, b, e) * I::n_symbol(e, c, d) * I::n_symbol(b, c, f) * I::n_symbol(a, f, d))
            as f64,
    )
}

fn b_symbol_from_f_symbol<I: Sector>(a: &I, b: &I, c: &I) -> Result<f64> {
    let f_symbol = I::f_symbol(a, b, &b.dual(), a, c, &I::unit())?;
    let scale =
        ((a.quantum_dim() as f64) * (b.quantum_dim() as f64) / (c.quantum_dim() as f64)).sqrt();
    Ok(scale * f_symbol)
}

fn a_symbol_from_f_symbol<I: Sector>(a: &I, b: &I, c: &I) -> Result<f64> {
    let unit = I::unit();
    let f_symbol = I::f_symbol(&a.dual(), a, b, b, &unit, c)?;
    let scale =
        ((a.quantum_dim() as f64) * (b.quantum_dim() as f64) / (c.quantum_dim() as f64)).sqrt();
    // GenericFusion/complex symbols must restore TensorKit's conjugation here.
    Ok(scale * I::frobenius_schur_phase(a)? * f_symbol)
}

fn frobenius_schur_phase_from_f_symbol<I: Sector>(a: &I) -> Result<f64> {
    let unit = I::unit();
    Ok(I::f_symbol(a, &a.dual(), a, a, &unit, &unit)?.signum())
}

fn unique_fusion_tensor<I: Sector>(a: &I, b: &I, c: &I) -> Result<Array4<f64>> {
    Ok(Array4::from_elem((1, 1, 1, 1), I::n_symbol(a, b, c) as f64))
}
