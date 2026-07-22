use std::cmp::Ordering;

use ndarray::Array4;
use smallvec::smallvec;

use crate::error::{Result, Tensor0Error};

use super::fermion_parity::FermionParity;
use super::product_ordering::{product_indices_at, product_sort_index};
use super::su2::SU2Irrep;
use super::u1::U1Irrep;
use super::{
    require_width, BraidingStyle, EncodedSectorValue, FusionStyle, Sector, SectorCardinality,
    SectorSpec, SectorTuple, SortKey,
};

/// Typed product sector whose component order is part of the Rust type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProductSector<T: SectorTuple> {
    sectors: T,
}

pub type FermionNumber = ProductSector<(U1Irrep, FermionParity)>;
pub type FermionParityU1Irrep = ProductSector<(FermionParity, U1Irrep)>;
pub type U1SU2Irrep = ProductSector<(U1Irrep, SU2Irrep)>;
pub type FermionParitySU2Irrep = ProductSector<(FermionParity, SU2Irrep)>;
pub type FermionParityU1SU2Irrep = ProductSector<(FermionParity, U1Irrep, SU2Irrep)>;

impl<T: SectorTuple> ProductSector<T> {
    pub fn new(sectors: T) -> ProductSector<T> {
        ProductSector { sectors }
    }
}

impl ProductSector<(U1Irrep, FermionParity)> {
    pub fn fermion_parity(&self) -> bool {
        self.sectors.1.is_odd()
    }
}

impl<T: SectorTuple> Sector for ProductSector<T> {
    fn sector_spec() -> SectorSpec {
        T::sector_spec()
    }

    fn encoded_width() -> usize {
        T::encoded_width()
    }

    fn decode_value(value: &[i64]) -> Result<Self> {
        Ok(ProductSector::new(T::decode_value(value)?))
    }

    fn encode_value(&self) -> EncodedSectorValue {
        self.sectors.encode_value()
    }

    fn unit() -> Self {
        ProductSector::new(T::unit())
    }

    fn dual(&self) -> Self {
        ProductSector::new(self.sectors.dual())
    }

    fn quantum_dim(&self) -> usize {
        self.sectors.quantum_dim()
    }

    fn fusion_style() -> FusionStyle {
        T::fusion_style()
    }

    fn braiding_style() -> BraidingStyle {
        T::braiding_style()
    }

    fn cardinality() -> Result<SectorCardinality> {
        T::cardinality()
    }

    fn fusion_outputs(&self, rhs: &Self) -> impl Iterator<Item = Self> {
        self.sectors
            .fusion_outputs(&rhs.sectors)
            .map(ProductSector::new)
    }

    fn n_symbol(a: &Self, b: &Self, c: &Self) -> usize {
        T::n_symbol(&a.sectors, &b.sectors, &c.sectors)
    }

    fn f_symbol(a: &Self, b: &Self, c: &Self, d: &Self, e: &Self, f: &Self) -> Result<f64> {
        T::f_symbol(
            &a.sectors, &b.sectors, &c.sectors, &d.sectors, &e.sectors, &f.sectors,
        )
    }

    fn r_symbol(a: &Self, b: &Self, c: &Self) -> f64 {
        T::r_symbol(&a.sectors, &b.sectors, &c.sectors)
    }

    fn twist(&self) -> f64 {
        self.sectors.twist()
    }

    fn fusion_tensor(a: &Self, b: &Self, c: &Self) -> Result<Array4<f64>> {
        T::fusion_tensor(&a.sectors, &b.sectors, &c.sectors)
    }

    fn sort_key(&self) -> SortKey {
        self.sectors.sort_key()
    }

    fn sort_index(&self) -> Result<u128> {
        self.sectors.sort_index()
    }

    fn value_at(index: u128) -> Result<Self> {
        Ok(ProductSector::new(T::value_at(index)?))
    }
}

impl<T: SectorTuple> Ord for ProductSector<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sectors.sort_key().cmp(&other.sectors.sort_key())
    }
}

impl<T: SectorTuple> PartialOrd for ProductSector<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

macro_rules! impl_sector_tuple {
    ($($name:ident: $index:tt),+) => {
        impl<$($name: Sector),+> SectorTuple for ($($name,)+) {
            fn sector_spec() -> SectorSpec {
                let components = vec![$($name::sector_spec()),+];
                SectorSpec::product(components)
                    .expect("product sector tuple has at least two canonical component specs")
            }

            fn encoded_width() -> usize {
                let mut width = 0usize;
                $(
                    width = width.saturating_add($name::encoded_width());
                )+
                width
            }

            fn decode_value(value: &[i64]) -> Result<Self> {
                require_width(value, Self::encoded_width())?;
                let mut offset = 0usize;
                Ok((
                    $(decode_component::<$name>(value, &mut offset)?,)+
                ))
            }

            fn encode_value(&self) -> EncodedSectorValue {
                let mut encoded = EncodedSectorValue::new();
                $(
                    encoded.extend_from_slice(&self.$index.encode_value());
                )+
                encoded
            }

            fn unit() -> Self {
                ($($name::unit(),)+)
            }

            fn dual(&self) -> Self {
                ($(self.$index.dual(),)+)
            }

            fn quantum_dim(&self) -> usize {
                let mut dim = 1usize;
                $(
                    dim = dim
                        .checked_mul(self.$index.quantum_dim())
                        .expect("product quantum dimension overflowed");
                )+
                dim
            }

            fn fusion_style() -> FusionStyle {
                combine_fusion_style(&[$($name::fusion_style()),+])
            }

            fn braiding_style() -> BraidingStyle {
                combine_braiding_style(&[$($name::braiding_style()),+])
            }

            fn cardinality() -> Result<SectorCardinality> {
                combine_cardinality(&[$($name::cardinality()?),+])
            }

            fn fusion_outputs(&self, rhs: &Self) -> impl Iterator<Item = Self> {
                let mut encoded_outputs = vec![EncodedSectorValue::new()];
                $(
                    let component_outputs = self.$index
                        .fusion_outputs(&rhs.$index)
                        .into_iter()
                        .map(|value| value.encode_value())
                        .collect::<Vec<_>>();
                    encoded_outputs = extend_encoded_outputs(encoded_outputs, component_outputs);
                )+

                let mut outputs = encoded_outputs
                    .into_iter()
                    .map(|value| Self::decode_value(&value))
                    .collect::<Result<Vec<_>>>()
                    .expect("product fusion outputs must decode to the same product sector");
                outputs.sort();
                outputs.dedup();
                outputs.into_iter()
            }

            fn n_symbol(a: &Self, b: &Self, c: &Self) -> usize {
                let mut fuses = true;
                $(fuses &= $name::n_symbol(&a.$index, &b.$index, &c.$index) != 0;)+
                usize::from(fuses)
            }

            fn f_symbol(a: &Self, b: &Self, c: &Self, d: &Self, e: &Self, f: &Self) -> Result<f64> {
                let mut symbol = 1.0;
                $(
                    symbol *= $name::f_symbol(
                        &a.$index,
                        &b.$index,
                        &c.$index,
                        &d.$index,
                        &e.$index,
                        &f.$index,
                    )?;
                )+
                Ok(symbol)
            }

            fn r_symbol(a: &Self, b: &Self, c: &Self) -> f64 {
                if Self::n_symbol(a, b, c) == 0 {
                    return 0.0;
                }
                let mut symbol = 1.0;
                $(
                    symbol *= $name::r_symbol(&a.$index, &b.$index, &c.$index);
                )+
                symbol
            }

            fn twist(&self) -> f64 {
                let mut value = 1.0;
                $(
                    value *= self.$index.twist();
                )+
                value
            }

            fn fusion_tensor(a: &Self, b: &Self, c: &Self) -> Result<Array4<f64>> {
                kron_all_array4([
                    $($name::fusion_tensor(&a.$index, &b.$index, &c.$index)?),+
                ])
            }

            fn sort_key(&self) -> SortKey {
                let indices = [$(sort_index_or_max(&self.$index)),+];
                sort_key_from_indices(&indices)
            }

            fn sort_index(&self) -> Result<u128> {
                let indices = [$(self.$index.sort_index()?),+];
                let caps = [$(cardinality_cap($name::cardinality()?)),+];
                product_sort_index(&indices, &caps)
            }

            fn value_at(index: u128) -> Result<Self> {
                let caps = [$(cardinality_cap($name::cardinality()?)),+];
                let indices = product_indices_at(index, &caps)?;
                Ok((
                    $($name::value_at(indices[$index])?,)+
                ))
            }
        }
    };
}

impl_sector_tuple!(A: 0, B: 1);
impl_sector_tuple!(A: 0, B: 1, C: 2);
impl_sector_tuple!(A: 0, B: 1, C: 2, D: 3);

fn decode_component<I: Sector>(value: &[i64], offset: &mut usize) -> Result<I> {
    let width = I::encoded_width();
    let decoded = I::decode_value(&value[*offset..*offset + width])?;
    *offset += width;
    Ok(decoded)
}

fn extend_encoded_outputs(
    prefixes: Vec<EncodedSectorValue>,
    component_outputs: Vec<EncodedSectorValue>,
) -> Vec<EncodedSectorValue> {
    let mut outputs = Vec::new();
    for prefix in &prefixes {
        for component_output in &component_outputs {
            let mut output = prefix.clone();
            output.extend_from_slice(component_output);
            outputs.push(output);
        }
    }
    outputs
}

fn kron_all_array4<I>(tensors: I) -> Result<Array4<f64>>
where
    I: IntoIterator<Item = Array4<f64>>,
{
    let mut iter = tensors.into_iter();
    let first = iter.next().ok_or_else(|| {
        Tensor0Error::Message("fusiontensor product requires at least one component".to_string())
    })?;
    iter.try_fold(first, |left, right| kron_array4(&left, &right))
}

fn kron_array4(left: &Array4<f64>, right: &Array4<f64>) -> Result<Array4<f64>> {
    let (left_a, left_b, left_c, left_n) = left.dim();
    let (right_a, right_b, right_c, right_n) = right.dim();
    let dims = (
        checked_usize_mul(left_a, right_a, "product fusiontensor dimension")?,
        checked_usize_mul(left_b, right_b, "product fusiontensor dimension")?,
        checked_usize_mul(left_c, right_c, "product fusiontensor dimension")?,
        checked_usize_mul(left_n, right_n, "product fusiontensor dimension")?,
    );
    let mut tensor = Array4::zeros(dims);

    for ((ka, kb, kc, n), value) in tensor.indexed_iter_mut() {
        *value = left[[ka / right_a, kb / right_b, kc / right_c, n / right_n]]
            * right[[ka % right_a, kb % right_b, kc % right_c, n % right_n]];
    }

    Ok(tensor)
}

fn checked_usize_mul(left: usize, right: usize, context: &str) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        Tensor0Error::Message(format!(
            "sector symbol integer arithmetic overflowed in {context}",
        ))
    })
}

fn combine_fusion_style(styles: &[FusionStyle]) -> FusionStyle {
    styles
        .iter()
        .copied()
        .max_by_key(|style| fusion_style_priority(*style))
        .unwrap_or(FusionStyle::UniqueFusion)
}

fn combine_braiding_style(styles: &[BraidingStyle]) -> BraidingStyle {
    styles
        .iter()
        .copied()
        .max_by_key(|style| braiding_style_priority(*style))
        .unwrap_or(BraidingStyle::Bosonic)
}

fn combine_cardinality(cardinalities: &[SectorCardinality]) -> Result<SectorCardinality> {
    let mut size = 1u128;
    for cardinality in cardinalities {
        match cardinality {
            SectorCardinality::Infinite => return Ok(SectorCardinality::Infinite),
            SectorCardinality::Finite(component_size) => {
                size = size
                    .checked_mul(*component_size)
                    .ok_or(Tensor0Error::SectorIndexOverflow)?;
            }
        }
    }
    Ok(SectorCardinality::Finite(size))
}

fn fusion_style_priority(style: FusionStyle) -> u8 {
    match style {
        FusionStyle::UniqueFusion => 0,
        FusionStyle::SimpleFusion => 1,
        FusionStyle::GenericFusion => 2,
    }
}

fn braiding_style_priority(style: BraidingStyle) -> u8 {
    match style {
        BraidingStyle::Bosonic => 0,
        BraidingStyle::NoBraiding => 1,
        BraidingStyle::Anyonic => 2,
        BraidingStyle::Fermionic => 3,
    }
}

fn sort_index_or_max<I: Sector>(value: &I) -> u128 {
    value.sort_index().unwrap_or(u128::MAX)
}

fn sort_key_from_indices(indices: &[u128]) -> SortKey {
    let total = indices
        .iter()
        .copied()
        .fold(0u128, |total, index| total.saturating_add(index));
    let mut key = smallvec![total];
    key.extend_from_slice(indices);
    key
}

fn cardinality_cap(cardinality: SectorCardinality) -> Option<u128> {
    match cardinality {
        SectorCardinality::Finite(size) => Some(size),
        SectorCardinality::Infinite => None,
    }
}
