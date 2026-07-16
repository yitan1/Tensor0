use crate::error::{Result, Tensor0Error};
use crate::sector::Sector;

use super::graded::GradedSpace;
use super::product::ProductSpace;
use super::spec::HomSpaceSpec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HomSpace<I: Sector> {
    codomain: ProductSpace<I>,
    domain: ProductSpace<I>,
}

impl<I: Sector> HomSpace<I> {
    pub fn new(codomain: ProductSpace<I>, domain: ProductSpace<I>) -> Self {
        HomSpace { codomain, domain }
    }

    pub fn from_factor_spaces(codomain: Vec<GradedSpace<I>>, domain: Vec<GradedSpace<I>>) -> Self {
        HomSpace::new(ProductSpace::new(codomain), ProductSpace::new(domain))
    }

    pub fn from_spec(spec: HomSpaceSpec) -> Result<Self> {
        let codomain = ProductSpace::<I>::from_spec(spec.codomain)?;
        let domain = ProductSpace::<I>::from_spec(spec.domain)?;
        Ok(HomSpace::new(codomain, domain))
    }

    pub fn to_spec(&self) -> HomSpaceSpec {
        hom_spec(&self.codomain, &self.domain)
    }

    pub fn codomain(&self) -> &ProductSpace<I> {
        &self.codomain
    }

    pub fn domain(&self) -> &ProductSpace<I> {
        &self.domain
    }

    pub fn dual(&self) -> Self {
        HomSpace::new(self.domain.clone(), self.codomain.clone())
    }

    pub fn numout(&self) -> usize {
        self.codomain.factors().len()
    }

    pub fn numin(&self) -> usize {
        self.domain.factors().len()
    }

    pub fn numind(&self) -> usize {
        self.numout() + self.numin()
    }

    pub fn visible_leg(&self, index: usize) -> Result<GradedSpace<I>> {
        let numout = self.numout();
        if index < numout {
            return Ok(self.codomain.factors()[index].clone());
        }

        let domain_index = index - numout;
        if domain_index < self.numin() {
            return Ok(self.domain.factors()[domain_index].dual());
        }

        Err(Tensor0Error::Message(
            "visible index out of range".to_string(),
        ))
    }

    pub fn visible_legs(&self) -> Vec<GradedSpace<I>> {
        let mut visible = self.codomain.factors().to_vec();
        for factor in self.domain.factors() {
            visible.push(factor.dual());
        }
        visible
    }

    /// Insert a left unit at a 0-based visible-index boundary.
    ///
    /// At the codomain/domain boundary the unit becomes the first domain
    /// factor. `dual` selects the unit factor attached to that side of the
    /// HomSpace, matching ProductSpace and TensorKit semantics.
    pub fn insert_left_unit(&self, position: usize, dual: bool) -> Result<Self> {
        self.validate_unit_insertion_boundary(position)?;

        if position < self.numout() {
            Ok(HomSpace::new(
                self.codomain.insert_unit(position, dual)?,
                self.domain.clone(),
            ))
        } else {
            Ok(HomSpace::new(
                self.codomain.clone(),
                self.domain.insert_unit(position - self.numout(), dual)?,
            ))
        }
    }

    /// Insert a right unit at a 0-based visible-index boundary.
    ///
    /// At the codomain/domain boundary the unit becomes the final codomain
    /// factor. `dual` selects the unit factor attached to that side of the
    /// HomSpace, matching ProductSpace and TensorKit semantics.
    pub fn insert_right_unit(&self, position: usize, dual: bool) -> Result<Self> {
        self.validate_unit_insertion_boundary(position)?;

        if position <= self.numout() {
            Ok(HomSpace::new(
                self.codomain.insert_unit(position, dual)?,
                self.domain.clone(),
            ))
        } else {
            Ok(HomSpace::new(
                self.codomain.clone(),
                self.domain.insert_unit(position - self.numout(), dual)?,
            ))
        }
    }

    /// Remove a canonical unit-space factor at a 0-based visible index.
    pub fn remove_unit(&self, index: usize) -> Result<Self> {
        if index >= self.numind() {
            return Err(Tensor0Error::Message(format!(
                "unit removal index {index} is out of range for rank {}",
                self.numind(),
            )));
        }

        if index < self.numout() {
            Ok(HomSpace::new(
                self.codomain.remove_unit(index)?,
                self.domain.clone(),
            ))
        } else {
            Ok(HomSpace::new(
                self.codomain.clone(),
                self.domain.remove_unit(index - self.numout())?,
            ))
        }
    }

    pub fn flip(&self, indices: &[usize]) -> Result<Self> {
        let rank = self.numind();
        let mut seen = vec![false; rank];
        for &index in indices {
            if index >= rank {
                return Err(Tensor0Error::Message(format!(
                    "flip visible index {index} is out of range for rank {rank}",
                )));
            }
            if seen[index] {
                return Err(Tensor0Error::Message(
                    "flip visible indices must be unique".to_string(),
                ));
            }
            seen[index] = true;
        }

        let mut codomain = self.codomain.factors().to_vec();
        let mut domain = self.domain.factors().to_vec();
        for &index in indices {
            if index < self.numout() {
                codomain[index] = codomain[index].flip();
            } else {
                let domain_index = index - self.numout();
                domain[domain_index] = domain[domain_index].flip();
            }
        }
        Ok(HomSpace::from_factor_spaces(codomain, domain))
    }

    pub fn permute(&self, p_codomain: &[usize], p_domain: &[usize]) -> Result<Self> {
        validate_visible_permutation(self.numind(), p_codomain, p_domain)?;
        let visible = self.visible_legs();
        let (codomain, domain) = select_visible_spaces(&visible, p_codomain, p_domain);
        Ok(HomSpace::new(codomain, domain))
    }

    fn validate_unit_insertion_boundary(&self, position: usize) -> Result<()> {
        if position <= self.numind() {
            Ok(())
        } else {
            Err(Tensor0Error::Message(format!(
                "unit insertion boundary {position} is out of range for rank {}",
                self.numind(),
            )))
        }
    }
}

fn validate_visible_permutation(
    visible_len: usize,
    p_codomain: &[usize],
    p_domain: &[usize],
) -> Result<()> {
    let invalid_permutation = || {
        Tensor0Error::Message(
            "transform permutation must contain each visible index exactly once".to_string(),
        )
    };

    if p_codomain.len() + p_domain.len() != visible_len {
        return Err(invalid_permutation());
    }

    let mut seen = vec![false; visible_len];
    for index in p_codomain.iter().chain(p_domain) {
        if *index >= visible_len || seen[*index] {
            return Err(invalid_permutation());
        }
        seen[*index] = true;
    }
    Ok(())
}

fn select_visible_spaces<I: Sector>(
    visible: &[GradedSpace<I>],
    p_codomain: &[usize],
    p_domain: &[usize],
) -> (ProductSpace<I>, ProductSpace<I>) {
    let codomain = p_codomain
        .iter()
        .map(|index| visible[*index].clone())
        .collect::<Vec<_>>();
    let domain = p_domain
        .iter()
        .map(|index| visible[*index].dual())
        .collect::<Vec<_>>();

    (ProductSpace::new(codomain), ProductSpace::new(domain))
}

fn hom_spec<I: Sector>(codomain: &ProductSpace<I>, domain: &ProductSpace<I>) -> HomSpaceSpec {
    HomSpaceSpec {
        codomain: codomain.to_spec(),
        domain: domain.to_spec(),
    }
}
