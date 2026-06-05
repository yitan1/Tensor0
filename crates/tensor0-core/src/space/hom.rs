use crate::error::{Result, Tensor0Error};
use crate::fingerprint::fingerprint;
use crate::sector::Sector;

use super::graded::GradedSpace;
use super::product::ProductSpace;
use super::spec::HomSpaceSpec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HomSpace<I: Sector> {
    codomain: ProductSpace<I>,
    domain: ProductSpace<I>,
    fingerprint: u128,
}

impl<I: Sector> HomSpace<I> {
    pub fn new(codomain: ProductSpace<I>, domain: ProductSpace<I>) -> Result<Self> {
        let fingerprint = fingerprint(&hom_spec(&codomain, &domain, 0))?;
        Ok(HomSpace {
            codomain,
            domain,
            fingerprint,
        })
    }

    pub fn from_spec(spec: HomSpaceSpec) -> Result<Self> {
        let codomain = ProductSpace::<I>::from_spec(spec.codomain)?;
        let domain = ProductSpace::<I>::from_spec(spec.domain)?;
        HomSpace::new(codomain, domain)
    }

    pub fn to_spec(&self) -> HomSpaceSpec {
        hom_spec(&self.codomain, &self.domain, self.fingerprint)
    }

    pub fn codomain(&self) -> &ProductSpace<I> {
        &self.codomain
    }

    pub fn domain(&self) -> &ProductSpace<I> {
        &self.domain
    }

    pub fn fingerprint(&self) -> u128 {
        self.fingerprint
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
            return self.domain.factors()[domain_index].dual();
        }

        Err(visible_index_error())
    }

    pub fn visible_legs(&self) -> Result<Vec<GradedSpace<I>>> {
        let mut visible = self.codomain.factors().to_vec();
        for factor in self.domain.factors() {
            visible.push(factor.dual()?);
        }
        Ok(visible)
    }

    pub fn permute(&self, p_codomain: &[usize], p_domain: &[usize]) -> Result<Self> {
        validate_visible_permutation(self.numind(), p_codomain, p_domain)?;
        let visible = self.visible_legs()?;
        let (codomain, domain) = select_visible_spaces(&visible, p_codomain, p_domain)?;
        HomSpace::new(codomain, domain)
    }
}

fn select_visible_spaces<I: Sector>(
    visible: &[GradedSpace<I>],
    p_codomain: &[usize],
    p_domain: &[usize],
) -> Result<(ProductSpace<I>, ProductSpace<I>)> {
    let codomain = p_codomain
        .iter()
        .map(|index| visible[*index].clone())
        .collect::<Vec<_>>();
    let domain = p_domain
        .iter()
        .map(|index| visible[*index].dual())
        .collect::<Result<Vec<_>>>()?;

    Ok((ProductSpace::new(codomain)?, ProductSpace::new(domain)?))
}

fn validate_visible_permutation(
    visible_len: usize,
    p_codomain: &[usize],
    p_domain: &[usize],
) -> Result<()> {
    if p_codomain.len() + p_domain.len() != visible_len {
        return Err(invalid_permutation_error());
    }

    let mut seen = vec![false; visible_len];
    for index in p_codomain.iter().chain(p_domain) {
        if *index >= visible_len || seen[*index] {
            return Err(invalid_permutation_error());
        }
        seen[*index] = true;
    }
    Ok(())
}

fn invalid_permutation_error() -> Tensor0Error {
    Tensor0Error::Message(
        "transform permutation must contain each visible index exactly once".to_string(),
    )
}

fn visible_index_error() -> Tensor0Error {
    Tensor0Error::Message("visible index out of range".to_string())
}

fn hom_spec<I: Sector>(
    codomain: &ProductSpace<I>,
    domain: &ProductSpace<I>,
    fingerprint: u128,
) -> HomSpaceSpec {
    HomSpaceSpec {
        codomain: codomain.to_spec(),
        domain: domain.to_spec(),
        fingerprint,
    }
}
