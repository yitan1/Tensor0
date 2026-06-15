mod degeneracy;
mod indices;
mod sector_structure;

pub use degeneracy::{
    build_degeneracy_structure, build_degeneracy_structure_from_sector_structure, BlockStructure,
    DegeneracyStructure, SubblockStructure,
};
pub(crate) use degeneracy::{subblockstructure_ref, SubblockStructureMapRef};
pub use sector_structure::{build_sector_structure, SectorStructure};
