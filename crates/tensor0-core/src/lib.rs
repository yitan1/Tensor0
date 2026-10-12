pub mod dense;
pub mod error;
pub mod fusion_tree;
pub mod layout;
pub mod sector;
pub mod space;
pub mod transform;

pub fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
