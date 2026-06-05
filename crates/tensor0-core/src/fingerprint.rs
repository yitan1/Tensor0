use serde::Serialize;

use crate::error::{Result, Tensor0Error};

pub fn fingerprint<T: Serialize>(value: &T) -> Result<u128> {
    let bytes = serde_json::to_vec(value)
        .map_err(|err| Tensor0Error::Message(format!("fingerprint serialization failed: {err}")))?;
    let hash = blake3::hash(&bytes);
    let mut out = [0_u8; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    Ok(u128::from_le_bytes(out))
}
