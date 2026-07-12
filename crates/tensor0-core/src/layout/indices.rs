use std::hash::Hash;

use indexmap::IndexSet;

use crate::error::{Result, Tensor0Error};

#[derive(Clone, Debug)]
pub(super) struct Indices<K: Eq + Hash> {
    inner: IndexSet<K>,
}

impl<K: Eq + Hash> Indices<K> {
    pub(super) fn new(values: Vec<K>) -> Result<Self> {
        let mut inner = IndexSet::with_capacity(values.len());
        for value in values {
            if !inner.insert(value) {
                return Err(Tensor0Error::Message(
                    "indices require unique values".to_string(),
                ));
            }
        }
        Ok(Self { inner })
    }

    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = &K> + '_ {
        self.inner.iter()
    }

    pub(super) fn index_of(&self, value: &K) -> Option<usize> {
        self.inner.get_index_of(value)
    }

    pub(super) fn len(&self) -> usize {
        self.inner.len()
    }

    pub(super) fn get_index(&self, index: usize) -> Option<&K> {
        self.inner.get_index(index)
    }
}

impl<K: Eq + Hash> PartialEq for Indices<K> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<K: Eq + Hash> Eq for Indices<K> {}

#[cfg(test)]
mod tests {
    use super::Indices;

    #[test]
    fn indices_iterate_and_lookup_in_canonical_order() {
        let indices = Indices::new(vec![2, 0, 1]).unwrap();

        assert_eq!(indices.len(), 3);
        assert_eq!(indices.iter().copied().collect::<Vec<_>>(), vec![2, 0, 1]);
        assert_eq!(indices.get_index(0), Some(&2));
        assert_eq!(indices.get_index(2), Some(&1));
        assert_eq!(indices.get_index(3), None);
        assert_eq!(indices.index_of(&0), Some(1));
        assert_eq!(indices.index_of(&3), None);
    }

    #[test]
    fn indices_equality_is_order_sensitive() {
        let left = Indices::new(vec![0, 1]).unwrap();
        let right = Indices::new(vec![1, 0]).unwrap();

        assert_ne!(left, right);
    }

    #[test]
    fn indices_reject_duplicate_values() {
        let err = Indices::new(vec![0, 1, 0]).unwrap_err();

        assert_eq!(err.to_string(), "indices require unique values");
    }
}
