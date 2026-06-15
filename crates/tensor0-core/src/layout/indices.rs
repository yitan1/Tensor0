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

#[derive(Clone, Copy, Debug)]
pub(crate) struct IndexedMappingRef<'a, K: Eq + Hash, V> {
    keys: &'a Indices<K>,
    values: &'a [V],
}

impl<'a, K: Eq + Hash, V> IndexedMappingRef<'a, K, V> {
    pub(super) fn new(keys: &'a Indices<K>, values: &'a [V]) -> Result<Self> {
        if keys.len() != values.len() {
            return Err(Tensor0Error::Message(
                "indexed mapping requires matching key and value lengths".to_string(),
            ));
        }
        Ok(Self { keys, values })
    }

    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = (&K, &V)> + '_ {
        self.keys.iter().zip(self.values.iter())
    }

    pub(crate) fn index_of(&self, key: &K) -> Option<usize> {
        self.keys.index_of(key)
    }

    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.index_of(key).and_then(|index| self.values.get(index))
    }
}

#[cfg(test)]
mod tests {
    use super::{IndexedMappingRef, Indices};

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

    #[test]
    fn indexed_mapping_ref_binds_indices_to_aligned_values() {
        let keys = Indices::new(vec![2, 0, 1]).unwrap();
        let values = vec!["two", "zero", "one"];
        let mapping = IndexedMappingRef::new(&keys, &values).unwrap();

        assert_eq!(mapping.len(), 3);
        assert_eq!(
            mapping
                .iter()
                .map(|(key, value)| (*key, *value))
                .collect::<Vec<_>>(),
            vec![(2, "two"), (0, "zero"), (1, "one")],
        );
        assert_eq!(mapping.index_of(&0), Some(1));
        assert_eq!(mapping.index_of(&3), None);
        assert_eq!(mapping.get(&1), Some(&"one"));
        assert_eq!(mapping.get(&3), None);
    }

    #[test]
    fn indexed_mapping_ref_rejects_mismatched_lengths() {
        let keys = Indices::new(vec![0, 1]).unwrap();
        let values = vec!["zero"];

        let err = IndexedMappingRef::new(&keys, &values).unwrap_err();

        assert_eq!(
            err.to_string(),
            "indexed mapping requires matching key and value lengths",
        );
    }
}
