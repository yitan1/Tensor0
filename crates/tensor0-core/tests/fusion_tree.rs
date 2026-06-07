use ndarray::{ArrayD, Dimension};
use tensor0_core::error::Result;
use tensor0_core::fusion_tree::{fusiontree_pair_tensor, fusiontree_tensor, FusionTree};
use tensor0_core::sector::{SU2Irrep, Sector};

fn su2(spin2: i64) -> SU2Irrep {
    SU2Irrep::spin2(spin2).unwrap()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}",
    );
}

fn fusion_tree<I: Sector>(
    uncoupled: Vec<I>,
    coupled: I,
    is_dual: Vec<bool>,
    innerlines: Vec<I>,
    vertices: Vec<usize>,
) -> FusionTree<I> {
    FusionTree::new(uncoupled, coupled, is_dual, innerlines, vertices).unwrap()
}

fn assert_sparse_tensor(tensor: &ArrayD<f64>, shape: &[usize], entries: &[(&[usize], f64)]) {
    assert_eq!(tensor.shape(), shape);
    for (index, actual) in tensor.indexed_iter() {
        let expected = entries
            .iter()
            .find_map(|&(entry_index, value)| (entry_index == index.slice()).then_some(value))
            .unwrap_or(0.0);
        assert_close(*actual, expected);
    }
}

#[test]
fn empty_fusiontree_tensor_is_unit_channel_scalar() -> Result<()> {
    let tree = fusion_tree(vec![], su2(0), vec![], vec![], vec![]);

    let tensor = fusiontree_tensor(&tree)?;

    assert_sparse_tensor(&tensor, &[1], &[(&[0], 1.0)]);
    Ok(())
}

#[test]
fn one_leg_dual_fusiontree_tensor_uses_z_isomorphism() -> Result<()> {
    let half = su2(1);
    let tree = fusion_tree(vec![half.clone()], half, vec![true], vec![], vec![]);

    let tensor = fusiontree_tensor(&tree)?;

    assert_sparse_tensor(&tensor, &[2, 2], &[(&[0, 1], 1.0), (&[1, 0], -1.0)]);
    Ok(())
}

#[test]
fn two_leg_fusiontree_tensor_selects_vertex_slice() -> Result<()> {
    let half = su2(1);
    let tree = fusion_tree(
        vec![half.clone(), half],
        su2(0),
        vec![false, false],
        vec![],
        vec![0],
    );

    let tensor = fusiontree_tensor(&tree)?;

    assert_sparse_tensor(
        &tensor,
        &[2, 2, 1],
        &[
            (&[0, 1, 0], 1.0 / 2.0_f64.sqrt()),
            (&[1, 0, 0], -1.0 / 2.0_f64.sqrt()),
        ],
    );
    Ok(())
}

#[test]
fn two_leg_fusiontree_tensor_applies_second_dual_leg_in_place() -> Result<()> {
    let half = su2(1);
    let tree = fusion_tree(
        vec![half.clone(), half],
        su2(0),
        vec![false, true],
        vec![],
        vec![0],
    );

    let tensor = fusiontree_tensor(&tree)?;

    assert_sparse_tensor(
        &tensor,
        &[2, 2, 1],
        &[
            (&[0, 0, 0], 1.0 / 2.0_f64.sqrt()),
            (&[1, 1, 0], 1.0 / 2.0_f64.sqrt()),
        ],
    );
    Ok(())
}

#[test]
fn multi_leg_fusiontree_tensor_contracts_innerlines() -> Result<()> {
    let half = su2(1);
    let tree = fusion_tree(
        vec![half.clone(), half.clone(), half.clone(), half],
        su2(0),
        vec![false, false, false, false],
        vec![su2(0), su2(1)],
        vec![0, 0, 0],
    );

    let tensor = fusiontree_tensor(&tree)?;

    assert_sparse_tensor(
        &tensor,
        &[2, 2, 2, 2, 1],
        &[
            (&[0, 1, 0, 1, 0], 0.5),
            (&[0, 1, 1, 0, 0], -0.5),
            (&[1, 0, 0, 1, 0], -0.5),
            (&[1, 0, 1, 0, 0], 0.5),
        ],
    );
    Ok(())
}

#[test]
fn fusiontree_pair_tensor_contracts_shared_coupled_axis() -> Result<()> {
    let half = su2(1);
    let row = fusion_tree(vec![half.clone()], half, vec![false], vec![], vec![]);
    let col = row.clone();

    let tensor = fusiontree_pair_tensor(&row, &col)?;

    assert_sparse_tensor(&tensor, &[2, 2], &[(&[0, 0], 1.0), (&[1, 1], 1.0)]);
    Ok(())
}

#[test]
fn fusiontree_pair_tensor_rejects_mismatched_coupled_sector() {
    let half = su2(1);
    let singlet_tree = fusion_tree(
        vec![half.clone(), half.clone()],
        su2(0),
        vec![false, false],
        vec![],
        vec![0],
    );
    let triplet_tree = fusion_tree(
        vec![half.clone(), half],
        su2(2),
        vec![false, false],
        vec![],
        vec![0],
    );

    let err = fusiontree_pair_tensor(&singlet_tree, &triplet_tree).unwrap_err();

    assert!(err.to_string().contains("coupled"));
}

#[test]
fn fusiontree_constructor_rejects_inconsistent_tree_shapes() {
    let half = su2(1);
    let cases = [
        (
            FusionTree::new(vec![half.clone()], half.clone(), vec![], vec![], vec![]),
            "dual flag arity",
        ),
        (
            FusionTree::new(
                vec![half.clone(), half.clone(), half.clone()],
                half.clone(),
                vec![false, false, false],
                vec![],
                vec![0, 0],
            ),
            "innerline arity",
        ),
        (
            FusionTree::new(
                vec![half.clone(), half],
                su2(0),
                vec![false, false],
                vec![],
                vec![],
            ),
            "vertex arity",
        ),
    ];

    for (tree, message) in cases {
        let err = tree.unwrap_err();
        assert!(err.to_string().contains(message));
    }
}
