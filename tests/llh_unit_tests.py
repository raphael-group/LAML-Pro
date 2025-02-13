import unittest
import jax.numpy as jnp
import networkx as nx

import phylogeny
import laml

def build_build_unit_test(msa, phi, nu, mutation_priors):
    # tree is ((a:1.0,b:1.0):1.0,c:1.0):1.0;
    tree = nx.DiGraph([(4, 3), (4, 0), (3, 1), (3, 2)], root=0)
    branch_lengths = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    character_matrix = jnp.array([[msa[2]], [msa[1]], [msa[0]]])
    mutation_priors = jnp.array([mutation_priors])
    model_parameters = jnp.array([nu, phi])

    root = 4
    num_leaves = 3
    num_characters = 1
    max_alphabet_size = mutation_priors.shape[1]

    phylo = phylogeny.Phylogeny(
        num_leaves=num_leaves,
        num_characters=num_characters,
        max_alphabet_size=max_alphabet_size,
        mutation_priors=mutation_priors,
        character_matrix=character_matrix,
        tree=tree,
        root=root
    )

    phylo_opt = phylogeny.PhylogenyOptimization(
        phylogeny=phylo,
        branch_lengths=branch_lengths,
        model_parameters=model_parameters,
        inside_log_likelihoods=jnp.zeros((num_characters, 2 * num_leaves - 1, max_alphabet_size + 2), dtype=jnp.float32)
    )

    return phylo_opt

class LLHTest(unittest.TestCase):
    def test_unit_1(self):
        phylo_opt = build_build_unit_test([1, 1, 1], 0.0, 0.0, [1.0])
        true_llh  = -0.20665578828621584
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_2(self):
        phylo_opt = build_build_unit_test([1, 1, 0], 0.0, 0.0, [1.0])
        true_llh  = -2.2495946917551692
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_3(self):
        phylo_opt = build_build_unit_test([1, 0, 1], 0.0, 0.0, [1.0])
        true_llh  = -3.917350291274164
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_4(self):
        phylo_opt = build_build_unit_test([0, 1, 1], 0.0, 0.0, [1.0])
        true_llh  = -3.917350291274164
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_5(self):
        phylo_opt = build_build_unit_test([1, 0, 0], 0.0, 0.0, [1.0])
        true_llh  = -4.4586751457870815
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_6(self):
        phylo_opt = build_build_unit_test([0, 1, 0], 0.0, 0.0, [1.0])
        true_llh  = -4.4586751457870815
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_7(self):
        phylo_opt = build_build_unit_test([0, 0, 1], 0.0, 0.0, [1.0])
        true_llh  = -4.4586751457870815
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=3)

    def test_unit_8(self):
        phylo_opt = build_build_unit_test([0, 0, 0], 0.0, 0.0, [1.0])
        true_llh  = -5.0
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=3)

    def test_unit_9(self):
        phylo_opt = build_build_unit_test([0, 0, -1], 0.1, 0.0, [1.0])
        true_llh  = -6.513306124309698
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_10(self):
        phylo_opt = build_build_unit_test([0, -1, 0], 0.1, 0.0, [1.0])
        true_llh  = -6.513306124309698
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_11(self):
        phylo_opt = build_build_unit_test([-1, 0, 0], 0.1, 0.0, [1.0])
        true_llh  = -6.513306124309698
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_12(self):
        phylo_opt = build_build_unit_test([0, 1, -1], 0.1, 0.0, [1.0])
        true_llh  = -5.97198126969678
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_13(self):
        phylo_opt = build_build_unit_test([0, -1, 1], 0.1, 0.0, [1.0])
        true_llh  = -5.97198126969678
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_14(self):
        phylo_opt = build_build_unit_test([-1, 0, 1], 0.1, 0.0, [1.0])
        true_llh  = -5.97198126969678
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_15(self):
        phylo_opt = build_build_unit_test([1, -1, 0], 0.1, 0.0, [1.0])
        true_llh  = -4.658719582178557
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_16(self):
        phylo_opt = build_build_unit_test([-1, 1, 0], 0.1, 0.0, [1.0])
        true_llh  = -4.658719582178557
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_17(self):
        phylo_opt = build_build_unit_test([1, 1, -1], 0.1, 0.0, [1.0])
        true_llh  = -2.5980566021648364
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_18(self):
        phylo_opt = build_build_unit_test([1, -1, 1], 0.1, 0.0, [1.0])
        true_llh  = -2.695795750497349
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_19(self):
        phylo_opt = build_build_unit_test([-1, 1, 1], 0.1, 0.0, [1.0])
        true_llh  = -2.695795750497349
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

    def test_unit_20(self):
        phylo_opt = build_build_unit_test([1, 1, 1], 0.0, 0.0, [0.5, 0.5])
        true_llh  = -1.0297894223949402
        llh    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh, places=4)

if __name__ == '__main__':
    unittest.main()
