import unittest
import jax.numpy as jnp
import networkx as nx

import phylogeny
import laml

def build_build_unit_test(msa, phi, nu):
    tree = nx.DiGraph([(4, 3), (4, 0), (3, 1), (3, 2)], root=0)
    branch_lengths = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    character_matrix = jnp.array([[msa[0]], [msa[1]], [msa[2]]])
    mutation_priors = jnp.array([[1.0]])
    model_parameters = jnp.array([nu, phi])

    root = 4
    num_leaves = 3
    num_characters = 1
    max_alphabet_size = 1

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
    def test_unit_10(self):
        phylo_opt = build_build_unit_test([0, 0, -1], 0.1, 0.0)
        true_llh  = -6.513306124309698
        llh, _    = laml.compute_llh(phylo_opt)
        self.assertAlmostEqual(llh, true_llh)

if __name__ == '__main__':
    unittest.main()


