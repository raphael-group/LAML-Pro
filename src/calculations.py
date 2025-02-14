import jax 
import jax.numpy as jnp
import equinox.internal as eqxi

DEPTH = 10
EPS = 1e-6

def compute_Q_v_product(
    v : jnp.array,
    log_mutation_priors : jnp.array,
    alphabet_size : int,
    node_idx : int,
    t1_array : jnp.array,
    t2_array : jnp.array,
    t3_array : jnp.array,
    alpha0_array : jnp.array
) -> jnp.array:
    """
    Computes the vectors u_k = Q_kv_k where Q_k is the 
    transition matrix for the k-th character on the edge
    into node u.
    """

    t1 = t1_array[node_idx]
    t2 = t2_array[node_idx]
    t3 = t3_array[node_idx]
    alpha0_val = alpha0_array[node_idx]
    
    # case alpha = 0
    out = v.at[:, 0].set(alpha0_val + v[:, 0])

    # case alpha = -1
    log_a = t2 + jax.nn.logsumexp(v[:, :-1], axis=1) # shape (num_characters,)
    log_b = v[:, -1] # shape (num_characters,)
    out = out.at[:, -1].set(jnp.logaddexp(log_a, log_b)) 

    # case alpha = 1, ..., A - 1
    log_0 = log_mutation_priors + t1 + t3 + v[:, 0][:, None] # shape (num_characters, alphabet_size - 2)
    log_alpha = t1 + v[:, 1:-1] # shape (num_characters, alphabet_size - 2)
    out = out.at[:, 1:-1].set(jnp.logaddexp(log_0, log_alpha))

    return out

def compute_Q_transpose_v_product(
    v : jnp.array,
    log_mutation_priors : jnp.array,
    alphabet_size : int,
    node_idx : int,
    t1_array : jnp.array,
    t2_array : jnp.array,
    t3_array : jnp.array,
    alpha0_array : jnp.array
) -> jnp.array:
    """
    Computes the vectors u_k = Q_k^Tv_k where Q_k is the 
    transition matrix for the k-th character on the edge
    into node u.
    """

    t1 = t1_array[node_idx]
    t2 = t2_array[node_idx]
    t3 = t3_array[node_idx]
    alpha0_val = alpha0_array[node_idx]
    
    summands = jnp.zeros_like(v)
    summands = summands.at[:, 0].set(alpha0_val)
    if alphabet_size > 2:
        summands = summands.at[:, 1:-1].set(t1 + log_mutation_priors + t3)
    summands = summands.at[:, -1].set(t2)

    val_for_alpha0 = jax.nn.logsumexp(summands + v, axis=1)

    idx = jnp.arange(alphabet_size)[None, :]  # shape (1, alphabet_size)
    out = jnp.where(
        idx == 0,
        val_for_alpha0[:, None],
        jnp.where(
            idx == (alphabet_size - 1),
            v[:, -1, None],
            jnp.logaddexp(t1 + v, t2 + v[:, -1, None])
        )
    )
    return out
    
def compute_single_edge_outside_ll(
    outside_ll : jnp.array,
    edge_inside_ll : jnp.array,
    log_mutation_priors : jnp.array,
    alphabet_size : int,
    node_idx : int,
    parent_idx : int,
    sibling_idx : int,
    t1_array : jnp.array,
    t2_array : jnp.array,
    t3_array : jnp.array,
    alpha0_array : jnp.array
) -> jnp.array:
    return compute_Q_v_product(
        outside_ll[:, parent_idx, :] + edge_inside_ll[:, sibling_idx, :], 
        log_mutation_priors, alphabet_size, node_idx, 
        t1_array, t2_array, t3_array, alpha0_array
    )

def compute_single_edge_inside_ll(
    inside_ll : jnp.array,
    child_idx  : int,
    log_mutation_priors : jnp.array,
    alphabet_size : int,
    t1_array : jnp.array,
    t2_array : jnp.array,
    t3_array : jnp.array,
    alpha0_array : jnp.array
) -> jnp.array:
    """
    Computes the edge inside log-likelihood P^c((u, v) | a) for all
    characters c and states a provided the log-likelihood of v.
    """
    return compute_Q_transpose_v_product(
        inside_ll[:, child_idx, :], 
        log_mutation_priors, alphabet_size, child_idx, 
        t1_array, t2_array, t3_array, alpha0_array
    )

def compute_outside_log_likelihoods(
    edge_inside_log_likelihoods: jnp.array,
    parent_sibling: jnp.array,
    branch_lengths: jnp.array,
    level_order: jnp.array,
    model_parameters: jnp.array,
    mutation_priors: jnp.array,
    root: int,
):
    """Computes the outside log-likelihoods for internal nodes via a depth-based traversal strategy."""

    depth = DEPTH
    ν = model_parameters[0]
    num_characters, num_nodes, alphabet_size = edge_inside_log_likelihoods.shape

    log_mutation_priors = jnp.log(mutation_priors)

    minus_blen_times_nu = -branch_lengths * ν
    t1_array = minus_blen_times_nu
    t2_array = jnp.log(jnp.where(-minus_blen_times_nu < EPS, EPS, 1.0 - jnp.exp(minus_blen_times_nu)))
    t3_array = jnp.log(jnp.where(branch_lengths < EPS, EPS, 1.0 - jnp.exp(-branch_lengths)))
    alpha0_array = -branch_lengths * (1.0 + ν) # for alpha=0 summand

    def body_fun(u, outside_ll):
        w, v = parent_sibling[u]

        res = compute_single_edge_outside_ll(
            outside_ll, edge_inside_log_likelihoods, log_mutation_priors, alphabet_size, 
            u, w, v, t1_array, t2_array, t3_array, alpha0_array
        )

        return res

    i_vector = jnp.arange(num_nodes)
    update_f = lambda outside_ll: jax.vmap(lambda i: body_fun(i, outside_ll))

    def scan_body(carry, d):
        outside_ll = carry
        cond = (level_order == d)[None, :, None]

        all_updates = jnp.where(
            cond,
            update_f(outside_ll)(i_vector).transpose(1, 0, 2),
            outside_ll
        )

        return all_updates, None

    # initialize root outside log-likelihood
    outside_log_likelihoods = jnp.zeros((num_characters, num_nodes, alphabet_size))
    outside_log_likelihoods = outside_log_likelihoods.at[:, root, 0].set(alpha0_array[root])
    outside_log_likelihoods = outside_log_likelihoods.at[:, root, -1].set(t2_array[root])
    outside_log_likelihoods = outside_log_likelihoods.at[:, root, 1:-1].set(t1_array[root] + log_mutation_priors + t3_array[root])

    outside_log_likelihoods, _ = jax.lax.scan(
        scan_body, outside_log_likelihoods, jnp.arange(1, depth + 1), #kind="checkpointed", buffers=lambda B: B, checkpoints="all"
    )

    return outside_log_likelihoods

def compute_internal_log_likelihoods(
    inside_log_likelihoods: jnp.array,
    internal_postorder: jnp.array,
    internal_postorder_children: jnp.array,
    branch_lengths: jnp.array,
    model_parameters: jnp.array,
    mutation_priors: jnp.array,
    root: int,
):
    """Computes log-likelihoods for internal nodes via a depth-based traversal strategy.

    Args:
        inside_log_likelihoods: Array of shape (num_characters, num_nodes, alphabet_size)
            storing current log-likelihoods; will be updated for internal nodes.
        internal_postorder: Array of shape (num_internal_nodes, 2) where each row is 
            (node_id, depth) specifying post-order traversal order and depth information.
        internal_postorder_children: Array of shape (num_internal_nodes, 2) where each row 
            contains the child node indices for the corresponding internal node in internal_postorder.
        branch_lengths: Array of shape (num_nodes,) storing branch lengths for each node.
        model_parameters: Array with [ν, ϕ] (heritable silencing rate, sequencing dropout rate).
        mutation_priors: Array of shape (alphabet_size,) with prior probabilities for mutations.
        root: Integer index of the root node.

    Returns:
        Tuple containing:
        - Updated inside_log_likelihoods array with internal node values.
        - Array of shape (num_characters, alphabet_size) with root node log-likelihoods.

    Notes:
        Uses a depth-based traversal strategy and JAX vectorization for efficient 
        computation, taking O(num_characters * num_nodes * alphabet_size * depth) time,
        rather than the standard O(num_characters * num_nodes * alphabet_size) time, but 
        performs each of the O(depth) steps completely in parallel.
    """

    depth = DEPTH
    ν = model_parameters[0]
    num_characters, num_nodes, alphabet_size = inside_log_likelihoods.shape

    log_mutation_priors = jnp.log(mutation_priors)

    minus_blen_times_nu = -branch_lengths * ν
    t1_array = minus_blen_times_nu
    t2_array = jnp.log(jnp.where(-minus_blen_times_nu < EPS, EPS, 1.0 - jnp.exp(minus_blen_times_nu)))
    t3_array = jnp.log(jnp.where(branch_lengths < EPS, EPS, 1.0 - jnp.exp(-branch_lengths)))
    alpha0_array = -branch_lengths * (1.0 + ν)  # for alpha=0 summand

    def body_fun(i, inside_ll):
        u = internal_postorder[i, 0]
        v, w = internal_postorder_children[i]
        llh_v = compute_single_edge_inside_ll(
            inside_ll, v, log_mutation_priors, alphabet_size, t1_array, t2_array, t3_array, alpha0_array
        )
        llh_w = compute_single_edge_inside_ll(
            inside_ll, w, log_mutation_priors, alphabet_size, t1_array, t2_array, t3_array, alpha0_array
        )
        return llh_v + llh_w

    i_vector = jnp.arange(internal_postorder.shape[0])
    update_f = lambda inside_ll: jax.vmap(lambda i: body_fun(i, inside_ll))

    def scan_body(carry, d):
        inside_ll = carry
        cond = (internal_postorder[:, 1] == depth - d)[:, None, None]

        all_updates = jnp.where(
            cond,
            update_f(inside_ll)(i_vector),
            inside_ll[:, internal_postorder[:, 0], :].transpose(1, 0, 2)
        )

        return inside_ll.at[:, internal_postorder[:,0], :].set(all_updates.transpose(1, 0, 2)), None

    # Using Equinox's internal scan to accumulate the result
    # due to an XLA bug with JAX's scan: 
    #      https://github.com/jax-ml/jax/issues/10197
    inside_log_likelihoods, _ = eqxi.scan(
        scan_body, inside_log_likelihoods, jnp.arange(depth + 1), kind="checkpointed", buffers=lambda B: B, checkpoints="all"
    )
    inside_root_llh = compute_single_edge_inside_ll(
        inside_log_likelihoods, root, log_mutation_priors, alphabet_size, t1_array, t2_array, t3_array, alpha0_array
    )
    return inside_log_likelihoods, inside_root_llh

def initialize_leaf_inside_log_likelihoods(
    inside_log_likelihoods : jnp.array, 
    leaves : jnp.array,
    model_parameters : jnp.array,
    character_matrix : jnp.array
) -> jnp.array:
    """Initializes leaf node log-likelihoods based on observed character states.

    Args:
        inside_log_likelihoods: Array of shape (num_characters, num_nodes, alphabet_size)
            to be populated with leaf node likelihoods.
        leaves: Array of leaf node indices to initialize.
        model_parameters: Array with [ν, ϕ] (heritable silencing rate, sequencing dropout rate).
        character_matrix: Array of shape (num_leaves, num_characters) storing observed states,
            where values are in {0 (missing), 1,..., A-1 (valid states), -1 (unknown)}.

    Returns:
        Array of same shape as inside_log_likelihoods with leaf log-likelihoods initialized.
    """

    num_characters = inside_log_likelihoods.shape[0]
    alphabet_size  = inside_log_likelihoods.shape[2]
    ϕ = model_parameters[1]
    
    leaf_characters = character_matrix[leaves, :]
    
    alpha_grid = jnp.arange(alphabet_size)
    
    leaf_chars_expanded = leaf_characters[..., None] # shape (L, C, 1)
    alpha_expanded      = alpha_grid[None, None, :]  # shape (1, 1, A)

    cond_1 = ((alpha_expanded == (alphabet_size - 1)) & (leaf_chars_expanded == -1)) # (1) alpha==A-1 & char==-1
    cond_2 = ((alpha_expanded == (alphabet_size - 1)) & (leaf_chars_expanded != -1)) # (2) alpha==A-1 & char!=-1
    cond_3 = (alpha_expanded == leaf_chars_expanded)                                 # (3) alpha==char
    cond_4 = (leaf_chars_expanded == -1)                                             # (4) char==-1           
    
    conditions = [cond_1, cond_2, cond_3, cond_4]
    choices    = [1.0, 0.0, 1.0 - ϕ, ϕ]
    choices    = jnp.maximum(jnp.array(choices), EPS)
    
    leaf_inside_probs = jnp.select(conditions, choices, default=0.0)
    leaf_inside_probs_T = jnp.swapaxes(leaf_inside_probs, 0, 1)  # now (C, L, A)
    
    inside_log_likelihoods = EPS * jnp.ones_like(inside_log_likelihoods) 
    inside_log_likelihoods = inside_log_likelihoods.at[:, leaves, :].set(
        leaf_inside_probs_T
    )
    
    return jnp.log(inside_log_likelihoods)

def compute_edge_inside_log_likelihoods(
    branch_lengths : jnp.array,
    mutation_priors : jnp.array,
    inside_log_likelihoods : jnp.array,
    model_parameters : jnp.array
) -> jnp.array:
    """Computes the edge inside log-likelihoods for all edges in the phylogeny. 

    Args:
        branch_lengths: Array of shape (num_nodes,) with branch lengths.
        mutation_priors: Array of shape (alphabet_size,) with mutation priors.
        inside_log_likelihoods: Pre-allocated array for likelihood computations.
        model_parameters: Array with [ν, ϕ] (heritable silencing rate, sequencing dropout rate).
    Returns:
        An array edge_inside_log_likelihoods with shape (num_characters, 2 * num_nodes - 1, alphabet_size) 
        containing the edge inside log-likelihoods, where edge_inside_log_likelihoods[c, v, a] is the
        edge inside log-likelihood for character c, edge (u, v), with state a.
    """
    num_nodes = branch_lengths.shape[0]
    alphabet_size  = inside_log_likelihoods.shape[2]
    ν = model_parameters[0]
    log_mutation_priors = jnp.log(mutation_priors)

    minus_blen_times_nu = -branch_lengths * ν
    t1_array = minus_blen_times_nu
    t2_array = jnp.log(jnp.where(-minus_blen_times_nu < EPS, EPS, 1.0 - jnp.exp(minus_blen_times_nu)))
    t3_array = jnp.log(jnp.where(branch_lengths < EPS, EPS, 1.0 - jnp.exp(-branch_lengths)))
    alpha0_array = -branch_lengths * (1.0 + ν)  # for alpha=0 summand

    edge_inside_log_likelihoods = jax.vmap(
        lambda node_idx: compute_single_edge_inside_ll(
            inside_log_likelihoods, node_idx, log_mutation_priors, alphabet_size, t1_array, t2_array, t3_array, alpha0_array
        )
    )(jnp.arange(num_nodes))

    return edge_inside_log_likelihoods.transpose(1, 0, 2)

def compute_log_likelihood(
    branch_lengths : jnp.array,
    mutation_priors : jnp.array,
    leaves : jnp.array,
    internal_postorder : jnp.array,
    internal_postorder_children : jnp.array,
    parent_sibling : jnp.array,
    level_order : jnp.array,
    inside_log_likelihoods : jnp.array,
    model_parameters : jnp.array,
    character_matrix : jnp.array,
    root : int
) -> jnp.array:
    """Computes the total log-likelihood of observed character matrix given 
    a phylogeny with branch lengths and model parameters.

    Args:
        branch_lengths: Array of shape (num_nodes,) with branch lengths.
        mutation_priors: Array of shape (alphabet_size,) with mutation priors.
        leaves: Array of leaf node indices.
        internal_postorder: Post-order traversal info for internal nodes.
        internal_postorder_children: Child mappings for internal nodes.
        inside_log_likelihoods: Pre-allocated array for likelihood computations.
        model_parameters: Array with [ν, ϕ] (heritable silencing rate, sequencing dropout rate).
        character_matrix: Observed states array of shape (num_leaves, num_characters).
        root: Root node index.

    Returns:
        Scalar sum of log-likelihoods over all characters at the root node.
    """

    inside_log_likelihoods = initialize_leaf_inside_log_likelihoods(
        inside_log_likelihoods, 
        leaves, 
        model_parameters, 
        character_matrix
    )

    inside_log_likelihoods, inside_root_llh = compute_internal_log_likelihoods(
        inside_log_likelihoods, 
        internal_postorder,
        internal_postorder_children,
        branch_lengths,
        model_parameters,
        mutation_priors,
        root
    )

    edge_inside_log_likelihoods = compute_edge_inside_log_likelihoods(
        branch_lengths, 
        mutation_priors, 
        inside_log_likelihoods, 
        model_parameters
    )

    outside_log_likelihoods = compute_outside_log_likelihoods(
        edge_inside_log_likelihoods, 
        parent_sibling, 
        branch_lengths, 
        level_order, 
        model_parameters, 
        mutation_priors, 
        root
    )
    
    likelihoods = jax.nn.logsumexp(outside_log_likelihoods + inside_log_likelihoods, axis=2)
    error = likelihoods - inside_root_llh[:, 0][:, None]
    assert jnp.allclose(error, 0.0, atol=1e-3), f"Error in log-likelihood computation: {jnp.max(jnp.abs(error))}"

    return inside_root_llh[:, 0].sum()