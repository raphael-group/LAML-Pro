#include "osqp.h"
#include "ultrametric.h"

std::tuple<std::vector<OSQPInt>, std::vector<OSQPInt>, std::vector<OSQPFloat>> triplet_format_to_csc(
    const std::vector<OSQPInt>& A_i,
    const std::vector<OSQPInt>& A_j,
    const std::vector<OSQPFloat>& A_x,
    OSQPInt n, 
    OSQPInt m
) {
    OSQPInt nnz = A_i.size();
    
    std::vector<std::tuple<OSQPInt, OSQPInt, OSQPFloat>> triplets;
    triplets.reserve(nnz);
    
    for (OSQPInt k = 0; k < nnz; ++k) {
        triplets.emplace_back(A_j[k], A_i[k], A_x[k]);
    }
    
    std::sort(triplets.begin(), triplets.end());
    
    std::vector<OSQPInt> csc_row_ind(nnz);
    std::vector<OSQPFloat> csc_values(nnz);
    std::vector<OSQPInt> csc_col_ptr(n + 1, 0);
    
    for (OSQPInt k = 0; k < nnz; ++k) {
        OSQPInt j = std::get<0>(triplets[k]);
        csc_row_ind[k] = std::get<1>(triplets[k]);
        csc_values[k] = std::get<2>(triplets[k]);
        csc_col_ptr[j + 1]++;
    }
    
    for (OSQPInt j = 0; j < n; ++j) {
        csc_col_ptr[j + 1] += csc_col_ptr[j];
    }
    
    csc_col_ptr[n] = nnz;

    return {csc_row_ind, csc_col_ptr, csc_values};
}

void ultrametric_projection(tree& t) {
    OSQPInt n = t.num_nodes + 1;            // n = # branches + 1
    OSQPInt m = t.num_leaves + t.num_nodes; // m = # leaves + # branches

    std::vector<OSQPInt> P_i;
    std::vector<OSQPInt> P_j;
    std::vector<OSQPFloat> P_x;

    for (size_t i = 0; i < t.num_nodes; ++i) {
        P_i.push_back(i);
        P_j.push_back(i);
        P_x.push_back(1.0);
    }
    
    auto [P_csc_row_ind, P_csc_col_ptr, P_csc_values] = triplet_format_to_csc(P_i, P_j, P_x, n, n);

    OSQPFloat* q = new OSQPFloat[n];
    for (int node_id : t.tree.nodes()) {
        int node = t.tree[node_id].data;
        q[node] = -t.branch_lengths[node];
    }
    q[n - 1] = 0.0;

    /* 
     * Construct the sparse constraint l <= Ax <= u matrix in 
     * triplet format.
     */
    std::vector<OSQPInt>   A_i;
    std::vector<OSQPInt>   A_j;
    std::vector<OSQPFloat> A_x;
    std::vector<OSQPFloat> l;
    std::vector<OSQPFloat> u;

    /* Add the constraints A_T * b' = c\mathbb{1} */
    int leaf_constraint_id = 0;
    for (int node_id : t.tree.nodes()) { // loops takes O(n^2) in the worst case
        if (t.tree.out_degree(node_id) != 0) {
            continue;
        }

        int current_id = node_id;
        while (true) {
            int current_node = t.tree[current_id].data;
            A_i.push_back(leaf_constraint_id);
            A_j.push_back(current_node);
            A_x.push_back(1.0);

            if (current_id == (int) t.root_id) {
                break;
            }

            current_id = t.tree.predecessors(current_id)[0];
        } 

        A_i.push_back(leaf_constraint_id);
        A_j.push_back(n - 1);
        A_x.push_back(-1.0);
        l.push_back(0.0);
        u.push_back(0.0);
        leaf_constraint_id++;        
    }

    /* Add the constraints b' >= 0 */
    int non_negative_constraint_id = leaf_constraint_id;
    for (int node_id : t.tree.nodes()) {
        A_i.push_back(non_negative_constraint_id);
        A_j.push_back(t.tree[node_id].data);
        A_x.push_back(1.0);
        l.push_back(0.0);
        u.push_back(1e5); // big M (should be infinity)
        non_negative_constraint_id++;
    }

    /* Convert triplet format to CSC format */
    auto [A_csc_row_ind, A_csc_col_ptr, A_csc_values] = triplet_format_to_csc(A_i, A_j, A_x, n, m);

    OSQPInt exitflag = 0;
    OSQPSolver *solver;
    OSQPCscMatrix* P = OSQPCscMatrix_new(n, n, P_i.size(), P_csc_values.data(), P_csc_row_ind.data(), P_csc_col_ptr.data());
    OSQPCscMatrix* A = OSQPCscMatrix_new(m, n, A_i.size(), A_csc_values.data(), A_csc_row_ind.data(), A_csc_col_ptr.data());
    OSQPSettings *settings = OSQPSettings_new();

    exitflag = osqp_setup(&solver, P, q, A, l.data(), u.data(), m, n, settings);
    if (exitflag != 0) {
        throw std::runtime_error("OSQP setup failed");
    }

    exitflag = osqp_solve(solver);
    if (exitflag != 0) {
        throw std::runtime_error("OSQP solve failed");
    }

    OSQPFloat* solution = solver->solution->x;
    for (int node_id : t.tree.nodes()) {
        int node = t.tree[node_id].data;
        t.branch_lengths[node] = solution[node];
    }
}