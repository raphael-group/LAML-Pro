#ifndef ULTRAMETRIC_H
#define ULTRAMETRIC_H

#include "phylogeny.h"

/* 
 * Project branch lengths b onto ultrametric set by solving the optimization 
 * problem:
 *               minimize_{c, b'}    ||b - b'||^2
 *                           s.t.    A_T * b' = c\mathbb{1},
 *                                   b' >= 0
 * where A_T is a num_branches * num_leaves matrix such that A_T[i][j] = 1
 * if branch i is on the path from leaf j to the root, and 0 otherwise. A_T * b'
 * is the vector of root-to-leaf distances, and c is the distance from the root to
 * the leaves.
*/
void ultrametric_projection(tree& t);

#endif