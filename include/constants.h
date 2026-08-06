#ifndef CONSTANTS_H
#define CONSTANTS_H

#define NEGATIVE_INFINITY (-1e8)

// PARAMETER BOUNDS
#define BRANCH_LENGTH_LB (1e-6)
#define BRANCH_LENGTH_UB (1e4)
#define NU_LB (1e-6)
#define NU_UB (1e3)
#define PHI_LB (1e-6)
#define PHI_UB (1.0 - 1e-6)

// Value nu is pinned to under --no-silencing. Not exactly 0: nu appears in
// log(1 - exp(-nu*b)), which is a domain error at 0. That term is short-circuited
// to NEGATIVE_INFINITY when silencing is off, so this value only enters through
// the negligible -b*nu and -b*(1+nu) terms. Equal to NU_LB.
#define NO_SILENCING_NU (1e-6)

#endif // CONSTANTS_H