#ifndef VERIFY_UTIL_H
#define VERIFY_UTIL_H

#include "graph_utils.h"

#include <stdio.h>
#include <stddef.h>

/* Compare expected and actual (both n*n row-major). Returns number of mismatches.
 * If max_print > 0 and there are mismatches, prints the first max_print (row,col) with
 * expected and actual values to stderr. */
static inline int verify_matrices(const int *expected, const int *actual,
                                  size_t n, int max_print) {
    int mismatches = 0;
    int printed = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const size_t idx = i * n + j;
            if (expected[idx] != actual[idx]) {
                ++mismatches;
                if (max_print > 0 && printed < max_print) {
                    fprintf(stderr, "  mismatch at (%zu,%zu): expected=%d actual=%d\n",
                            i, j, expected[idx], actual[idx]);
                    ++printed;
                }
            }
        }
    }
    return mismatches;
}

#endif /* VERIFY_UTIL_H */
