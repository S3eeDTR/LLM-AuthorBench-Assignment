#include <stdio.h>
#include <stdlib.h> // For malloc, free, exit
#include <math.h>   // For fabs (absolute value)

#define MAX_EQUATIONS 7
#define EPSILON 1e-9 // A small value to check for near-zero

// Function to print the augmented matrix (for debugging/visualization)
void print_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= n; j++) { // j <= n because of the augmented column
            printf("%10.4f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Function to perform Gauss-Jordan elimination
// matrix: The augmented matrix (n x n+1)
// n: The number of equations (and variables)
int gauss_jordan(double **matrix, int n) {
    // Forward Elimination
    for (int i = 0; i < n; i++) { // Pivot row
        // Find pivot (largest absolute value in the current column below/at the pivot)
        int pivot_row = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(matrix[k][i]) > fabs(matrix[pivot_row][i])) {
                pivot_row = k;
            }
        }

        // Swap rows if necessary (if pivot is not in current row i)
        if (pivot_row != i) {
            double *temp = matrix[i];
            matrix[i] = matrix[pivot_row];
            matrix[pivot_row] = temp;
        }

        // Check if the pivot element is zero or very close to zero
        if (fabs(matrix[i][i]) < EPSILON) {
            // This indicates either no unique solution (infinite solutions or no solution)
            // For simplicity, we'll return an error code here.
            // A more robust solution would check for consistency.
            fprintf(stderr, "Error: Singular matrix or no unique solution detected.\n");
            return 0; // Return 0 for failure
        }

        // Normalize the pivot row (make pivot element 1)
        double pivot_val = matrix[i][i];
        for (int j = i; j <= n; j++) {
            matrix[i][j] /= pivot_val;
        }

        // Eliminate
        for (int k = 0; k < n; k++) { // For each other row
            if (k != i) { // Don't eliminate from the pivot row itself
                double factor = matrix[k][i];
                for (int j = i; j <= n; j++) {
                    matrix[k][j] -= factor * matrix[i][j];
                }
            }
        }
    }
    return 1; // Return 1 for success
}

int main() {
    int num_equations = 7;

    // Allocate memory for the augmented matrix
    // matrix[i][j] where i is row, j is column
    // The last column (index num_equations) will store the constants (b_i)
    double **matrix = (double **)malloc(num_equations * sizeof(double *));
    if (matrix == NULL) {
        perror("Failed to allocate memory for rows");
        return 1;
    }
    for (int i = 0; i < num_equations; i++) {
        matrix[i] = (double *)malloc((num_equations + 1) * sizeof(double));
        if (matrix[i] == NULL) {
            perror("Failed to allocate memory for columns");
            // Free previously allocated rows before exiting
            for (int k = 0; k < i; k++) {
                free(matrix[k]);
            }
            free(matrix);
            return 1;
        }
    }

    // --- Example 1: A system with a unique solution ---
    // x1 + x2 + x3 + x4 + x5 + x6 + x7 = 28
    // 2x1 + x2 + x3 + x4 + x5 + x6 + x7 = 31
    // 3x1 + 2x2 + x3 + x4 + x5 + x6 + x7 = 35
    // 4x1 + 3x2 + 2x3 + x4 + x5 + x6 + x7 = 40
    // 5x1 + 4x2 + 3x3 + 2x4 + x5 + x6 + x7 = 46
    // 6x1 + 5x2 + 4x3 + 3x4 + 2x5 + x6 + x7 = 53
    // 7x1 + 6x2 + 5x3 + 4x4 + 3x5 + 2x6 + x7 = 61

    // Expected solution for this setup:
    // x1 = 3
    // x2 = 4
    // x3 = 5
    // x4 = 6
    // x5 = 7
    // x6 = 1
    // x7 = 2

    // Initialize the matrix with coefficients and constants
    // You would typically get these values from user input or a file
    matrix[0][0] = 1; matrix[0][1] = 1; matrix[0][2] = 1; matrix[0][3] = 1; matrix[0][4] = 1; matrix[0][5] = 1; matrix[0][6] = 1; matrix[0][7] = 28;
    matrix[1][0] = 2; matrix[1][1] = 1; matrix[1][2] = 1; matrix[1][3] = 1; matrix[1][4] = 1; matrix[1][5] = 1; matrix[1][6] = 1; matrix[1][7] = 31;
    matrix[2][0] = 3; matrix[2][1] = 2; matrix[2][2] = 1; matrix[2][3] = 1; matrix[2][4] = 1; matrix[2][5] = 1; matrix[2][6] = 1; matrix[2][7] = 35;
    matrix[3][0] = 4; matrix[3][1] = 3; matrix[3][2] = 2; matrix[3][3] = 1; matrix[3][4] = 1; matrix[3][5] = 1; matrix[3][6] = 1; matrix[3][7] = 40;
    matrix[4][0] = 5; matrix[4][1] = 4; matrix[4][2] = 3; matrix[4][3] = 2; matrix[4][4] = 1; matrix[4][5] = 1; matrix[4][6] = 1; matrix[4][7] = 46;
    matrix[5][0] = 6; matrix[5][1] = 5; matrix[5][2] = 4; matrix[5][3] = 3; matrix[5][4] = 2; matrix[5][5] = 1; matrix[5][6] = 1; matrix[5][7] = 53;
    matrix[6][0] = 7; matrix[6][1] = 6; matrix[6][2] = 5; matrix[6][3] = 4; matrix[6][4] = 3; matrix[6][5] = 2; matrix[6][6] = 1; matrix[6][7] = 61;


    printf("Original Augmented Matrix:\n");
    print_matrix(matrix, num_equations);

    if (gauss_jordan(matrix, num_equations)) {
        printf("Matrix after Gauss-Jordan Elimination:\n");
        print_matrix(matrix, num_equations);

        printf("Solutions:\n");
        for (int i = 0; i < num_equations; i++) {
            printf("x%d = %f\n", i + 1, matrix[i][num_equations]);
        }
    } else {
        printf("Failed to find a unique solution.\n");
    }

    // Free allocated memory
    for (int i = 0; i < num_equations; i++) {
        free(matrix[i]);
    }
    free(matrix);

    return 0;
}