#include <mpi.h>
#include <stdio.h>
#include <cstdlib>

const int N = 4;

void doWork(int myrank, int nprocs);
inline double getitem(const double* matrix, int i, int j, int n)
{
    return matrix[i * n + j];
}
inline void setitem(double* matrix, int i, int j, int n, double value)
{
    matrix[i * n + j] = value;
}

int main(int argc, char* argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    doWork(myrank, nprocs);

    if (myrank == 0) {
        system("pause");
    }
    MPI_Finalize();
    return 0;
}

void doWork(int myrank, int nprocs)
{
    int i, j, k, map[N];
    double a_[N][N];
    double* a = reinterpret_cast<double*>(a_);
    if (myrank == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                auto value = 1.0 / (i + j + 2 - 1);
                setitem(a, i, j, N, value);
            }
        }
    }
    MPI_Bcast(static_cast<void*>(a), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (i = 0; i < N; i++)
        map[i] = i % nprocs;
    for (k = 0; k < N - 1; k++) {
        if (map[k] == myrank) {
            auto divider = getitem(a, k, k, N);
            for (i = k + 1; i < N; i++) {
                auto old_value = getitem(a, k, i, N);
                auto new_value = old_value / divider;
                setitem(a, k, i, N, new_value);
            }
        }
        auto chunk = a + k * N + k + 1;
        MPI_Bcast(chunk, N - k - 1, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
        for (i = k + 1; i < N; i++) {
            if (map[i] == myrank) {
                for (j = k + 1; j < N; j++) {
                    auto multiplier = getitem(a, i, k, N);
                    auto k_row_item = getitem(a, k, j, N);
                    auto old_value = getitem(a, i, j, N);
                    auto new_value = old_value - k_row_item * multiplier;
                    setitem(a, i, j, N, new_value);
                }
            }
        }
    }
    // Printing the entries of the matrix
    for (i = 0; i < N; i++)
        if (map[i] == myrank) {
            printf("%d:\t", i + 1);
            for (j = 0; j < N; j++)
                printf("%lg, ", getitem(a, i, j, N));
            printf("\n");
        }
    printf("\n");
    fflush(stdout);
}
