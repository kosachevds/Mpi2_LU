#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <vector>

const int N = 4;

void doWork(int myrank, int nprocs, int size);
inline double getitem(const std::vector<double>& matrix, int i, int j, int n)
{
    return matrix[i * n + j];
}
inline void setitem(std::vector<double>& matrix, int i, int j, int n, double value)
{
    matrix[i * n + j] = value;
}

int main(int argc, char* argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    doWork(myrank, nprocs, N);

    if (myrank == 0) {
        system("pause");
    }
    MPI_Finalize();
    return 0;
}

void doWork(int myrank, int nprocs, int size)
{
    int i, j, k;
    std::vector<double> a(size * size);
    if (myrank == 0) {
        for (i = 0; i < size; i++) {
            for (j = 0; j < size; j++) {
                auto value = 1.0 / (i + j + 2 - 1);
                setitem(a, i, j, size, value);
            }
        }
    }
    MPI_Bcast(static_cast<void*>(a.data()), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<int> map(size);
    for (i = 0; i < size; i++)
        map[i] = i % nprocs;
    for (k = 0; k < size - 1; k++) {
        if (map[k] == myrank) {
            auto divider = getitem(a, k, k, size);
            for (i = k + 1; i < size; i++) {
                auto old_value = getitem(a, k, i, size);
                auto new_value = old_value / divider;
                setitem(a, k, i, size, new_value);
            }
        }
        auto chunk = a.data() + k * size + k + 1;
        MPI_Bcast(chunk, size - k - 1, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
        for (i = k + 1; i < size; i++) {
            if (map[i] == myrank) {
                for (j = k + 1; j < size; j++) {
                    auto multiplier = getitem(a, i, k, size);
                    auto k_row_item = getitem(a, k, j, size);
                    auto old_value = getitem(a, i, j, size);
                    auto new_value = old_value - k_row_item * multiplier;
                    setitem(a, i, j, size, new_value);
                }
            }
        }
    }
    // Printing the entries of the matrix
    for (i = 0; i < size; i++)
        if (map[i] == myrank) {
            printf("%d:\t", i + 1);
            for (j = 0; j < size; j++)
                printf("%lg, ", getitem(a, i, j, size));
            printf("\n");
        }
    printf("\n");
    fflush(stdout);
}
