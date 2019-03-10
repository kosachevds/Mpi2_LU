#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <vector>

const int N = 4;

void doWork(int myrank, int nprocs, int size);
void fillHilbertMatrix(std::vector<double>& matrix, int size);

inline double getitem(const std::vector<double>& matrix, int rows_count, int i, int j)
{
    return matrix[i * rows_count + j];
}
inline void setitem(std::vector<double>& matrix, int rows_count, int i, int j, double value)
{
    matrix[i * rows_count + j] = value;
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
    int item_count = size * size;
    std::vector<double> a(item_count);
    if (myrank == 0) {
        fillHilbertMatrix(a, size);
    }
    auto raw_void_ptr = static_cast<void*>(a.data());
    MPI_Bcast(raw_void_ptr, item_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<int> map(size);
    for (i = 0; i < size; i++) {
        map[i] = i % nprocs;
    }
    for (k = 0; k < size - 1; k++) {
        if (map[k] == myrank) {
            auto divider = getitem(a, size, k, k);
            for (i = k + 1; i < size; i++) {
                auto old_value = getitem(a, size, k, i);
                auto new_value = old_value / divider;
                setitem(a, size, k, i, new_value);
            }
        }
        auto chunk = a.data() + k * size + k + 1;
        MPI_Bcast(chunk, size - k - 1, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
        for (i = k + 1; i < size; i++) {
            if (map[i] != myrank) {
                continue;
            }
            for (j = k + 1; j < size; j++) {
                auto multiplier = getitem(a, size, i, k);
                auto k_row_item = getitem(a, size, k, j);
                auto old_value = getitem(a, size, i, j);
                auto new_value = old_value - k_row_item * multiplier;
                setitem(a, size, i, j, new_value);
            }
        }
    }
    // Printing the entries of the matrix
    for (i = 0; i < size; i++) {
        if (map[i] != myrank) {
            continue;
        }
        printf("%d:\t", i + 1);
        for (j = 0; j < size; j++) {
            printf("%lg, ", getitem(a, size, i, j));
        }
        printf("\n");
    }
    printf("\n");
    fflush(stdout);
}

void fillHilbertMatrix(std::vector<double>& matrix, int size)
{
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            auto value = 1.0 / (i + j + 2 - 1);
            setitem(matrix, size, i, j, value);
        }
    }
}
