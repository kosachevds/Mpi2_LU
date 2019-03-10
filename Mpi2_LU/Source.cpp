#include <mpi.h>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <chrono>

using Matrix = std::vector<double>;
using MatrixRef = Matrix&;
using MatrixConstRef = const Matrix&;
using DividerSelector = double (*)(MatrixRef, int);

const int N = 4;

void doWork(int myrank, const std::vector<int>& sizes, DividerSelector getDivider);
void fillHilbertMatrix(MatrixRef matrix, int size);
void printMatrix(MatrixConstRef matrix, const std::vector<int>& map, int myrank);
double calculate(int myrank, MatrixRef a, const std::vector<int>& map, DividerSelector getDivider);
double getKKItem(MatrixRef matrix, int k_index);

inline double getitem(MatrixConstRef matrix, int rows_count, int i, int j)
{
    return matrix[i * rows_count + j];
}
inline void setitem(MatrixRef matrix, int rows_count, int i, int j, double value)
{
    matrix[i * rows_count + j] = value;
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    int myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    doWork(myrank, { N }, getKKItem);

    std::cout.flush();

    if (myrank == 0) {
        system("pause");
    }
    MPI_Finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////

void doWork(int myrank, const std::vector<int>& sizes, DividerSelector getDivider)
{
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    Matrix a;
    std::vector<int> map;
    for (int size: sizes) {
        int item_count = size * size;
        a.resize(item_count);
        if (myrank == 0) {
            fillHilbertMatrix(a, size);
        }
        auto raw_void_ptr = static_cast<void*>(a.data());
        MPI_Bcast(raw_void_ptr, item_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        map.resize(size);
        for (int i = 0; i < size; ++i) {
            map[i] = i % nprocs;
        }

        calculate(myrank, a, map, getDivider);

        printMatrix(a, map, myrank);
    }
}

void fillHilbertMatrix(MatrixRef matrix, int size)
{
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            auto value = 1.0 / (i + j + 2 - 1);
            setitem(matrix, size, i, j, value);
        }
    }
}

void printMatrix(MatrixConstRef matrix, const std::vector<int>& map, int myrank)
{
    int size = map.size();
    for (int i = 0; i < size; i++) {
        if (map[i] != myrank) {
            continue;
        }
        printf("%d:\t", i + 1);
        for (int j = 0; j < size; j++) {
            std::cout << getitem(matrix, size, i, j) << ", ";
        }
        std::cout << '\n';
    }
}

double calculate(int myrank, MatrixRef a, const std::vector<int>& map, DividerSelector getDivider)
{
    int size = map.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < size - 1; k++) {
        if (map[k] == myrank) {
            auto divider = getDivider(a, k);
            for (int i = k + 1; i < size; i++) {
                auto old_value = getitem(a, size, k, i);
                auto new_value = old_value / divider;
                setitem(a, size, k, i, new_value);
            }
        }
        auto chunk = a.data() + k * size + k + 1;
        MPI_Bcast(chunk, size - k - 1, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
        for (int i = k + 1; i < size; i++) {
            if (map[i] != myrank) {
                continue;
            }
            for (int j = k + 1; j < size; j++) {
                auto multiplier = getitem(a, size, i, k);
                auto k_row_item = getitem(a, size, k, j);
                auto old_value = getitem(a, size, i, j);
                auto new_value = old_value - k_row_item * multiplier;
                setitem(a, size, i, j, new_value);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    using Nanoseconds = std::chrono::nanoseconds;
    auto duration = std::chrono::duration_cast<Nanoseconds>(end - start);
    return duration.count() / 1e9;
}

double getKKItem(MatrixRef matrix, int k_index)
{
    auto size = static_cast<int>(sqrt(matrix.size()));
    return getitem(matrix, size, k_index, k_index);
}
