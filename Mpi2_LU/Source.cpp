#include <mpi.h>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>

using Matrix = std::vector<double>;
using MatrixRef = Matrix&;
using MatrixConstRef = const Matrix&;
using DividerSelector = double (*)(MatrixRef, const std::vector<int>&, int, int);

void doWork(const std::vector<int>& sizes, DividerSelector getDivider, std::ostream& out, const char* title);
void fillHilbertMatrix(MatrixRef matrix, int size);
void printMatrix(MatrixConstRef matrix, const std::vector<int>& map, int myrank);
double calculate(int myrank, MatrixRef a, const std::vector<int>& map, DividerSelector getDivider);
double getKKItem(MatrixRef matrix, const std::vector<int>& map, int k_index, int rank);
double getMaxInRow(MatrixRef matrix, const std::vector<int>& map, int k, int rank);
void swapValuesInColumns(MatrixRef matrix, int size, int row, int col1, int col2);

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
    MPI_Init(&argc, &argv);

    //std::vector<int> sizes { 10, 50, 100, 500, 1000 };
    std::vector<int> sizes { 5 };
    auto& out = std::cout;

    //doWork(sizes, getKKItem, std::cout, "Simple");
    doWork(sizes, getMaxInRow, std::cout, "MaxInRow");

    out.flush();
    MPI_Finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////

void doWork(const std::vector<int>& sizes, DividerSelector getDivider, std::ostream& out, const char* title)
{
    const int WIDTH_1 = 6;
    const int WIDTH_2 = 13;
    //const int WIDTH_3 = 15;

    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (myrank == 0) {
        out << title << std::endl;
        out << ";" << std::setw(WIDTH_1 - 1) << "N";
        out << std::setw(WIDTH_2) << "Time, sec";
        out << std::endl;
    }

    //auto max_size = *std::max_element(sizes.begin(), sizes.end());
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
        auto time_sec = calculate(myrank, a, map, getDivider);

        if (myrank == 0) {
            out << std::setw(WIDTH_1) << size;
            out << std::setw(WIDTH_2) << time_sec;
            out << std::endl;
        }
        //printMatrix(a, map, myrank);
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
        auto divider = getDivider(a, map, k, myrank);
        if (map[k] == myrank) {
            for (int i = k + 1; i < size; i++) {
                auto old_value = getitem(a, size, k, i);
                auto new_value = old_value / divider;
                setitem(a, size, k, i, new_value);
            }
        }
        auto chunk = a.data() + k * size + k + 1;
        MPI_Bcast(chunk, size - k - 1, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
        //std::cout << myrank << ": " << size - k - 1 << std::endl;
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
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    using Nanoseconds = std::chrono::nanoseconds;
    auto duration = std::chrono::duration_cast<Nanoseconds>(end - start);
    return duration.count() / 1e9;
}

double getKKItem(MatrixRef matrix, const std::vector<int>&, int k_index, int)
{
    auto size = static_cast<int>(sqrt(matrix.size()));
    return getitem(matrix, size, k_index, k_index);
}

double getMaxInRow(MatrixRef matrix, const std::vector<int>& map, int k, int rank)
{
    auto size = static_cast<int>(map.size());
    auto max_value = getitem(matrix, size, k, k);
    int column_with_max = k;
    if (map[k] == rank) {
        for (int col = k + 1; col < size; ++col) {
            auto item = getitem(matrix, size, k, col);
            if (item > max_value) {
                max_value = item;
                column_with_max = col;
            }
        }
    }
    MPI_Bcast(&column_with_max, 1, MPI_INT, rank, MPI_COMM_WORLD);
    if (column_with_max == k) {
        return max_value;
    }
    for (int row = 0; row < size; ++row) {
        if (map[row] != rank) {
            continue;
        }
        swapValuesInColumns(matrix, size, row, k, column_with_max);
    }
    return max_value;
}

void swapValuesInColumns(MatrixRef matrix, int size, int row, int col1, int col2)
{
    auto value_col1 = getitem(matrix, size, row, col1);
    auto value_col2 = getitem(matrix, size, row, col2);
    setitem(matrix, size, row, col1, value_col2);
    setitem(matrix, size, row, col2, value_col1);
}
