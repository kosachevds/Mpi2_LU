#include <mpi.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>

using Nanoseconds = std::chrono::nanoseconds;
using Matrix = std::vector<double>;
using MatrixRef = Matrix&;
using MatrixConstRef = const Matrix&;
using MatrixPreparer = void (*) (MatrixRef matrix, const std::vector<int>& map, int k, int rank);

void doWork(const std::vector<int>& sizes, std::ostream& out, const char* title, MatrixPreparer preparer);
void fillHilbertMatrix(MatrixRef matrix, int size);
void printMatrix(MatrixConstRef matrix, const std::vector<int>& map, int myrank);
double calculate(MatrixRef a, const std::vector<int>& map, MatrixPreparer prepare);

void swapWithMaxColumn(MatrixRef matrix, const std::vector<int>& map, int k, int rank);
void swapWithMaxRow(MatrixRef matrix, const std::vector<int>& map, int k, int rank);
void swapWithMaxInSubmatrix(MatrixRef matrix, const std::vector<int>& map, int k, int rank);

void swapColumns(MatrixRef matrix, const std::vector<int>& map, int rank, int col1, int col2);
void swapRows(MatrixRef matrix, const std::vector<int>& map, int rank, int row1, int row2);

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

    std::vector<int> sizes { 10, 50, 100, 500, 1000 };
    //std::vector<int> sizes { 4 };
    auto& out = std::cout;

    doWork(sizes, std::cout, "Simple", nullptr);
    doWork(sizes, std::cout, "MaxInRow", swapWithMaxColumn);
    doWork(sizes, std::cout, "maxInColumn", swapWithMaxRow);

    out.flush();
    MPI_Finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////

void doWork(const std::vector<int>& sizes, std::ostream& out, const char* title, MatrixPreparer preparer)
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
        double time_sec = calculate(a, map, preparer);

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

double calculate(MatrixRef a, const std::vector<int>& map, MatrixPreparer prepare)
{
    int size = map.size();
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    auto start = std::chrono::high_resolution_clock::now();
    if (prepare != nullptr) {
        for (int k = 0; k < size; ++k) {
            prepare(a, map, k, myrank);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int k = 0; k < size - 1; k++) {
        if (map[k] == myrank) {
            auto divider = getitem(a, size, k, k);
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
    auto duration = std::chrono::duration_cast<Nanoseconds>(end - start);
    return duration.count() / 1e9;
}

void swapWithMaxColumn(MatrixRef matrix, const std::vector<int>& map, int k, int rank)
{
    auto size = static_cast<int>(map.size());
    auto max_value = getitem(matrix, size, k, k);
    int column_with_max = k;
    int owner = map[k];
    if (owner == rank) {
        for (int col = k + 1; col < size; ++col) {
            auto item = getitem(matrix, size, k, col);
            if (item > max_value) {
                max_value = item;
                column_with_max = col;
            }
        }
    }
    MPI_Bcast(&column_with_max, 1, MPI_INT, owner, MPI_COMM_WORLD);
    if (column_with_max == k) {
        return;
    }
    swapColumns(matrix, map, rank, k, column_with_max);
}

void swapWithMaxRow(MatrixRef matrix, const std::vector<int>& map, int k, int rank)
{
    const int TAG1 = 0;
    auto size = static_cast<int>(map.size());
    auto max_value = getitem(matrix, size, k, k);
    auto row_with_max = k;
    auto k_owner = map[k];
    for (int row = k + 1; row < size; ++row) {
        auto row_owner = map[row];
        double row_value = 0;
        if (rank == row_owner) {
            row_value = getitem(matrix, size, row, k);
            if (rank != k_owner) {
                MPI_Send(&row_value, 1, MPI_DOUBLE, k_owner, TAG1, MPI_COMM_WORLD);
            }
        }
        if (rank != k_owner) {
            continue;
        }
        if (rank != row_owner) {
            MPI_Status status;
            MPI_Recv(&row_value, 1, MPI_DOUBLE, map[row], TAG1, MPI_COMM_WORLD, &status);
        }
        if (row_value > max_value) {
            row_with_max = row;
            max_value = row_value;
        }
    }
    MPI_Bcast(&row_with_max, 1, MPI_INT, k_owner, MPI_COMM_WORLD);
    if (row_with_max == k || (rank != map[row_with_max] && rank != k_owner)) {
        return;
    }
    swapRows(matrix, map, rank, row_with_max, k);
}

void swapWithMaxInSubmatrix(MatrixRef matrix, const std::vector<int>& map, int k, int rank)
{
    const int TAG0 = 0;
    const int TAG1 = 0;
    auto size = static_cast<int>(map.size());
    auto k_owner = map[k];
    auto max_value = getitem(matrix, size, k, k);
    auto row_with_max = k, col_with_max = k;
    for (int row = k; row < size; ++row) {
        double max_in_row = getitem(matrix, size, row, k);
        int col_with_max_in_row = k;
        auto row_owner = map[row];
        if (rank == row_owner) {
            for (int col = k; col < size; ++col) {
                auto item = getitem(matrix, size, row, col);
                if (item > max_in_row) {
                    max_in_row = item;
                    col_with_max_in_row = col;
                }
            }
            if (rank != k_owner) {
                MPI_Send(&max_in_row, 1, MPI_DOUBLE, k_owner, TAG0, MPI_COMM_WORLD);
                MPI_Send(&col_with_max_in_row, 1, MPI_INT, k_owner, TAG1, MPI_COMM_WORLD);
            }
        }
        if (rank != k_owner) {
            continue;
        }
        if (rank != row_owner) {
            MPI_Status status;
            MPI_Recv(&max_in_row, 1, MPI_DOUBLE, row_owner, TAG0, MPI_COMM_WORLD, &status);
            MPI_Recv(&col_with_max_in_row, 1, MPI_INT, row_owner, TAG1, MPI_COMM_WORLD, &status);
        }
        if (max_in_row > max_value) {
            max_value = max_in_row;
            row_with_max = row;
            col_with_max = col_with_max_in_row;
        }
    }
    if (col_with_max != k) {
        swapColumns(matrix, map, rank, k, col_with_max);
    }
    if (row_with_max != k && (rank == k_owner || rank == map[row_with_max])) {
        swapRows(matrix, map, rank, k, row_with_max);
    }
}

void swapColumns(MatrixRef matrix, const std::vector<int>& map, int rank, int col1, int col2)
{
    auto size = static_cast<int>(map.size());
    for (int row = 0; row < size; ++row) {
        if (map[row] != rank) {
            continue;
        }
        auto value_col1 = getitem(matrix, size, row, col1);
        auto value_col2 = getitem(matrix, size, row, col2);
        setitem(matrix, size, row, col1, value_col2);
        setitem(matrix, size, row, col2, value_col1);
    }
}

void swapRows(MatrixRef matrix, const std::vector<int>& map, int rank, int row1, int row2)
{
    const int TAG1 = 1;
    const int TAG2 = 2;
    auto size = static_cast<int>(map.size());
    std::vector<double> row_buffer(size);
    MPI_Status status;
    int destination_row;
    auto row1_owner = map[row1];
    auto row2_owner = map[row2];
    // Fix: what if row1_owner == row2_owner
    if (rank == row1_owner) {
        destination_row = row1;
        auto row = matrix.data() + size * destination_row;
        MPI_Send(row, size, MPI_DOUBLE, row2_owner, TAG1, MPI_COMM_WORLD);
        MPI_Recv(row_buffer.data(), size, MPI_DOUBLE, row2_owner, TAG2, MPI_COMM_WORLD, &status);
    } else if (rank == row2_owner) {
        destination_row = row2;
        MPI_Recv(row_buffer.data(), size, MPI_DOUBLE, row1_owner, TAG1, MPI_COMM_WORLD, &status);
        auto row = matrix.data() + size * destination_row;
        MPI_Send(row, size, MPI_DOUBLE, row1_owner, TAG2, MPI_COMM_WORLD);
    } else {
        return;
    }
    for (int i = 0; i < size; ++i) {
        setitem(matrix, size, destination_row, i, row_buffer[i]);
    }
}
