#include "matrix.h"

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(rows, std::vector<T>(cols, T())) {}

template <typename T>
Matrix<T>::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}

template <typename T>
Matrix<T>::~Matrix() {}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    Matrix<T> result(rows, other.cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            for (size_t k = 0; k < cols; ++k) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }

    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const std::vector<std::vector<T>>& values) {
    if (values.size() == 0 || values[0].size() == 0) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }

    rows = values.size();
    cols = values[0].size();
    data = values;

    return *this;
}

template <typename T>
void Matrix<T>::relu() {
    static_assert(std::is_arithmetic_v<T>, "Type T must be arithmetic (comparable to 0)");

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = std::max(data[i][j], T(0));
        }
    }
}

template class Matrix<float>;
template class Matrix<double>;

