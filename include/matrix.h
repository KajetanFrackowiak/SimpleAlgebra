#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>

template <typename T>
class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    Matrix(const Matrix& other);
 
    ~Matrix();

    Matrix operator*(const Matrix& other) const;
    Matrix& operator=(const std::vector<std::vector<T>>& values);

    void relu();

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    const std::vector<std::vector<T>>& getData() const { return data; }

private:
    size_t rows;
    size_t cols;
    std::vector<std::vector<T>> data;
};

#endif  // MATRIX_H
