#include <gtest/gtest.h>
#include "matrix.h"

TEST(MatrixTest, Multiplication) {
    Matrix<float> matrix1(2, 3);
    Matrix<float> matrix2(3, 4);

    matrix1 = {{1.0f, 2.0f, 3.0f},
               {4.0f, 5.0f, 6.0f}};

    matrix2 = {{7.0f, 8.0f, 9.0f, 10.0f},
               {11.0f, 12.0f, 13.0f, 14.0f},
               {15.0f, 16.0f, 17.0f, 18.0f}};

    Matrix<float> result = matrix1 * matrix2;

    ASSERT_EQ(result.getRows(), 2);
    ASSERT_EQ(result.getCols(), 4);

    ASSERT_FLOAT_EQ(result.getData()[0][0], 74.0f);
    ASSERT_FLOAT_EQ(result.getData()[0][1], 80.0f);
    ASSERT_FLOAT_EQ(result.getData()[0][2], 86.0f);
    ASSERT_FLOAT_EQ(result.getData()[0][3], 92.0f);
    ASSERT_FLOAT_EQ(result.getData()[1][0], 173.0f);
    ASSERT_FLOAT_EQ(result.getData()[1][1], 188.0f);
    ASSERT_FLOAT_EQ(result.getData()[1][2], 203.0f);
    ASSERT_FLOAT_EQ(result.getData()[1][3], 218.0f);
}

// TEST(MatrixTest, Relu) {
//     Matrix<float> matrix(2, 2);

//     matrix = {{-1.0f, 2.0f},
//               {0.5f, -3.0f}};

//     // Assuming you have a 'relu()' function in your Matrix class
//     matrix.relu();

//     // Add assertions to test the result after applying Relu
// }

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
