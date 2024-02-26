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

    try {
        Matrix<float> result = matrix1 * matrix2;

        ASSERT_EQ(result.getRows(), 2);
        ASSERT_EQ(result.getCols(), 4);

        // Define the expected result matrix manually
        Matrix<float> expectedResult(2, 4);
        expectedResult = {{74.0f, 80.0f, 86.0f, 92.0f},
                           {173.0f, 188.0f, 203.0f, 218.0f}};

        // Compare each element of the result with the expected result
        for (int i = 0; i < result.getRows(); ++i) {
            for (int j = 0; j < result.getCols(); ++j) {
                ASSERT_FLOAT_EQ(result.getData()[i][j], expectedResult.getData()[i][j]);
            }
        }
    } catch (...) {
        // If an exception is caught, fail the test
        FAIL() << "Matrix multiplication operation threw an exception.";
    }
}

// #include <gtest/gtest.h>
// #include "matrix.h"

// // TEST WITH FLOAT MATRIX MULTIPLICATION
// TEST(MatrixTest, Multiplication) {
//     Matrix<float> matrix1(2, 3);
//     Matrix<float> matrix2(3, 4);

//     matrix1 = {{1.0f, 2.0f, 3.0f},
//                {4.0f, 5.0f, 6.0f}};

//     matrix2 = {{7.0f, 8.0f, 9.0f, 10.0f},
//                {11.0f, 12.0f, 13.0f, 14.0f},
//                {15.0f, 16.0f, 17.0f, 18.0f}};

//     Matrix<float> result = matrix1 * matrix2;

//     ASSERT_EQ(result.getRows(), 2);
//     ASSERT_EQ(result.getCols(), 4);

//     ASSERT_FLOAT_EQ(result.getData()[0][0], 74.0f);
//     ASSERT_FLOAT_EQ(result.getData()[0][1], 80.0f);
//     ASSERT_FLOAT_EQ(result.getData()[0][2], 86.0f);
//     ASSERT_FLOAT_EQ(result.getData()[0][3], 92.0f);
//     ASSERT_FLOAT_EQ(result.getData()[1][0], 173.0f);
//     ASSERT_FLOAT_EQ(result.getData()[1][1], 188.0f);
//     ASSERT_FLOAT_EQ(result.getData()[1][2], 203.0f);
//     ASSERT_FLOAT_EQ(result.getData()[1][3], 218.0f);
// }

// // TEST WITH DOUBLE MATRIX MULTIPLICATION
// TEST(MatrixTest, MultiplicationDouble) {
//     Matrix<double> matrix1(2, 3);
//     Matrix<double> matrix2(3, 4);

//     matrix1 = {{1.5, 2.7, 3.1},
//              {4.2, 5.3, 6.4}};

//     matrix2 = {{7.8, 8.9, 9.01, 10.12},
//              {11.23, 12.34, 13.45, 14.56},
//              {15.67, 16.78, 17.89, 18.90}};

//     Matrix<double> result = matrix1 * matrix2;

//     ASSERT_EQ(result.getRows(), 2);
//     ASSERT_EQ(result.getCols(), 4);

//     ASSERT_NEAR(result.getData()[0][0], 90.6, 1e-2);
//     ASSERT_NEAR(result.getData()[0][1], 98.7, 1e-1);
//     ASSERT_NEAR(result.getData()[0][2], 105.3, 1e-1);
//     ASSERT_NEAR(result.getData()[0][3], 113.07, 1e-1);
//     ASSERT_NEAR(result.getData()[1][0], 192.57, 1e-1);
//     ASSERT_NEAR(result.getData()[1][1], 210.17, 1e-1);
//     ASSERT_NEAR(result.getData()[1][2], 223.62, 1e-1);
//     ASSERT_NEAR(result.getData()[1][3], 240.63, 1e-2); 
// }

// // TEST WITH MATRIX MULTIPLICATION, THEN USE RELU
// TEST(MatrixTest, MultiplicationAndReluDouble) {
//     Matrix<double> matrix1(2, 3);
//     Matrix<double> matrix2(3, 4);
 
//     matrix1 = {{1.5, -2.7, 3.1},
//              {4.2, -5.3, 6.4}};

//     matrix2 = {{7.8, 8.9, 9.01, 10.12},
//              {11.23, 12.34, 13.45, 14.56},
//              {15.67, 16.78, 17.89, 18.90}};

//     Matrix<double> result = matrix1 * matrix2;
 
//     result.relu();

//     ASSERT_EQ(result.getRows(), 2);
//     ASSERT_EQ(result.getCols(), 4);

//     ASSERT_NEAR(result.getData()[0][0], 30.0, 1e-1);
//     ASSERT_DOUBLE_EQ(result.getData()[0][1], 32.05);
//     ASSERT_DOUBLE_EQ(result.getData()[0][2], 32.659);
//     ASSERT_DOUBLE_EQ(result.getData()[0][3], 34.458);
//     ASSERT_NEAR(result.getData()[1][0], 73.53, 1e-1);
//     ASSERT_DOUBLE_EQ(result.getData()[1][1], 79.37);
//     ASSERT_DOUBLE_EQ(result.getData()[1][2], 81.053);
//     ASSERT_DOUBLE_EQ(result.getData()[1][3], 86.296);
// }


// // TEST WITH USE RELU
// TEST(MatrixTest, ReluWithVariousValues) {
//     Matrix<double> matrix(2, 3);

//     matrix = {{0.0, -1.0, 3.1},
//             {-.000001, 2.5, 4.2}};

//     matrix.relu();

//     ASSERT_DOUBLE_EQ(matrix.getData()[0][0], 0.0);
//     ASSERT_DOUBLE_EQ(matrix.getData()[0][1], 0.0);
//     ASSERT_DOUBLE_EQ(matrix.getData()[0][2], 3.1);
//     ASSERT_DOUBLE_EQ(matrix.getData()[1][0], 0.0);
//     ASSERT_DOUBLE_EQ(matrix.getData()[1][1], 2.5);
//     ASSERT_DOUBLE_EQ(matrix.getData()[1][2], 4.2); 

// }

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}