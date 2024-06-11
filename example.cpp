#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <armadillo>
#include <iostream>

using namespace mlpack;
//using namespace mlpack::regression;
using namespace arma;

int main()
{
    // Create a simple dataset.
    mat X = { {1, 2, 3, 4}, {4, 5, 6, 7} };
    Row<size_t> y = {0, 1, 0, 1};

    // Create and train the logistic regression model.
    LogisticRegression<mat> lr(X, y, 0.5);

    // Predict using the trained model.
    Row<size_t> predictions;
    lr.Classify(X, predictions);

    // Output predictions.
    predictions.print("Predictions:");

    return 0;
}
