#pragma once

#include <cstddef>
#include <vector>

//Grid Class for Storing the 2D Heat-Diffusion Values
class Grid {
private:
  //Store the Number of Rows and Columns
  std::size_t number_of_rows_;
  std::size_t number_of_columns_;

  //Store all Grid Values in One Vector
  std::vector<double> grid_values_;

public:
  //Create the Grid and Start Every Value at 0.0 (Required)
  Grid(std::size_t rows, std::size_t cols)
    : number_of_rows_(rows),
      number_of_columns_(cols),
      grid_values_(rows * cols, 0.0) {
  }

  //Return a Grid Value that can be Read or Changed
  double& operator()(std::size_t row, std::size_t column) {

    //Find Where this Row and Column is Stored in the One-Dimensional Vector
    std::size_t index = (row * number_of_columns_) + column;

    return grid_values_[index];
  }

  //Return a Grid Value from a Read-Only Grid
  double operator()(std::size_t row, std::size_t column) const {

    //Find the Same Row and Column Position in the One-Dimensional Vector
    std::size_t index = (row * number_of_columns_) + column;

    return grid_values_[index];
  }

  //Return Direct Access to all Grid Values
  double* data() {
    return grid_values_.data();
  }

  //Return Direct Read-Only Access to all Grid Values
  const double* data() const {
    return grid_values_.data();
  }

  //Return the Number of Rows
  std::size_t rows() const {
    return number_of_rows_;
  }

  //Return the Number of Columns
  std::size_t cols() const {
    return number_of_columns_;
  }
};

//Apply One Step of the Five-Point Stencil
inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {

  //Get the Grid Dimensions
  std::size_t number_of_rows = old_grid.rows();
  std::size_t number_of_columns = old_grid.cols();

  //Stop if the Grid has no Rows or Columns
  if (number_of_rows == 0 || number_of_columns == 0) {
    return;
  }

  //If Both Grids are the Same Object, Make a Copy First
  if (&old_grid == &new_grid) {
    Grid old_grid_copy = old_grid;
    apply_stencil(old_grid_copy, new_grid);
    return;
  }

  //Get Direct Access to the Old and New Grid Values
  const double* old_values = old_grid.data();
  double* new_values = new_grid.data();

  //Copy the Top and Bottom Boundary Rows
  for (std::size_t column = 0; column < number_of_columns; column++) {

    //Copy the Top Boundary
    new_values[column] = old_values[column];

    //Find and Copy the Bottom Boundary
    std::size_t bottom_index =
      ((number_of_rows - 1) * number_of_columns) + column;

    new_values[bottom_index] = old_values[bottom_index];
  }

  //Split the Interior Rows Between CPU Threads
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t row = 1; row < number_of_rows - 1; row++) {

    //Find Where this Row Starts in the One-Dimensional Vector
    std::size_t row_start = row * number_of_columns;

    //Copy the Left Boundary
    new_values[row_start] = old_values[row_start];

    //Copy the Right Boundary
    std::size_t right_boundary =
      row_start + number_of_columns - 1;

    new_values[right_boundary] =
      old_values[right_boundary];

    //Calculate Multiple Columns Efficiently when OpenMP is Available
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t column = 1;
         column < number_of_columns - 1;
         column++) {

      //Find the Current Cell's Position in the Vector
      std::size_t index = row_start + column;

      //Apply the Five-Point Stencil
      new_values[index] =
        (0.5 * old_values[index]) +
        (0.125 * (
          old_values[index - number_of_columns] +
          old_values[index + number_of_columns] +
          old_values[index - 1] +
          old_values[index + 1]
        ));
    }
  }
}