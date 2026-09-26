#pragma once

#include <cstddef>
#include <vector>

//Grid Class for Storing the 2D Heat-Diffusion Values
class Grid {
private:
  //Store the Logical Number of Rows and Columns in the Grid
  std::size_t number_of_rows_;
  std::size_t number_of_columns_;

  //Store the Entire 2D Grid in One Continuous Vector
  //This Avoids Separate Row Allocations and Keeps Nearby Values Together in Memory
  std::vector<double> grid_values_;

public:
  //Create the Grid and Initialize Every Value to 0.0 as Required
  Grid(std::size_t rows, std::size_t cols)
    : number_of_rows_(rows),
      number_of_columns_(cols),
      grid_values_(rows * cols, 0.0) {
  }

  //Return a Grid Value that can be Read or Changed
  double& operator()(std::size_t row, std::size_t column) {

    //Convert a 2D Row and Column into its Position in the 1D Vector
    //Each Complete Row Contains number_of_columns_ Values
    std::size_t index =
      (row * number_of_columns_) + column;

    return grid_values_[index];
  }

  //Return a Grid Value from a Read-Only Grid
  //This Overload is Used when the Grid is const, such as old_grid
  double operator()(std::size_t row, std::size_t column) const {

    //Use the Same 2D to 1D Position Calculation
    std::size_t index =
      (row * number_of_columns_) + column;

    return grid_values_[index];
  }

  //Return Direct Access to the Continuous Grid Storage
  //This lets the Stencil avoid Calling operator() for Every Individual Value
  double* data() {
    return grid_values_.data();
  }

  //Return Direct Read-Only Access to the Grid Storage
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

  //Get the Grid Dimensions Once so they can be Reused Throughout the Function
  std::size_t number_of_rows = old_grid.rows();
  std::size_t number_of_columns = old_grid.cols();

  //A Grid with no Rows or Columns has no Values to Update
  if (number_of_rows == 0 || number_of_columns == 0) {
    return;
  }

  //The Stencil must Read all Values from the Original Grid
  //If both Parameters Refer to the Same Grid, Copy it First to Keep those Values Safe
  if (&old_grid == &new_grid) {
    Grid old_grid_copy = old_grid;
    apply_stencil(old_grid_copy, new_grid);
    return;
  }

  //Get Direct Access to the Continuous Old and New Grid Storage
  //old_values is Read-Only while new_values Stores the Results
  const double* old_values = old_grid.data();
  double* new_values = new_grid.data();

  //Copy the Top and Bottom Boundary Rows Unchanged
  //The Problem only Applies the Stencil to Interior Grid Points
  for (std::size_t column = 0;
       column < number_of_columns;
       column++) {

    //The Top Row Starts at Index 0
    new_values[column] = old_values[column];

    //Convert the Bottom Row and Current Column into a 1D Index
    std::size_t bottom_index =
      ((number_of_rows - 1) * number_of_columns) + column;

    new_values[bottom_index] =
      old_values[bottom_index];
  }

  //Each Interior Row can be Calculated Independently
  //Static Scheduling divides the Similar-Sized Rows between Threads
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t row = 1;
       row < number_of_rows - 1;
       row++) {

    //Find the First Vector Position Belonging to this Row
    std::size_t row_start =
      row * number_of_columns;

    //Copy the Left Boundary Value Unchanged
    new_values[row_start] =
      old_values[row_start];

    //Find and Copy the Right Boundary Value Unchanged
    std::size_t right_boundary =
      row_start + number_of_columns - 1;

    new_values[right_boundary] =
      old_values[right_boundary];

    //Each Interior Column is Independent because Every Calculation
    //Reads only from old_values and Writes to its own Position in new_values
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t column = 1;
         column < number_of_columns - 1;
         column++) {

      //Find the Current Cell's Position in the 1D Vector
      std::size_t index =
        row_start + column;

      //Apply the Five-Point Stencil:
      //50% of the Center Value and 12.5% from Each of its Four Neighbors
      new_values[index] =
        (0.5 * old_values[index]) +
        (0.125 * (
          old_values[index - number_of_columns] + //Top
          old_values[index + number_of_columns] + //Bottom
          old_values[index - 1] +                 //Left
          old_values[index + 1]                   //Right
        ));
    }
  }
}