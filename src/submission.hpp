#pragma once

#include <cstddef>
#include <vector>

//Grid Class for Storing the 2D Heat-Diffusion Values
class Grid {
private:
  //Store the Number of Rows and Columns in the Grid
  std::size_t number_of_rows_;
  std::size_t number_of_columns_;

  //Store all Grid Values in One Continuous Vector
  //This keeps nearby Grid Values close together in memory
  std::vector<double> grid_values_;

public:
  //Create the Grid and Start Every Value at 0.0 as Required
  //The Vector needs rows * cols Values to Store the Entire Grid
  Grid(std::size_t rows, std::size_t cols)
    : number_of_rows_(rows),
      number_of_columns_(cols),
      grid_values_(rows * cols, 0.0) {
  }

  //Return a Grid Value that can be Read or Changed
  double& operator()(std::size_t row, std::size_t column) {

    //Convert the 2D Row and Column into its Position in the 1D Vector
    //Each Full Row contains number_of_columns_ Values
    std::size_t index = (row * number_of_columns_) + column;

    //Return the Actual Stored Value so it can be Modified
    return grid_values_[index];
  }

  //Return a Grid Value from a Read-Only Grid
  //This Version is Used when the Grid is const and cannot be Changed
  double operator()(std::size_t row, std::size_t column) const {

    //Use the Same Row and Column Calculation to Find the Value
    std::size_t index = (row * number_of_columns_) + column;

    //Return the Stored Value without Allowing it to be Changed
    return grid_values_[index];
  }

  //Return Direct Access to the Start of all Grid Data
  //This lets the Stencil Access the Continuous Vector Storage Directly
  double* data() {
    return grid_values_.data();
  }

  //Return Direct Read-Only Access to the Start of all Grid Data
  //This is Used for old_grid because its Values should not be Changed
  const double* data() const {
    return grid_values_.data();
  }

  //Return the Number of Rows in the Grid
  std::size_t rows() const {
    return number_of_rows_;
  }

  //Return the Number of Columns in the Grid
  std::size_t cols() const {
    return number_of_columns_;
  }
};

//Apply One Step of the Five-Point Stencil
//Read all Original Values from old_grid and Store Results in new_grid
inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {

  //Get the Grid Dimensions Once so they can be Reused
  std::size_t number_of_rows = old_grid.rows();
  std::size_t number_of_columns = old_grid.cols();

  //Stop if the Grid has no Rows or Columns because there is Nothing to Process
  if (number_of_rows == 0 || number_of_columns == 0) {
    return;
  }

  //If Both Parameters Refer to the Same Grid, Make a Copy First
  //This Prevents Updated Values from Affecting Later Stencil Calculations
  if (&old_grid == &new_grid) {
    Grid old_grid_copy = old_grid;
    apply_stencil(old_grid_copy, new_grid);
    return;
  }

  //Get Direct Access to the Old and New Grid Values
  //old_values is Read-Only while new_values Stores the New Results
  //Restrict Tells the Compiler these Two Areas of Memory do not Overlap
  const double* __restrict__ old_values = old_grid.data();
  double* __restrict__ new_values = new_grid.data();

  //Copy the Top and Bottom Boundary Rows
  //Boundary Values must Stay the Same and are not Recalculated
  for (std::size_t column = 0; column < number_of_columns; column++) {

    //The Top Row Starts at Index 0, so the Column is already the Correct Index
    new_values[column] = old_values[column];

    //Find the Matching Position in the Last Row
    std::size_t bottom_index =
      ((number_of_rows - 1) * number_of_columns) + column;

    //Copy the Bottom Boundary Value without Changing it
    new_values[bottom_index] = old_values[bottom_index];
  }

  //Go Through Every Interior Row
  //Start at Row 1 and Stop Before the Last Row because those are Boundaries
  for (std::size_t row = 1; row < number_of_rows - 1; row++) {

    //Find the First Vector Position Belonging to the Current Row
    std::size_t row_start = row * number_of_columns;

    //Copy the Left Boundary at Column 0
    new_values[row_start] = old_values[row_start];

    //Find the Last Column Position in the Current Row
    std::size_t right_boundary =
      row_start + number_of_columns - 1;

    //Copy the Right Boundary without Changing it
    new_values[right_boundary] =
      old_values[right_boundary];

    //Go Through Every Interior Column in the Current Row
    //Skip Column 0 and the Last Column because they are Boundaries
    for (std::size_t column = 1; column < number_of_columns - 1; column++) {

      //Find the Current Cell's Position in the 1D Vector
      std::size_t index = row_start + column;

      //Apply the Five-Point Stencil
      //50% Comes from the Current Cell
      //12.5% Comes from Each of the Top, Bottom, Left, and Right Neighbors
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