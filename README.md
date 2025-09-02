# Seismic Analyzer

Seismic Analyzer is a Python application designed to process text files containing seismic acceleration data. 

## Usage

1. **Input File Format:**

   - Text file with 6 columns and no headers.  
   - RPC Binary Files (.tim) used with MTS StexPro & 469D.  

   - Each line in the input file represents seismic data with 6 columns, in the following order:
     - X: X-coordinate
     - Y: Y-coordinate
     - Z: Z-coordinate
     - Rx: Degrees of freedom (Rotation-X)
     - Ry: Degrees of freedom (Rotation-Y)
     - Rz: Degrees of freedom (Rotation-Z)

2. **Delimiter:**

   -  The application will automatically detect the delimiter used by the user.
     

3. **Processing Data:**

   - Seismic Analyzer processes the input data to perform various seismic analysis tasks. The basis for all computations is numpy. 


