## 3-D interconnect capacitance extraction Dataset

### Raw data

Raw data in `cap3d` format will be released soon.

### Numpy Array Data Representation

Each `.npz` file contains 2 arrays in the same size. 
One is the desity map of the whole layer, and the other is the id map of the conductors.
The raw data are converted into arrays with the method described in section 3.2 of the paper.

### Label files

There are several lines in a label file. Each line corresponds to a test case.

For total capacitance, there are 6 numbers in one line. 
The first 4 numbers are parameters for the cutting window.
The 5th number is the id of the master conductor.
The last number is the total capacitance in farad (abbreviated F).

For coupling capacitance, there are 9 numbers in one line. 
The first 4 numbers are parameters for the cutting window.
The 5th number is the id of the master conductor.
The 6th number is the id of the environmental conductor.
The 7th number is the id of the layer of the environmental conductor.
The 8th number is the total capacitance in F.
The last number is the coupling capacitance in F.
