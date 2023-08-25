## 3-D interconnect capacitance extraction Dataset

### Raw data

Extract `cap3d.tar.gz`, you will get raw data in `cap3d` format, 
including the whole layout (named sram_novia_noTO.cap3d) and the layouts cut by windows.

### Numpy Array Data Representation

Each `.npz` file contains 2 arrays in the same size. 
One is the desity map of the whole layer, and the other is the id map of the conductors.
The raw data are converted into arrays with the method described in section 3.2 of the paper.

### Label files

There are several lines in a label file. Each line corresponds to a test case.

For total capacitance, there are 6 numbers in one line. 
The first 4 numbers are parameters for the cutting window.
The tensors for CNN input are sliced from the full size numpy array according to these 4 parameters.
The 5th number is the id of the master conductor.
The last number is the total capacitance in farad (abbreviated F).

For coupling capacitance, there are 9 numbers in one line. 
The first 4 numbers are parameters for the cutting window.
The 5th number is the id of the master conductor.
The 6th number is the id of the environmental conductor.
The 7th number is the id of the layer of the environmental conductor.
The 8th number is the total capacitance in F.
The last number is the coupling capacitance in F.

For example, coupling capacitance case
```
120 80 22 37 1 25 MET2 4.5800104e-16 -8.282949e-17
```
corresponds to cap3d file `120_80_22_37_1_MET1.cap3d`. The id of the master conductor is 1, 
the id of the environmental conductor is 25, 
the master conductor is on layer `MET1`, and the environmental conductor is on layer `MET2`.
The total capacitance is 4.5800104e-16 F and the coupling capacitance between conductor 1 and conductor 25 is 8.282949e-17 F.
