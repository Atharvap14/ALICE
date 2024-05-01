# Exploration of Latent Spaces of 3D Shapes via Isomap and Inverse Mapping

## Run the instruction as: 

```bash
python code.py
```

This code takes the 128D latent vectors of all the 400 chair dataset to reduced them to 2D embedding. Here the file *'chairs_neighbors_7_isomap_reduced_2_test_points_111_360.csv'* is a given sample test data set that is prepared by connecting paths between chair 111 and chiar 360 using 20 equdistantly place points. The *'inverse'* function inversely maps these 20 2D embedding points to 128D and creates *'chairs_neighbors_7_isomap_reduced_2_inversed.csv'* file in the output folder. Here the considered nearest neighbor number for the isomap function is 7. 

###### Alternative run instruction
 
```bash
python code.py arg1 arg2 arg3
```

By uncommenting lines 7,9,11 (and commenting 8,10,12) we can access other datasets rather than chair by changing arg1 (chiar = 1, lamp = 2, table = 3), and use different number of nearest neighbors rather than 7 by changing arg3. arg3 represents the reduced dimension after isomap.  

*chairs_neighbors_7_isomap_reduced_2_test_train.png* is the pictural representation of the test and train data used in this specific implementation.

The shape decoder implementation is in [this](https://github.com/IsaacGuan/implicit-decoder) repository
