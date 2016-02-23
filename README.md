# StreamBasedAL
This folder contains the C++ code of a Stream-Based Active Learning (StreamBasedAL) environment with Mondrian forests.

Mondrian Forests are introduced in:

B. Lakshminarayanan, D. M. Roy, and Y. W. Teh, “Mondrian Forests:
Efficient Online Random Forests,” in Advances in Neural Information
Processing Systems (NIPS), 2014.

The training and test data set is from the KITTI benchmark (http://www.cvlibs.net/datasets/kitti/) and consists of 18 streams of 
segmented 3D point clouds from urban traffic environments, which are 
concatenate to one long stream. The computed 60-dimensional feature vector are 
proposed in:

H. Himmelsbach, M. and Luettel, T. and Wuensche, “Real-time Object Classification in 3D Point Clouds Using Point Feature Histograms,” in Intern. Conf. on Intell. Robots and Systems (IROS), 2009.

To run this code you will need the following main packages:
- Armadillo: libarmadillo (c++ linear algebra library)
  (it is recommended to install LAPACK, BLAS and ATLAS), along with the 
  corresponding development/header files)
- boost: libboost 1.54

