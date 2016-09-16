mex cov_4uint8_to_float.cpp
mex cov_8uint8_to_2floats.cpp
mex cov_uint8_to_double.cpp
mex cov_uint8_to_PhoStat.cpp
mex -v -largeArrayDims  COMPFLAGS="$COMPFLAGS -fopenmp -std=c++11" -lgomp HelicalToFanFunc_mex.cpp

%------------------------------------------------------------------------- 
% GE Confidential. General Electric Proprietary Data (c) 2015 General Electric Company
% Date: Mar 28, 2016
% Routine: compileCUDA.m
% Author
%	Rui Liu
% Organization: 
%  Wake Forest Health Sciences.
%-------------------------------------------------------------------------
% % 

system( '/usr/local/cuda/bin/nvcc -std=c++11 -Xcompiler -fopenmp -O3 --use_fast_math --compile -o DD2Proj_4gpus.o  --compiler-options -fPIC  -I"/usr/local/MATLAB/R2015b/extern/extern/include " -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc "DD2Proj_4gpus.cu" ' );
system( '/usr/local/cuda/bin/nvcc -std=c++11 -Xcompiler -fopenmp -O3 --use_fast_math --compile -o DD2Back_4gpus.o  --compiler-options -fPIC  -I"/usr/local/MATLAB/R2015b/extern/extern/include " -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc "DD2Back_4gpus.cu" ' );

mex -v -largeArrayDims  COMPFLAGS="$COMPFLAGS -fopenmp -std=c++11" -L/usr/local/cuda/lib64 -lcudart -lgomp DD2Proj.cpp DD2Proj_4gpus.o 
mex -v -largeArrayDims  COMPFLAGS="$COMPFLAGS -fopenmp -std=c++11" -L/usr/local/cuda/lib64 -lcudart -lgomp DD2Back.cpp DD2Back_4gpus.o 

pcode CollectImageCfg.m
pcode CollectReconCfg.m
pcode ConvertToReconConf.m
pcode DD2MutiSlices.m
pcode HelicalToFan_routine.m
pcode OSSART_AAPM.m
pcode readProj.m