export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=4

# Single
./build/bin/hpgmg-fv 7 8

# MPI
#mpirun -np 2 ./build/bin/hpgmg-fv 7 8

# CUDA-aware MPI
#mpirun -np 2 -env MV2_USE_CUDA 1 -env MV2_CPU_MAPPING 1:2 ./build/bin/hpgmg-fv 7 8
