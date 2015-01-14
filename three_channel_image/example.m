% File for running an example where a three channel image is passed to a kernel and we want to check in what format is the
% data passed to the kernel.

clearvars
close all

device = gpuDevice;
reset(device)

% create data
data = zeros(2,2,3);

data(1,1,1) = 1; % first top left
data(1,2,1) = 19; % first top right
data(1,1,2) = 2; % second top left
data(1,1,3) = 3; % third top left

% a kernel being started
k = parallel.gpu.CUDAKernel('threechannel.ptx','threechannel.cu','threechannel');

k.GridSize = [1 1 1];
k.ThreadBlockSize = [12 1 1];

red = 1;
green = 1;
blue = 1;
ret = zeros(12,1);

% check position of the the numbers 1,2,19 and 3 in the return structure retu to understand the positioning being used when
% sending data from matlab to cuda.
[a, r, g, b, retu] = feval(k, data, red, green, blue, ret, 12);

retcpu = gather(retu);

