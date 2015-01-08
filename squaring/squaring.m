% File for comparing and running a simple vector squaring example comparing cpu and gpu on Matlab. For the GPU, the file will
% call a kernel written in CUDA and residing on a PTX file.

clearvars
close all

% create data
out = zeros(1000000,1);
out = single(out);
in = rand(1000000,1);
in = in * 100;
in = single(in);
num = 1000000;

% cpu code
tic
for k = 1:20
for i = 1:1000000
    out(i) = in(i) * in(i);
end
end
toc

% cpu code with vectorization
tic 
for k = 1:20
    out(i) = in(i).^2;
end
toc 

% gpu code
k = parallel.gpu.CUDAKernel('gpusquare.ptx','gpusquare.cu','gpusquare');

k.GridSize = [1000 1 1];
k.ThreadBlockSize = [1024 1 1];

tic
for p = 1:20
[gin, gout] = feval(k, in, out, num);
gpu_out = gather(gout);
end
toc

sum(gpu_out - out)




