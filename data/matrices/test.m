load .\A16x16.txt
load .\b16x1.txt
 
A = reshape(A16x16(2:end,3),[16,16]);
b = b16x1(2:end,3);
x = A\b;
norm(A*x-b)

load .\A64x64.txt
load .\b64x1.txt

A = reshape(A64x64(2:end,3),[64,64]);
b = b64x1(2:end,3);
x = A\b;
norm(A*x-b)
 
load .\A200x200.txt
load .\b200x1.txt
 
A = reshape(A200x200(2:end,3),[200,200]);
b = b200x1(2:end,3);
x = A\b;
norm(A*x-b)

tic,
xcg=conjgrad(A,b, 1e-60);
toc
norm(A*xcg-b)

