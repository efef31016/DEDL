function dydx = bvpfcn(x,y)
dydx = [y(2)
       -1/0.05*y(2) + 1/0.05^2*y(1)];
end