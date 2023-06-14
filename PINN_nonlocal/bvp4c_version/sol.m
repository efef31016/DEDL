x1 = linspace(0,0.5,10000);
x2 = linspace(0.5,1,10000);
x = union(x1,x2);
solinit = bvpinit(x, @guess);
sol_v = bvp4c(@bvpfcn, @bcfcn_v, solinit);
sol_w = bvp4c(@bvpfcn, @bcfcn_w, solinit);
v = deval(sol_v, x);
w = deval(sol_w, x);
v = v(1,:);
w = w(1,:);

res_11 = 1 - 1/0.05 * trapz(x2, v(length(x1):length(x)));
res_12 = -1/0.05 * trapz(x2, w(length(x1):length(x)));
res_21 = -1/0.05*trapz(x1, v(1:length(x1)));
res_22 = 1-1/0.05*trapz(x1, w(1:length(x1)));

A = [res_11, res_12; res_21, res_22];
b = [1; 1];
u_bdd = linsolve(A, b);
solve = u_bdd(1).*v + u_bdd(2).*w;

plot(x, solve, '-o');
writematrix(solve, "solved_matlab");