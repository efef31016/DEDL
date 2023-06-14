curve(12*(x-1/2)^2,0,1, ylim=c(0,5))
curve(x-x+3,0,1,add=T)

# target pdf : 12(x-1/2)^2 in x\in[0,1] and others zero
M = 1
n = 5000
y = runif(n)
u = runif(n)
x = y[u <= 12*(y-1/2)^2 / (M*1)]
length(x)/n

sam = sample(x,1000)
hist(sam, 30)

file_path = "C:/Users/jiao/Desktop/M_D/¬ã¨s/2022 ¤W/PINN_nonlocal/sample.txt"
write.table(sort(sam), file = file_path, row.names = FALSE, col.names = FALSE, quote = FALSE)

getwd()
