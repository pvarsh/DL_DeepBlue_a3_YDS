set.seed(1235)
x = runif(5)
betas = seq(0.1,500,0.1)

logexp = function(x, beta){
  N = length(x)
  big_sum = sum(exp(x*beta))
  return(1/beta * log(1/N * big_sum))
}

y = rep(0, length(betas))
i = 1
for (beta in betas) {
  y[i] = logexp(x, beta)
  i = i + 1
}
mean(x)
y[1]
max(x)
tail(y,1)
plot(betas, y, type = 'l')