using Distributions
using Gadfly

n = 60;
X = [rand(Poisson(1),div(n,3)); rand(Poisson(3),div(n,3));rand(Poisson(5),div(n,3))];

K = 10;
lambda0 = Gamma(1,3);
alpha0 = Normal(0,1);
Norm1 = Normal(0,1);

lambda = rand(lambda0,K);
alpha = rand(alpha0,K);
u = rand(Norm1,(K,n)) .+ alpha;
z = Array(Int64,n);

for i in 1:n

  pcum = 1;
  compsum = 1;
  psum = 1;
  for k in 1:K
    if k < K
      phi = cdf(Norm1,u[k,i])
      pk = phi * pcum;
    else
      pk = pcum;
    end

    if rand() <= pk/psum
      z[i] = k;
      break
    else
      psum -= pk;
      pcum *= 1-phi;
    end
  end

end

plot(x=z,Geom.histogram)
