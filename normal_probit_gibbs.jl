using Distributions,Gadfly,StatsFuns

n = 60;
y = [rand(Poisson(1),div(n,3)); rand(Poisson(3),div(n,3));rand(Poisson(5),div(n,3))];
X = ones(n);

#priors
K = 10;
a0 = 1;
b0 = 1/3;
SigmaB0 = eye(1);
muB0 = 0;

Norm1 = Normal(0,1);

#initialize
z = Array(Int64,n);
lambda = rand(Gamma(a0,inv(b0)),K);
eta = rand(Norm1,K-1)/2;

#precompute some regression shit
SigmaB = inv( inv(SigmaB0) + X'X );

#sample group memberships
lpkx = Array(Float64,K);
for i in 1:n

  lpcum = 0;
  for k in 1:K
    #prior weight from psbp
    if k < K
      lpk = logcdf(Norm1,eta[k]) + lpcum;
      lpcum += logccdf(Norm1,eta[k]);
    else
      lpk = lpcum;
    end

    #likelihood
    lpx = logpdf(Poisson(lambda[k]),y[i]);

    lpkx[k] = lpk + lpx;
  end

  #normalize and sample category membership z
  lp = lpkx - logsumexp(lpkx);
  z[i] = findfirst(rand(Multinomial(1,exp(lp))));
end

#sample poisson rates
sx = zeros(Int64,K);
nx = zeros(Int64,K);
for i in 1:n
  k = z[i];
  sx[k] += y[i];
  nx[k] += 1;
end

for k in 1:K
  lambda[k] = rand(Gamma(a0 + sx[k], inv( inv(b0) + nx[k])));
end

#sample latent utilities
u = Array(Float64,(K-1,n));
for i in 1:n
  for k in 1:(K-1)

    if k < z[i]
      u[k,i] = rand(TruncatedNormal(eta[k],1,-Inf,0));
    elseif (k == z[i])
      u[k,i] = rand(TruncatedNormal(eta[k],1,0,Inf));
    elseif k > z[i]
      u[k,i] = rand(Normal(eta[k],1));
    end

  end
end

u
z
#sample eta
muB = u*X';
for i in 1:K
  eta[k] = rand(Normal(muB[k],SigmaB),1);
end
