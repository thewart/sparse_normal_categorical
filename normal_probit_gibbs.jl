using Distributions,Gadfly,StatsFuns

n = 60;
y = [rand(Poisson(1),div(n,3)); rand(Poisson(3),div(n,3));rand(Poisson(5),div(n,3))];
X = ones(n);


function psbpm(y,X,loglik,prior,burn,thin,iter)

  #priors
  K = 10;
  a0 = 1;
  b0 = 1/3;
  SigmaB0 = eye(1);
  muB0 = 0;

  #initialize
  samples = Dict{Symbol,Array{Float64}}();
  z = Array(Int64,n);
  eta = rand(Normal(),K-1)/2;

  #precompute some regression shit
  SigmaB = inv( inv(SigmaB0) + X'X );

  #main loop
  for t in 1:iter

  end

  return samples

end

  #sample group memberships
function sample_z!(z,y,eta,theta,loglik)
  K = length(eta) + 1;
  n = size(y)[1];
  lpkx = Array(Float64,K);

  for i in 1:n
    lpcum = 0;
    for k in 1:K
      #prior weight from psbp
      if k < K
        lpk = logcdf(Normal(),eta[k]) + lpcum;
        lpcum += logccdf(Normal(),eta[k]);
      else
        lpk = lpcum;
      end

      #likelihood
      lpx = loglik(y[i],lambda[k]);

      lpkx[k] = lpk + lpx;
    end

    #normalize and sample category membership z
    lp = lpkx - logsumexp(lpkx);
    z[i] = findfirst(rand(Multinomial(1,exp(lp))));
  end
end

#sample poisson rates
function sample_poisson(z,a0,b0,K)
  sx = zeros(Int64,K);
  nx = zeros(Int64,K);
  n = length(z);

  for i in 1:n
    k = z[i];
    sx[k] += y[i];
    nx[k] += 1;
  end

  for k in 1:K
    lambda[k] = rand(Gamma(a0 + sx[k], inv( inv(b0) + nx[k])));
  end
end

#sample latent utilities
function sample_u!(u,eta)
  (K,n) = size(u);

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
end

#sample eta
function sample_eta(u,X,SigmaB)
  K = size(u)[1]+1;
  eta = Array(Float64,K-1);
  muB = SigmaB.*(u*X);
  for k in 1:(K-1)
    eta[k] = rand(Normal(muB[k],sqrt(SigmaB[1])));
  end
end

