using Distributions,Gadfly,StatsFuns

n = 60;
y = [rand(Poisson(1),div(n,3)); rand(Poisson(3),div(n,3));rand(Poisson(5),div(n,3))];
X = ones(n);


function psbpm(y,X,loglik,rtheta,prior,burn,thin,iter)

  #priors
  K = prior[:K];
  #a0 = 1;
  #b0 = 1/3;
  #SigmaB0 = eye(1);
  #muB0 = 0;

  #initialize
  saveiter = (burn+1):thin:iter;
  nsave = length(saveiter);
  niter = maximum(saveiter);

  samples = Dict{Symbol,Array{Float64}}();
  samples[:theta] = Array{Float64}(
    tuple(vcat(collect(size(theta0)),nsave)...));
  samples[:eta] = Array{Float64}(K-1,nsave);
  samples[:z] = Array{Int64}(n,nsave);

  z = Array{Int64}(n);
  u = Array{Float64}
  eta = rand(Normal(),K-1)/2;

  #precompute some regression shit
  SigmaB = inv( inv(prior[:SigmaB0]) + X'X );

  #main loop
  for t in 1:iter

    #sample group memberships
    sample_z!(z,y,eta,theta,loglik);

    #sample likelihood parameters
    lambda = rtheta(z,prior);

    #sample latent utilities
    sample_u!(u,z,eta);

    #sample eta
    eta = sample_eta(u,X,SigmaB);

    nsamp = findin(saveiter,t);
    if !isempty(nsamp)
      samples[:z][:,nsamp] = z;
      samples[:eta][:,nsamp] = eta;
      linind = (length(theta)*(nsamp-1)+1):(length(theta)*nsamp);
      samples[:theta][linind] = theta;
    end
  end

  return samples

end

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
function sample_poisson(z,prior)
  K = prior[:K];
  sz = zeros(Int64,K);
  nz = zeros(Int64,K);
  n = length(z);

  for i in 1:n
    k = z[i];
    sz[k] += y[i];
    nz[k] += 1;
  end

  for k in 1:K
    lambda[k] = rand(Gamma(prior[:a0] + sx[k], inv(prior[:b0] + nx[k])));
  end
end

function sample_u!(u,z,eta)
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

