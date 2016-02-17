function psbpm(y,X,loglik,rtheta,prior,burn,thin,iter)

  n = length(y);
  K = prior[:K];
  if (isempty(X))
    X = ones(1,n);
  end
  p = size(X)[1];

  #initialize
  saveiter = (burn+1):thin:iter;
  nsave = length(saveiter);
  niter = maximum(saveiter);

  samples = Dict{Symbol,Array{Float64}}();
  samples[:theta] = Array{Float64}(
    tuple(vcat(collect(prior[:theta_dim]),nsave)...));
  samples[:B] = Array{Float64}(p,K-1,nsave);
  samples[:z] = Array{Int64}(n,nsave);

  z = rand(1:K,n);
  u = Array{Float64}(K-1,n);
  eta = Array{Float64}(K-1,n);
  B = rand(Normal(),(p,K-1))/2;

  #precompute some regression shit
  SigmaB = inv( inv(prior[:SigmaB0]) + X*X' );

  #main loop
  for t in 1:iter

    @into! eta = B'*X;

    #sample likelihood parameters
    theta = rtheta(z,prior);

    #sample group memberships
    sample_z!(z,y,eta,theta,loglik);

    #sample latent utilities
    sample_u!(u,z,eta);

    #sample betas
    sample_B!(B,u,X,SigmaB);

    nsamp = findin(saveiter,t);
    if !isempty(nsamp)
      nsamp = nsamp[1];
      samples[:z][:,nsamp] = z;
      samples[:B][:,:,nsamp] = B;
      linind = (length(theta)*(nsamp-1)+1):(length(theta)*nsamp);
      samples[:theta][linind] = theta;
    end
  end

  return samples

end

function sample_z!(z,y,eta,theta,loglik)
  (K,n) = size(eta);
  K += 1;
  lpkx = Array(Float64,K);
  for i in 1:n
    lpcum = 0.0;
    for k in 1:K
      #prior weight from psbp
      if k < K
        lpk = normlogcdf(eta[k,i]) + lpcum;
        lpcum += normlogccdf(eta[k,i]);
      else
        lpk = lpcum;
      end

      #likelihood
      lpx = loglik(y[i],theta[k]);

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
  lambda = Array(Float64,K);

  for i in 1:n
    k = z[i];
    sz[k] += y[i];
    nz[k] += 1;
  end

  for k in 1:K
    lambda[k] = rand(Gamma(prior[:a0] + sz[k], inv(prior[:b0] + nz[k])));
  end

  return lambda;
end

function sample_u!(u,z,eta)
  (K,n) = size(u);
  K += 1;
  for i in 1:n
    for k in 1:(K-1)

      if k < z[i]
        u[k,i] = rand(TruncatedNormal(eta[k,i],1,-Inf,0));
      elseif (k == z[i])
        u[k,i] = rand(TruncatedNormal(eta[k,i],1,0,Inf));
      elseif k > z[i]
        u[k,i] = rand(Normal(eta[k,i],1));
      end

    end
  end
end

function sample_B!(B,u,X,SigmaB)
  K = size(u)[1]+1;

  muB = SigmaB*X*u';
  for k in 1:(K-1)
    B[:,k] = rand(MvNormal(muB[:,k],SigmaB));
  end
end

function eta2p(x,B)
  eta = B'x;
  K = length(eta)+1;
  p = Array{Float64}(K);
  pcol = 1;
  for k in 1:(K-1)
    pi = normcdf(eta[k]);
    p[k] = pi * pcol;
    pcol *= 1-pi;
  end
  p[K] = pcol;
  return p
end
