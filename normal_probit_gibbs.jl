function psbpm(y,loglik,rtheta,prior,burn,thin,iter)

  n = length(y);
  K = prior[:K];

  #initialize
  saveiter = (burn+1):thin:iter;
  nsave = length(saveiter);
  niter = maximum(saveiter);

  samples = Dict{Symbol,Array{Float64}}();
  samples[:theta] = Array{Float64}(
    tuple(vcat(collect(prior[:theta_dim]),K,nsave)...));
  samples[:alpha] = Array{Float64}(K-1,nsave);
  samples[:z] = Array{Int64}(n,nsave);
  samples[:u] = Array{Float64}(K-1,n,nsave);
#  samples[:prealpha] = Array{Float64}(K-1,nsave);
#  samples[:postalpha] = Array{Float64}(K-1,nsave);

  z = rand(1:K,n);
  u = Array{Float64}(K-1,n);
  alpha = randn(K-1)*prior[:sigmaAlpha] + prior[:muAlpha];
  lpk = Array{Float64}(K);

  #main loop
  for t in 1:iter

    #sample likelihood parameters
    theta = rtheta(y,z,prior);

    #sample group memberships
    sample_z!(z,y,alpha,theta,loglik,lpk);
#    prealpha = deepcopy(alpha);
    if prior[:nls] > 0
      for i in 1:prior[:nls]
        flip,(j,l) = labelswitch2!(z,lpk,alpha);
        if flip
          theta[[j,l]] = theta[[l,j]];
          lpk[[j,l]] = lpk[[l,j]];
          if l < K
            alpha[[j,l]] = alpha[[l,j]];
          end
        end
      end
    end
#    postalpha = deepcopy(alpha);
    #sample latent utilities
    sample_u!(u,z,alpha);

    #sample betas
    sample_alpha!(alpha,u,prior[:muAlpha],prior[:sigmaAlpha])

    nsamp = findin(saveiter,t);
    if !isempty(nsamp)
      nsamp = nsamp[1];
      samples[:z][:,nsamp] = z;
      samples[:alpha][:,nsamp] = alpha;
 #     samples[:prealpha][:,nsamp] = prealpha;
 #     samples[:postalpha][:,nsamp] = postalpha;
      linind = (length(theta)*(nsamp-1)+1):(length(theta)*nsamp);
      samples[:theta][linind] = theta;
      samples[:u][:,:,nsamp] = u;
    end
  end

  return samples

end

function sample_z!(z,y,alpha,theta,loglik,lpk)
  n = length(y);
  K = length(alpha) + 1;
  lpkx = Array(Float64,K);

  #prior weight from psbp
  lpcum = 0.0;
  for k in 1:K
      if k < K
        lpk[k] = normlogcdf(alpha[k]) + lpcum;
        lpcum += normlogccdf(alpha[k]);
      else
        lpk[k] = lpcum;
      end
  end

  for i in 1:n
    for k in 1:K

      #likelihood
      lpx = loglik(y[i],theta[k]);

      lpkx[k] = lpk[k] + lpx;
    end

    #normalize and sample category membership z
    lp = lpkx - logsumexp(lpkx);
    z[i] = findfirst(rand(Multinomial(1,exp(lp))));
  end
end

function sample_u!(u,z,alpha)
  (K,n) = size(u);
  K += 1;
  for i in 1:n
    for k in 1:(K-1)

      if k < z[i]
        u[k,i] = rand(TruncatedNormal(alpha[k],1,-Inf,0));
      elseif (k == z[i])
        u[k,i] = rand(TruncatedNormal(alpha[k],1,0,Inf));
      elseif k > z[i]
        u[k,i] = rand(Normal(alpha[k],1));
      end

    end
  end
end

function sample_alpha!(alpha,y,muAlpha,sigmaAlpha)
  K,n = size(y);
  K += 1;
  yhat = mean(y,2);
  csigma = inv( inv(sigmaAlpha) + n );
  for k in 1:(K-1)
    cmu = csigma * (muAlpha/sigmaAlpha + n*yhat[k]);
    alpha[k] = randn()*sqrt(csigma) + cmu;
  end
end

function a2p(alpha)
  K = length(alpha)+1;
  p = Array{Float64}(K);
  pcol = 1;
  for k in 1:(K-1)
    pi = normcdf(alpha[k]);
    p[k] = pi * pcol;
    pcol *= 1-pi;
  end
  p[K] = pcol;
  return p
end

function labelswitch2!(z,lpk,alpha)
  K = size(lpk)[1];
  j::Int = sample(1:(K-1));
  l = j+1;
  zj = find(z.==j);
  zl = find(z.==l);
  p = 0.0;

  for i in zl
    p -= normlogccdf(alpha[j]);
  end

  if l < K
    for i in zj
      p += normlogcdf(alpha[l]);
    end
  end

  flip = rand() < exp(p);
  if flip
      z[zj] = l;
      z[zl] = j;
  end

  return flip,(j,l)
end
