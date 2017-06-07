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
<<<<<<< HEAD
  samples[:u] = Array{Float64}(K-1,n,nsave);
#  samples[:prealpha] = Array{Float64}(K-1,nsave);
#  samples[:postalpha] = Array{Float64}(K-1,nsave);
=======
  #samples[:muB0] = Array{Float64}(nsave);
>>>>>>> 26b95295436ec772ade45fd761a3ca6edb206dee

  z = rand(1:K,n);
  #z = [fill(1,250);fill(2,125);fill(3,125)];
  u = Array{Float64}(K-1,n);
<<<<<<< HEAD
  alpha = randn(K-1)*prior[:sigmaAlpha] + prior[:muAlpha];
  lpk = Array{Float64}(K);
=======
  eta = Array{Float64}(K-1,n);
  B = rand(Normal(),(p,K-1))/2;
  #B = zeros(1,K-1); B[3] = 2;
  #muB0 = 0;
  lpk = Array{Float64}(K,n);

  #precompute some regression shit
  SigmaB = inv( inv(prior[:SigmaB0]) + X*X' );
>>>>>>> 26b95295436ec772ade45fd761a3ca6edb206dee

  #main loop
  for t in 1:iter

    #sample likelihood parameters
    theta = rtheta(y,z,prior);

    #sample group memberships
<<<<<<< HEAD
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
=======
    sample_z!(z,y,eta,theta,loglik,lpk);

    #do the label switch
    for i in 1:round(Int,K/2)
      flip,(j,l) = labelswitch2!(z,lpk,eta);
      if flip
        theta[[j,l]] = theta[[l,j]];
        if l < K
          B[:,[j,l]] = B[:,[l,j]];
          eta[[j,l],:] = eta[[l,j],:];
        end
          lpk[[j,l],:] = lpk[[l,j],:];
      end
    end

>>>>>>> 26b95295436ec772ade45fd761a3ca6edb206dee
    #sample latent utilities
    sample_u!(u,z,alpha);

    #sample betas
<<<<<<< HEAD
    sample_alpha!(alpha,u,prior[:muAlpha],prior[:sigmaAlpha])
=======
    sample_B!(B,u,X,SigmaB,prior[:muB0],prior[:SigmaB0][1]);

    #sample hyperparameters for beta
    muB0 = sample_hyper(B,prior[:SigmaB0][1],prior[:mu_mu0],prior[:sigma_mu0]);
>>>>>>> 26b95295436ec772ade45fd761a3ca6edb206dee

    nsamp = findin(saveiter,t);
    if !isempty(nsamp)
      nsamp = nsamp[1];
      samples[:z][:,nsamp] = z;
      samples[:alpha][:,nsamp] = alpha;
 #     samples[:prealpha][:,nsamp] = prealpha;
 #     samples[:postalpha][:,nsamp] = postalpha;
      linind = (length(theta)*(nsamp-1)+1):(length(theta)*nsamp);
      samples[:theta][linind] = theta;
<<<<<<< HEAD
      samples[:u][:,:,nsamp] = u;
=======
      #samples[:muB0][nsamp] = muB0;
>>>>>>> 26b95295436ec772ade45fd761a3ca6edb206dee
    end
  end

  return samples

end

<<<<<<< HEAD
function sample_z!(z,y,alpha,theta,loglik,lpk)
  n = length(y);
  K = length(alpha) + 1;
=======
function sample_z!(z,y,eta,theta,loglik,lpk)
  (K,n) = size(eta);
  K += 1;
>>>>>>> 26b95295436ec772ade45fd761a3ca6edb206dee
  lpkx = Array(Float64,K);

  #prior weight from psbp
  lpcum = 0.0;
  for k in 1:K
      if k < K
<<<<<<< HEAD
        lpk[k] = normlogcdf(alpha[k]) + lpcum;
        lpcum += normlogccdf(alpha[k]);
      else
        lpk[k] = lpcum;
=======
        lpk[k,i] = normlogcdf(eta[k,i]) + lpcum;
        lpcum += normlogccdf(eta[k,i]);
      else
        lpk[k,i] = lpcum;
>>>>>>> 26b95295436ec772ade45fd761a3ca6edb206dee
      end
  end

  for i in 1:n
    for k in 1:K

      #likelihood
      lpx = loglik(y[i],theta[k]);

<<<<<<< HEAD
      lpkx[k] = lpk[k] + lpx;
=======
      lpkx[k] = lpk[k,i] + lpx;

>>>>>>> 26b95295436ec772ade45fd761a3ca6edb206dee
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

<<<<<<< HEAD
function sample_alpha!(alpha,y,muAlpha,sigmaAlpha)
  K,n = size(y);
  K += 1;
  yhat = mean(y,2);
  csigma = inv( inv(sigmaAlpha) + n );
=======
function sample_B!(B,u,X,SigmaB,muB0,sigmaB0)
  K = size(u)[1]+1;

  muB = SigmaB*(X*u' .+ muB0/sigmaB0);
>>>>>>> 26b95295436ec772ade45fd761a3ca6edb206dee
  for k in 1:(K-1)
    cmu = csigma * (muAlpha/sigmaAlpha + n*yhat[k]);
    alpha[k] = randn()*sqrt(csigma) + cmu;
  end
end

<<<<<<< HEAD
function a2p(alpha)
  K = length(alpha)+1;
=======
function sample_hyper(B,sigmaB0,mu0,sigma0)
  n = length(B);
  yhat = mean(B);
  sigma = 1/(1/sigma0 + n/sigmaB0);
  mu = sigma*(mu0/sigma0 + n*yhat/sigmaB0)
  muB0 = rand(Normal(mu,sqrt(sigma)));

  return muB0
end

function eta2p(x,B)
  eta = B'x;
  K = length(eta)+1;
>>>>>>> 26b95295436ec772ade45fd761a3ca6edb206dee
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
