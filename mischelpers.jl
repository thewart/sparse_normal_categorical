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

#sample poisson set
function sample_multipoisson(Y,z,prior)
  K = prior[:K];
  d = length(Y[1]);
  n = length(z);
  sz = zeros(Int64,d,K);
  nz = zeros(Int64,K);
  n = length(z);
  lambda = Array(Float64,K);

  for i in 1:n
    k = z[i];
    nz[k] += 1;
    for j in 1:d
      sz[k,j] += Y[i][j];
    end
  end

  for k in 1:K
    for j in 1:d
      lambda[k] = rand(Gamma(prior[:a0] + sz[k], inv(prior[:b0] + nz[k])));
    end
  end

  return lambda;
end
