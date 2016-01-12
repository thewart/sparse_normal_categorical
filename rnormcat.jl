function rnormalprobit(n,d,se,nsamp=1000)
  pi = zeroes(Float64,(d,n));
  eta = Array(Float64,d);
  w = Array(Float64,d);
  for i in 1:n
    rand!(Normal(0,se),eta);
    for j in 1:nsamp
      for k in 1:d w[k] = eta[k] + rand(Normal(0,se)); end
      c = findmax(w)[2];
      pi[c,i] += 1/nsamp
    end
  end
  return y
end

function rnormallogit(n,d,se)
  w = Array(Float64,d);
  for i in 1:n
    rand!(Normal(0,se),w);
    pi[:,i] = exp(w) ./ sum(exp(w));
  end
  return pi
end

