function labelswitch2!(z,lpk,eta)
  K = size(lpk)[1];
  j::Int = sample(1:(K-1));
  l = j+1;
  zj = find(z.==j);
  zl = find(z.==l);

  plj = 0.0;
  pll = 0.0;
  for i in zl
    plj += lpk[l,i] - normlogccdf(eta[j,i]);
    pll += lpk[l,i];
  end

  pjl = 0.0;
  pjj = 0.0;
  for i in zj
    if l < K
      pjl += lpk[j,i] + normlogcdf(eta[l,i]);
    else
      pjl += lpk[l,i];
    end
    pjj += lpk[j,i];
  end

  flip = rand() < exp(plj + pjl - pll - pjj);
  if flip
      z[zj] = l;
      z[zl] = j;
  end

  return flip,(j,l)
end

function labelswitch1!(z,lpk,theta)
  n = length(z);
  K = size(lpk)[1];
  j,l = sample(1:K,2,replace=false);
  zj = find(z.==j);
  zl = find(z.==l);

  pjj = 0;
  pjl = 0;
  plj = 0;
  pll = 0;

  for i in zj
    pjj += lpk[j,i];
    pjl += lpk[l,i];
  end

  for i in zl
    pll += lpk[l,i];
    plj += lpk[j,i];
  end

  p = exp(plj + pjl - pll - pjj);
  if rand() < p
    z[zj] = l;
    z[zl] = j;
    theta[[j,l]] = theta[[l,j]];
    lpk[[j,l],:] = lpk[[l,j],:];
  end

end