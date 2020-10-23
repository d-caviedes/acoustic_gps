functions {
  matrix sinc(vector[] x,
             real sigma,
             real k) {
    int N = size(x);
    matrix[N, N] K;
    for (i in 1:(N-1)) {
      K[i, i] = sigma^2;
      for (j in (i + 1):N) {
        if(distance(x[i],x[j])==0){
          K[i, j] =  sigma^2;  
        }
        else{
          K[i, j] =  (sigma^2 * bessel_first_kind(0, k * distance(x[i],x[j])));
        }
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = sigma^2;
    return K;
  }
}
data {
  int<lower=0> N_meas;                     // Total Number of independent measurements.
  int<lower=0> N_reps;                     // Repetitions.
  int<lower=0> D;                          // Dimensions                      
  vector[2*N_meas] y[N_reps];           // Measured Pressure at receivers
  vector[D] x[N_meas];
  matrix[2*N_meas, 2*N_meas] Sigma;                     // Noise
  real a_tau_sigma;
  real b_tau_sigma;
  real delta;
  real k;
}
transformed data{
  vector[2*N_meas] mu = rep_vector(0, 2*N_meas);
}
parameters {
  real<lower=0> sigma;
  real<lower=0> tau_sigma;
}
transformed parameters{
  real inv_tau_sigma = 1 ./ tau_sigma;
  matrix[N_meas, N_meas] K_sinc;
  matrix[2*N_meas, 2*N_meas] K;
  matrix[2*N_meas, 2*N_meas] L_K;
  matrix[N_meas, N_meas] K_zeros = rep_matrix(0, N_meas, N_meas);

  K_sinc = sinc(x, sigma, k);
  K = append_row(append_col(K_sinc, K_zeros), 
                 append_col(K_zeros', K_sinc)) +
                 Sigma + 
                 delta;
  L_K = cholesky_decompose(K);
}
model {
  tau_sigma ~ gamma(a_tau_sigma, b_tau_sigma);
  sigma ~ normal(0, inv_tau_sigma);
  for (nrep in 1:N_reps){
      y[nrep] ~ multi_normal_cholesky(mu, L_K);
      }
}
