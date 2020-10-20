functions {
  matrix sinc(vector[] x,
             real alpha,
             real k) {
    int N = size(x);
    matrix[N, N] K;
    for (i in 1:(N-1)) {
      K[i, i] = alpha^2;
      for (j in (i + 1):N) {
        if(distance(x[i],x[j])==0){
          K[i, j] =  alpha^2;  
        }
        else{
          K[i, j] =  (alpha^2 * bessel_first_kind(0, k * distance(x[i],x[j])));
        }
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = alpha^2;
    return K;
  }
}
data {
  int<lower=0> N_meas;                     // Total Number of independent measurements.
  int<lower=0> N_reps;                     // Repetitions.
  int<lower=0> D;                          // Dimensions                      
  vector[2*N_meas] y[N_reps];           // Measured Pressure at receivers
  vector[D] x[N_meas];
  real<lower=0> sigma;                     // Noise
  real a_tau_alpha;
  real b_tau_alpha;
  real delta;
  real k;
}
transformed data{
  vector[2*N_meas] mu = rep_vector(0, 2*N_meas);
}
parameters {
  real<lower=0> alpha;
  real<lower=0> tau_alpha;
}
transformed parameters{
  real inv_tau_alpha = 1 ./ tau_alpha;
  matrix[N_meas, N_meas] K_sinc;
  matrix[2*N_meas, 2*N_meas] K;
  matrix[2*N_meas, 2*N_meas] L_K;
  matrix[N_meas, N_meas] K_zeros = rep_matrix(0, N_meas, N_meas);

  K_sinc = sinc(x, alpha, k);
  K = append_row(append_col(K_sinc, K_zeros), 
                 append_col(K_zeros', K_sinc)) +
                 diag_matrix(rep_vector(square(sigma), 2*N_meas)) + 
                 delta;
  L_K = cholesky_decompose(K);
}
model {
  tau_alpha ~ gamma(a_tau_alpha, b_tau_alpha);
  alpha ~ normal(0, inv_tau_alpha);
  for (nrep in 1:N_reps){
      y[nrep] ~ multi_normal_cholesky(mu, L_K);
      }
}
