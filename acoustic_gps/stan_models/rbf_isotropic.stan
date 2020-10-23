data {
  int<lower=0> N_meas;                     // Total Number of independent measurements.
  int<lower=0> N_reps;                     // Repetitions.
  int<lower=0> D;                          // Dimensions                      
  vector[2*N_meas] y[N_reps];           // Measured Pressure at receivers
  vector[D] x[N_meas];
  matrix[2*N_meas, 2*N_meas] Sigma;                     // Noise
  real a_tau_rho;
  real b_tau_rho;
  real a_tau_alpha;
  real b_tau_alpha;
  real delta;
}
transformed data{
  vector[2*N_meas] mu = rep_vector(0, 2*N_meas);
}
parameters {
  real<lower=0> alpha;
  real<lower=0> rho;
  real<lower=0> tau_rho;
  real<lower=0> tau_alpha;
}
transformed parameters{
  real<lower = 0> inv_tau_rho = 1 ./ tau_rho;
  real<lower = 0> inv_tau_alpha = 1 ./ tau_alpha;
  matrix[N_meas, N_meas] K_real;
  matrix[2*N_meas, 2*N_meas] K;
  matrix[2*N_meas, 2*N_meas] L_K;
  matrix[N_meas, N_meas] K_zeros = rep_matrix(0, N_meas, N_meas);

  K_real = cov_exp_quad(x, alpha, rho);
  K = append_row(append_col(K_real, K_zeros), 
                 append_col(K_zeros', K_real)) +
                 Sigma + 
                 delta;
  L_K = cholesky_decompose(K);
}
model {
  tau_alpha ~ gamma(a_tau_alpha, b_tau_alpha);
  tau_rho ~ gamma(a_tau_rho, b_tau_rho);
  rho ~ normal(0, inv_tau_rho);
  alpha ~ normal(0, inv_tau_alpha);
  for (nrep in 1:N_reps){
      y[nrep] ~ multi_normal_cholesky(mu, L_K);
      }
}
