functions {
  matrix rbf(vector[] x,
             real alpha,
             vector rho) {
    int N = size(x);
    int D = num_elements(x[1]);
    real exponent;
    matrix[N, N] K;
    for (i in 1:(N-1)) {
      K[i, i] = alpha^2;
      for (j in (i + 1):N) {
        exponent = 0;
        for (d in 1:D){
          exponent +=  1/(rho[d]^2)*(x[i, d] - x[j, d])^2;
        }
        K[i, j] =  alpha^2 * exp(-0.5 * exponent);
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
  vector[2] directions[D];            // possible plane wave directions
  vector[2] x[N_meas];
  real<lower=0> sigma;                     // Noise
  real a_tau_rho;
  vector[D] b_tau_rho;
  real a_tau_alpha;
  real b_tau_alpha;
  real delta;
}
transformed data{
  vector[2*N_meas] mu = rep_vector(0, 2*N_meas);
  vector[D] x_projected[N_meas];
  for (i in 1:N_meas){
    for (d in 1:D){
      x_projected[i, d] = dot_product(x[i], directions[d]);
    }
  }
}
parameters {
  real<lower=0> alpha;
  vector<lower=0>[D] rho;
  real<lower=0> tau_alpha;
  vector<lower=0>[D] tau_rho;
}
transformed parameters{
  vector[D] inv_tau_rho = 1 ./ tau_rho;
  real inv_tau_alpha = 1 ./ tau_alpha;
  matrix[N_meas, N_meas] K_rbf;
  matrix[2*N_meas, 2*N_meas] K;
  matrix[2*N_meas, 2*N_meas] L_K;
  matrix[N_meas, N_meas] K_zeros = rep_matrix(0, N_meas, N_meas);

  K_rbf = rbf(x_projected, alpha, rho);
  K = append_row(append_col(K_rbf, K_zeros), 
                 append_col(K_zeros', K_rbf)) +
                 diag_matrix(rep_vector(square(sigma), 2*N_meas)) + 
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
