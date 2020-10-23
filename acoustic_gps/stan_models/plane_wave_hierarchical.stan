functions {
  matrix cosine_kernel(vector[] x,
                       vector sigma_l,
                       real k) {
    int N = size(x);
    int D = num_elements(x[1]);
    matrix[N, N] K;
    for (i in 1:(N-1)) {
      K[i, i] = sum(sigma_l)/2;
      for (j in (i + 1):N) {
        K[i, j] = 0;
        for (d in 1:D){
          K[i, j] +=  (sigma_l[d]/2) * cos(k  * (x[i, d]-x[j, d]));
        }
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = sum(sigma_l)/2;
    return K;
  }
  matrix sine_kernel(vector[] x,
                            vector sigma_l,
                            real k) {
    int N = size(x);
    int D = num_elements(x[1]);
    matrix[N, N] K;
    for (i in 1:(N-1)) {
      K[i, i] = 0;
      for (j in (i + 1):N) {
        K[i, j] = 0;
        for (d in 1:D){
          K[i, j] +=  (sigma_l[d]/2) * (sin(k  * (x[i, d]-x[j, d])));
        }
        K[j, i] = -K[i, j];
      }
    }
    K[N, N] = 0;
    return K;
  }
}
data {
  int<lower=0> N_meas;                     // Number of measurement positions.
  int<lower=0> N_reps;                     // Repetitions.
  int<lower=0> D;                          // Number of basis elements                    
  vector[2*N_meas] y[N_reps];              // Measured Pressure at receiver. Real and imag stacked.
  vector[2] x[N_meas];                     // Spatial positions
  vector[2] wave_directions[D];            // possible plane wave directions
  matrix[2*N_meas, 2*N_meas] Sigma;
  real k;
  real a;
  real b_log_std;
  real b_log_mean;
  real delta;
}
transformed data{
  vector[2*N_meas] mu = rep_vector(0, 2*N_meas);
  vector[D] x_projected[N_meas];
  for (i in 1:N_meas){
    for (d in 1:D){
      x_projected[i, d] = dot_product(x[i], wave_directions[d]);
    }
  }
}
parameters {
  vector<lower=0>[D] sigma_l;
  real<lower=0> b_log;
}
transformed parameters{
  matrix[N_meas, N_meas] K_self;
  matrix[N_meas, N_meas] K_realimag;
  matrix[2*N_meas, 2*N_meas] K;
  matrix[2*N_meas, 2*N_meas] L_K;
  real<lower=0> b = pow(10, -b_log);
  K_self = cosine_kernel(x_projected, sigma_l, k);
  K_realimag = sine_kernel(x_projected, sigma_l, k);
  K = append_row(append_col(K_self, K_realimag), 
                 append_col(K_realimag', K_self)) +
                 Sigma + 
                 delta;
  L_K = cholesky_decompose(K);
}
model {
  sigma_l ~ inv_gamma(a, b);
  b_log ~ normal(b_log_mean, b_log_std);
  for (nrep in 1:N_reps){
      y[nrep] ~ multi_normal_cholesky(mu, L_K);
      }
}
