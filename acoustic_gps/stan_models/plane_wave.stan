functions {
  matrix cosine_kernel(vector[] x,
                       vector alpha,
                       real k) {
    int N = size(x);
    int D = num_elements(x[1]);
    matrix[N, N] K;
    for (i in 1:(N-1)) {
      K[i, i] = sum(alpha)/2;
      for (j in (i + 1):N) {
        K[i, j] = 0;
        for (d in 1:D){
          K[i, j] +=  (alpha[d]/2) * cos(k  * (x[i, d]-x[j, d]));
        }
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = sum(alpha)/2;
    return K;
  }
  matrix sine_kernel(vector[] x,
                            vector alpha,
                            real k) {
    int N = size(x);
    int D = num_elements(x[1]);
    matrix[N, N] K;
    for (i in 1:(N-1)) {
      K[i, i] = 0;
      for (j in (i + 1):N) {
        K[i, j] = 0;
        for (d in 1:D){
          K[i, j] +=  (alpha[d]/2) * (sin(k  * (x[i, d]-x[j, d])));
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
  real<lower=0> sigma;
  real k;
  real a_tau;
  vector[D] b_tau;
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
  vector<lower=0>[D] alpha;
  vector<lower=0>[D] tau;
}
transformed parameters{
  vector[D] inv_tau = 1 ./ tau;
  matrix[N_meas, N_meas] K_self;
  matrix[N_meas, N_meas] K_realimag;
  matrix[2*N_meas, 2*N_meas] K;
  matrix[2*N_meas, 2*N_meas] L_K;
  K_self = cosine_kernel(x_projected, alpha, k);
  K_realimag = sine_kernel(x_projected, alpha, k);
  K = append_row(append_col(K_self, K_realimag), 
                 append_col(K_realimag', K_self)) +
                 diag_matrix(rep_vector(square(sigma), 2*N_meas)) + 
                 delta;
  L_K = cholesky_decompose(K);
}
model {
  tau ~ gamma(a_tau, b_tau);
  alpha ~ normal(0, inv_tau);
  for (nrep in 1:N_reps){
      y[nrep] ~ multi_normal_cholesky(mu, L_K);
      }
}
