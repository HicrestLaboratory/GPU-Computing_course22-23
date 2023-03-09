#define MU_SOL \
double mu_fn_sol(dtype *v, int len) {   \
                                        \
    double mu = 0.0;                    \
    for (int i=0; i<len; i++)           \
        mu += (double)v[i];             \
    mu /= (double)len;                  \
                                        \
    return(mu);                         \
}

#define SIGMA_SOL \
double sigma_fn_sol(dtype *v, double mu, int len) {         \
                                                            \
    double sigma = 0.0;                                     \
    for (int i=0; i<len; i++) {                             \
        sigma += ((double)v[i] - mu)*((double)v[i] - mu);   \
    }                                                       \
    sigma /= (double)len;                                   \
                                                            \
    return(sigma);                                          \
}
