#ifndef LOCALVQE_AEC_FRONTEND_H
#define LOCALVQE_AEC_FRONTEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque state for the LocalVQE v1.4 adaptive echo-cancellation front end.
typedef struct localvqe_aec_daf localvqe_aec_daf;

/// Number of Float32 controller weights expected by `localvqe_aec_daf_create`.
size_t localvqe_aec_daf_weight_count(void);

/// Create the front end from the flattened 2,742-parameter controller.
/// Returns NULL when the pointer is NULL or the count is invalid.
localvqe_aec_daf *localvqe_aec_daf_create(
    const float *weights,
    size_t count
);

void localvqe_aec_daf_destroy(localvqe_aec_daf *frontend);
void localvqe_aec_daf_reset(localvqe_aec_daf *frontend);
void localvqe_aec_daf_set_prealignment(
    localvqe_aec_daf *frontend,
    bool enabled
);

/// Estimate and freeze bulk reference delay from a complete synchronized clip.
bool localvqe_aec_daf_prime_delay(
    localvqe_aec_daf *frontend,
    const float *microphone,
    const float *reference,
    size_t sample_count
);

/// Process a sample count divisible by 128. `residual` and `echo_estimate`
/// must each have room for `sample_count` Float32 values.
bool localvqe_aec_daf_process(
    localvqe_aec_daf *frontend,
    const float *microphone,
    const float *reference,
    size_t sample_count,
    float *residual,
    float *echo_estimate
);

int32_t localvqe_aec_daf_current_delay_samples(
    const localvqe_aec_daf *frontend
);

float localvqe_aec_daf_delay_confidence(
    const localvqe_aec_daf *frontend
);

#ifdef __cplusplus
}
#endif

#endif
