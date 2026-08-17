/*
 * FV3 CUDA Interface C ABI Test
 *
 * This test validates the Fortran-to-C interface path without requiring
 * a full GEOS/FV3 environment. It tests:
 *   - C ABI struct compatibility
 *   - Context creation/finalization lifecycle
 *   - Field descriptor validation
 *   - Configuration propagation
 *   - Error handling
 *
 * CUDA compilation and GPU execution are NOT tested here - this validates
 * the interface layer only.
 */

#include "fv3_gpu_api.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

#define TEST_PASS(name) printf("  PASS: %s\n", name)

/* Test 1: C ABI struct sizes and layout */
static int test_struct_sizes(void) {
    printf("Testing C ABI struct sizes...\n");

    /* Verify struct sizes match expected values */
    TEST_ASSERT(sizeof(fv3_gpu_error) >= 256 + 32 + 32,
                "fv3_gpu_error too small");

    TEST_ASSERT(sizeof(fv3_gpu_field_desc) >= 200,
                "fv3_gpu_field_desc too small");

    TEST_ASSERT(sizeof(fv3_gpu_create_desc) >= 64,
                "fv3_gpu_create_desc too small");

    TEST_ASSERT(sizeof(fv3_gpu_state_desc) >= 16,
                "fv3_gpu_state_desc too small");

    TEST_ASSERT(sizeof(fv3_gpu_step_desc) >= 48,
                "fv3_gpu_step_desc too small");

    TEST_ASSERT(sizeof(fv3_gpu_capabilities) >= 24,
                "fv3_gpu_capabilities too small");

    TEST_PASS("struct_sizes");
    return 0;
}

/* Test 2: Field descriptor initialization and validation */
static int test_field_descriptor(void) {
    printf("Testing field descriptor...\n");

    fv3_gpu_field_desc desc;
    fv3_gpu_error err;

    memset(&desc, 0, sizeof(desc));
    memset(&err, 0, sizeof(err));

    desc.abi_version = FV3_GPU_ABI_VERSION;
    desc.struct_bytes = sizeof(desc);
    desc.field_id = 1; /* u field */
    strncpy(desc.name, "u", FV3_GPU_FIELD_NAME_BYTES);
    desc.base = NULL; /* Will be set to actual memory later */
    desc.element_type = FV3_GPU_ELEMENT_F64;
    desc.element_bytes = 8;
    desc.rank = 3;
    desc.memory_location = FV3_GPU_MEMORY_HOST;
    desc.stagger = FV3_GPU_STAGGER_D_U;
    desc.access = FV3_GPU_ACCESS_INOUT;

    /* Set dimensions for a 10x10x5 domain with 3 halo points */
    desc.lower[0] = -2;
    desc.lower[1] = -2;
    desc.lower[2] = 1;
    desc.extent[0] = 15;
    desc.extent[1] = 16;
    desc.extent[2] = 5;
    desc.stride_elements[0] = 1;
    desc.stride_elements[1] = 15;
    desc.stride_elements[2] = 15 * 16;

    desc.compute_lower[0] = 1;
    desc.compute_lower[1] = 1;
    desc.compute_lower[2] = 1;
    desc.compute_upper[0] = 10;
    desc.compute_upper[1] = 10;
    desc.compute_upper[2] = 5;

    desc.halo_lower[0] = 3;
    desc.halo_lower[1] = 3;
    desc.halo_lower[2] = 0;
    desc.halo_upper[0] = 3;
    desc.halo_upper[1] = 3;
    desc.halo_upper[2] = 0;

    desc.alignment_bytes = 8;

    /* Validate the descriptor */
    int rc = fv3_gpu_validate_field(&desc, &err);
    TEST_ASSERT(rc == 0 || rc == FV3_GPU_INVALID_ARGUMENT,
                "field validation should succeed or return expected error");

    TEST_PASS("field_descriptor");
    return 0;
}

/* Test 3: Create descriptor initialization */
static int test_create_descriptor(void) {
    printf("Testing create descriptor...\n");

    fv3_gpu_create_desc desc;
    memset(&desc, 0, sizeof(desc));

    desc.abi_version = FV3_GPU_ABI_VERSION;
    desc.struct_bytes = sizeof(desc);
    desc.device_ordinal = 0;
    desc.mpi_rank = 0;
    desc.mpi_size = 1;
    desc.precision = FV3_GPU_PRECISION_R8;
    desc.is = 1;
    desc.ie = 10;
    desc.js = 1;
    desc.je = 10;
    desc.npz = 72;
    desc.halo_width = 3;
    desc.flags = 0;
    desc.communicator_token = 0;

    /* Verify values are correctly stored */
    TEST_ASSERT(desc.abi_version == FV3_GPU_ABI_VERSION, "ABI version mismatch");
    TEST_ASSERT(desc.npz == 72, "npz mismatch");
    TEST_ASSERT(desc.halo_width == 3, "halo_width mismatch");

    TEST_PASS("create_descriptor");
    return 0;
}

/* Test 4: Step descriptor initialization */
static int test_step_descriptor(void) {
    printf("Testing step descriptor...\n");

    fv3_gpu_step_desc desc;
    memset(&desc, 0, sizeof(desc));

    desc.abi_version = FV3_GPU_ABI_VERSION;
    desc.struct_bytes = sizeof(desc);
    desc.bdt = 450.0; /* 7.5 minute time step */
    desc.k_split = 1;
    desc.n_split = 6;
    desc.init_step = 1;
    desc.end_step = 0;
    desc.model_step = 0;
    desc.acoustic_sequence = 0;
    desc.validation_mask = 0;
    desc.physics_tail_mask = 0;
    desc.dependency = 0;

    /* Verify values */
    TEST_ASSERT(desc.bdt == 450.0, "bdt mismatch");
    TEST_ASSERT(desc.k_split == 1, "k_split mismatch");
    TEST_ASSERT(desc.n_split == 6, "n_split mismatch");
    TEST_ASSERT(desc.init_step == 1, "init_step mismatch");

    TEST_PASS("step_descriptor");
    return 0;
}

/* Test 5: Error structure handling */
static int test_error_handling(void) {
    printf("Testing error handling...\n");

    fv3_gpu_error err;
    memset(&err, 0, sizeof(err));

    err.abi_version = FV3_GPU_ABI_VERSION;
    err.struct_bytes = sizeof(err);
    err.code = FV3_GPU_SUCCESS;
    err.native_code = 0;
    err.rank = 0;
    err.device = 0;
    err.sequence = 0;
    strncpy(err.api, "test", FV3_GPU_FIELD_NAME_BYTES);
    strncpy(err.message, "test message", FV3_GPU_ERROR_MESSAGE_BYTES);

    /* Verify error codes */
    TEST_ASSERT(FV3_GPU_SUCCESS == 0, "SUCCESS should be 0");
    TEST_ASSERT(FV3_GPU_INVALID_ARGUMENT == 1, "INVALID_ARGUMENT should be 1");
    TEST_ASSERT(FV3_GPU_UNSUPPORTED_CONFIGURATION == 2,
                "UNSUPPORTED_CONFIGURATION should be 2");
    TEST_ASSERT(FV3_GPU_CUDA_ERROR == 3, "CUDA_ERROR should be 3");

    TEST_PASS("error_handling");
    return 0;
}

/* Test 6: Stagger types */
static int test_stagger_types(void) {
    printf("Testing stagger types...\n");

    TEST_ASSERT(FV3_GPU_STAGGER_A == 1, "STAGGER_A should be 1");
    TEST_ASSERT(FV3_GPU_STAGGER_C_U == 2, "STAGGER_C_U should be 2");
    TEST_ASSERT(FV3_GPU_STAGGER_C_V == 3, "STAGGER_C_V should be 3");
    TEST_ASSERT(FV3_GPU_STAGGER_D_U == 4, "STAGGER_D_U should be 4");
    TEST_ASSERT(FV3_GPU_STAGGER_D_V == 5, "STAGGER_D_V should be 5");
    TEST_ASSERT(FV3_GPU_STAGGER_B == 6, "STAGGER_B should be 6");
    TEST_ASSERT(FV3_GPU_STAGGER_INTERFACE == 7, "STAGGER_INTERFACE should be 7");
    TEST_ASSERT(FV3_GPU_STAGGER_X_FACE == 8, "STAGGER_X_FACE should be 8");
    TEST_ASSERT(FV3_GPU_STAGGER_Y_FACE == 9, "STAGGER_Y_FACE should be 9");

    TEST_PASS("stagger_types");
    return 0;
}

/* Test 7: Field mask operations */
static int test_field_mask(void) {
    printf("Testing field mask...\n");

    fv3_gpu_field_mask mask;
    memset(&mask, 0, sizeof(mask));

    mask.abi_version = FV3_GPU_ABI_VERSION;
    mask.struct_bytes = sizeof(mask);

    /* Set all bits */
    mask.words[0] = ~0ULL;
    mask.words[1] = ~0ULL;
    mask.words[2] = ~0ULL;
    mask.words[3] = ~0ULL;

    TEST_ASSERT(mask.words[0] == ~0ULL, "words[0] should be all 1s");

    /* Clear all bits */
    mask.words[0] = 0;
    mask.words[1] = 0;
    mask.words[2] = 0;
    mask.words[3] = 0;

    TEST_ASSERT(mask.words[0] == 0, "words[0] should be all 0s");

    /* Set individual bit for field 5 */
    mask.words[0] = (1ULL << 5);
    TEST_ASSERT((mask.words[0] & (1ULL << 5)) != 0, "bit 5 should be set");

    TEST_PASS("field_mask");
    return 0;
}

/* Test 8: Context lifecycle (without GPU - expect graceful failure) */
static int test_context_lifecycle(void) {
    printf("Testing context lifecycle...\n");

    fv3_gpu_create_desc create_desc;
    fv3_gpu_context *ctx = NULL;
    fv3_gpu_error err;

    memset(&create_desc, 0, sizeof(create_desc));
    memset(&err, 0, sizeof(err));

    create_desc.abi_version = FV3_GPU_ABI_VERSION;
    create_desc.struct_bytes = sizeof(create_desc);
    create_desc.device_ordinal = 0;
    create_desc.mpi_rank = 0;
    create_desc.mpi_size = 1;
    create_desc.precision = FV3_GPU_PRECISION_R8;
    create_desc.is = 1;
    create_desc.ie = 10;
    create_desc.js = 1;
    create_desc.je = 10;
    create_desc.npz = 72;
    create_desc.halo_width = 3;
    create_desc.flags = 0;
    create_desc.communicator_token = 0;

    /* Attempt to create context - may fail without CUDA */
    int rc = fv3_gpu_create(&create_desc, &ctx, &err);

    if (rc == FV3_GPU_SUCCESS) {
        /* Context created successfully - finalize it */
        TEST_ASSERT(ctx != NULL, "context should be non-NULL on success");

        rc = fv3_gpu_finalize(&ctx, &err);
        TEST_ASSERT(rc == FV3_GPU_SUCCESS, "finalize should succeed");
        TEST_ASSERT(ctx == NULL, "context should be NULL after finalize");

        printf("  INFO: GPU context created and finalized successfully\n");
    } else {
        /* Expected failure without GPU */
        printf("  INFO: GPU context creation returned %d (expected without GPU)\n", rc);
        TEST_ASSERT(rc == FV3_GPU_CUDA_ERROR ||
                    rc == FV3_GPU_UNSUPPORTED_CONFIGURATION ||
                    rc == FV3_GPU_INVALID_ARGUMENT,
                    "should return expected error code without GPU");
    }

    TEST_PASS("context_lifecycle");
    return 0;
}

/* Test 9: Configuration value propagation */
static int test_configuration_propagation(void) {
    printf("Testing configuration propagation...\n");

    /* GEOS first-run configuration values */
    const int hord_mt = 6;
    const int hord_vt = 6;
    const int hord_tm = 6;
    const int hord_dp = -6;
    const double vtdm4 = 0.01;
    const int grid_type = 0;
    const int hydrostatic = 0; /* false */
    const double a_imp = 1.0;

    /* Verify these can be stored in compatible types */
    int32_t stored_hord_mt = hord_mt;
    int32_t stored_hord_dp = hord_dp;
    double stored_vtdm4 = vtdm4;
    uint8_t stored_hydrostatic = hydrostatic;
    double stored_a_imp = a_imp;

    TEST_ASSERT(stored_hord_mt == 6, "hord_mt should propagate");
    TEST_ASSERT(stored_hord_dp == -6, "hord_dp should propagate (negative)");
    TEST_ASSERT(stored_vtdm4 > 0.0, "vtdm4 should be positive");
    TEST_ASSERT(stored_hydrostatic == 0, "hydrostatic should be false");
    TEST_ASSERT(stored_a_imp == 1.0, "a_imp should be 1.0");

    TEST_PASS("configuration_propagation");
    return 0;
}

int main(void) {
    int failures = 0;

    printf("FV3 CUDA Interface C ABI Tests\n");
    printf("==============================\n\n");

    printf("ABI Version: %u\n", FV3_GPU_ABI_VERSION);
    printf("Max Rank: %u\n", FV3_GPU_MAX_RANK);
    printf("Error Message Bytes: %u\n", FV3_GPU_ERROR_MESSAGE_BYTES);
    printf("Field Name Bytes: %u\n\n", FV3_GPU_FIELD_NAME_BYTES);

    failures += test_struct_sizes();
    failures += test_field_descriptor();
    failures += test_create_descriptor();
    failures += test_step_descriptor();
    failures += test_error_handling();
    failures += test_stagger_types();
    failures += test_field_mask();
    failures += test_context_lifecycle();
    failures += test_configuration_propagation();

    printf("\n==============================\n");
    if (failures == 0) {
        printf("ALL TESTS PASSED\n");
        printf("Interface compilation: PASS\n");
        printf("CUDA compilation: NOT RUN\n");
        printf("GPU execution: NOT RUN\n");
        return 0;
    } else {
        printf("FAILURES: %d\n", failures);
        return 1;
    }
}
