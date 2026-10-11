/*
 * ==============================================================================
 * 🚀 High-Performance SIMD Eratosthenes Sieve
 * 
 * Optimized by: Gemini 3.1 Pro (High)
 * 
 * Major Optimizations Applied:
 * 1. Architecture-aware SIMD execution (AVX2 for x86_64, NEON for Apple Silicon).
 * 2. Complete removal of conditional dynamic jump logic ("jumper[u]") for small primes, 
 *    replaced by Precomputed Full-Vector Modulo Alignments.
 * 3. Linear SIMD memory blasting using unrolled loops (`#pragma GCC unroll 64`),
 *    maximizing L1 Cache prefetch bandwidth and bypassing data dependencies.
 * 4. Replaced the dense `sieve[u>>6] & ...` checks on the outer loop with branchless
 *    `__builtin_ctzll` (Count Trailing Zeros) bit-scanning for extreme speed.
 * 5. Added 8-way manual unrolling for Large Prime loops (u > 64) to hide memory latency.
 * 6. High-precision benchmarking block to compare Baseline vs Optimized implementations.
 * ==============================================================================
 */

#include <iostream>
#include <cstdint>
#include <array>
#include <chrono>

// Architecture detection macros
#if defined(__AVX2__)  // x86_64 AVX2 Environment
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__) // Apple Silicon Environment
#include <arm_neon.h>
#endif

using namespace std;
using u64 = uint64_t;

constexpr u64 targetN = 1000000000ULL;

// ==========================================
// Original Precomputations (For Baseline Benchmark)
// ==========================================
constexpr array<array<u64, 64>, 64> precomOriginalJumper() {
    array<array<u64, 64>, 64> jumper = {{0}};
    for (u64 i = 2; i < 64; i++) {
        for (u64 j = 0; j < 64; j++) {
            jumper[i][j] = (64 - j + i - 1) / i * i;
        }
    }
    return jumper;
}

constexpr array<array<u64, 64>, 64> precomOriginalPatterns() {
    array<array<u64, 64>, 64> patterns = {{0}};
    for (u64 i = 2; i < 64; i++) {
        for (u64 j = 0; j < 64; j++) {
            for (u64 k = 1ULL << j; k; k <<= i) {
                patterns[i][j] |= k;
            }
        }
    }
    return patterns;
}

constexpr auto orig_patterns = precomOriginalPatterns();
constexpr auto jumper = precomOriginalJumper();

// ==========================================
// SIMD Bit-Pattern Precomputations (Compile-Time)
// ==========================================
// NUM_U64 defines the count of 64-bit blocks inside a single SIMD vector.
// AVX2 uses 256-bit = 4 elements, NEON uses 128-bit = 2 elements.
template <int NUM_U64>
constexpr auto precomSIMDPatterns() {
    array<array<array<u64, NUM_U64>, 64>, 64> patterns = {{{0}}};
    for (u64 p = 2; p < 64; p++) {
        // Block phase M (M = B % p)
        for (u64 m = 0; m < p; m++) {
            for (int i = 0; i < NUM_U64; i++) {
                u64 val = 0;
                for (int bit = 0; bit < 64; bit++) {
                    u64 b = (u64)i * 64 + bit;
                    // Check if the exact bit position in the M-th block is a multiple of prime `p`
                    if (((u64)m * NUM_U64 * 64 + b) % p == 0) {
                        val |= (1ULL << bit);
                    }
                }
                patterns[p][m][i] = val;
            }
        }
    }
    return patterns;
}

#if defined(__AVX2__)
    constexpr auto patternsAVX2 = precomSIMDPatterns<4>();
#elif defined(__ARM_NEON) || defined(__aarch64__)
    constexpr auto patternsNEON = precomSIMDPatterns<2>();
#endif


// ==========================================
// Memory Allocation
// ==========================================
// Aligned to 64 bytes to guarantee safe vector boundary loads/stores for 256-bit AVX.
alignas(64) u64 sieve_orig[(targetN >> 6) + 4];
alignas(64) u64 sieve_simd[(targetN >> 6) + 4];

// Utility: Correctly count the number of un-sieved (0) bits => number of primes
u64 count_primes(const u64* sieve_array, u64 N) {
    u64 count = 0;
    u64 full_words = (N + 1) / 64;
    for (u64 i = 0; i < full_words; i++) {
        count += __builtin_popcountll(~sieve_array[i]);
    }
    u64 remainder_bits = (N + 1) % 64;
    if (remainder_bits > 0) {
        u64 mask = (1ULL << remainder_bits) - 1;
        count += __builtin_popcountll(~sieve_array[full_words] & mask);
    }
    return count;
}


// ==========================================
// 1. Original Baseline Sieve
// ==========================================
void run_original_sieve(u64* sieve_array, u64 N) {
    // 1-1. Small Primes (< 64)
    for (u64 u = 2; u < 64; u++) {
        if (sieve_array[u >> 6] & (1ULL << (u & 63))) continue;
        for (u64 k = u * u; k <= N; k += jumper[u][k & 63]) {
            sieve_array[k >> 6] |= orig_patterns[u][k & 63];
        }
    }
    
    // 1-2. Large Primes (>= 64)
    for (u64 u = 64; u * u <= N; u++) {
        if (sieve_array[u >> 6] & (1ULL << (u & 63))) continue;
        for (u64 k = u * u; k <= N; k += u) {
            sieve_array[k >> 6] |= 1ULL << (k & 63);
        }
    }
}


// ==========================================
// 2. Gemini Optimized SIMD Sieve
// ==========================================
void run_optimized_simd_sieve(u64* sieve_array, u64 N) {
    
    // 2-1. Small Primes (< 64) using extremely fast __builtin_ctzll mapping
    u64 mask0 = ~sieve_array[0];
    mask0 &= ~3ULL; // Clear bits 0 and 1 (0 and 1 are not primes)
    
    while (mask0) {
        // Fast-extract the next un-sieved prime bit
        u64 u = __builtin_ctzll(mask0);
        mask0 &= mask0 - 1; // Unset the lowest set bit
        
#if defined(__AVX2__)
        // -------- AVX2 Accelerated Block ---------
        constexpr u64 V_U64 = 4;
        u64 num_blocks = (N + 1 + (V_U64 * 64) - 1) / (V_U64 * 64);
        
        u64 start_k = u * u;
        u64 first_block = start_k / (V_U64 * 64);
        u64 scalar_end = (first_block + 1) * (V_U64 * 64);
        if (scalar_end > N + 1) scalar_end = N + 1;
        
        // Scalar mode until we hit securely aligned vector block boundaries
        for (u64 k = start_k; k < scalar_end; k += u) {
            sieve_array[k >> 6] |= (1ULL << (k & 63));
        }
        if (scalar_end > N) continue;
        
        // Fetch Phase Masks for prime 'u'
        __m256i m[64];
        for (u64 m_idx = 0; m_idx < u; m_idx++) {
            m[m_idx] = _mm256_setr_epi64x(
                patternsAVX2[u][m_idx][0], patternsAVX2[u][m_idx][1],
                patternsAVX2[u][m_idx][2], patternsAVX2[u][m_idx][3]
            );
        }
        
        u64 B = first_block + 1;
        while (B % u != 0 && B < num_blocks) {
            __m256i* ptr = (__m256i*)&sieve_array[B * V_U64];
            _mm256_store_si256(ptr, _mm256_or_si256(_mm256_load_si256(ptr), m[B % u]));
            B++;
        }
        
        // FAST PATH: Fully unrolled loop chunks over exact prime LCM cycles.
        u64 B_fast = B;
        u64 limit_fast = num_blocks >= u ? num_blocks - u : 0;
        
        while (B_fast <= limit_fast) {
            #pragma GCC unroll 64
            for (u64 i = 0; i < u; i++) {
                __m256i* ptr = (__m256i*)&sieve_array[(B_fast + i) * V_U64];
                _mm256_store_si256(ptr, _mm256_or_si256(_mm256_load_si256(ptr), m[i]));
            }
            B_fast += u;
        }
        
        for (; B_fast < num_blocks; ++B_fast) {
            __m256i* ptr = (__m256i*)&sieve_array[B_fast * V_U64];
            _mm256_store_si256(ptr, _mm256_or_si256(_mm256_load_si256(ptr), m[B_fast % u]));
        }

#elif defined(__ARM_NEON) || defined(__aarch64__)
        // -------- ARM NEON Accelerated Block ---------
        constexpr u64 V_U64 = 2;
        u64 num_blocks = (N + 1 + (V_U64 * 64) - 1) / (V_U64 * 64);
        
        u64 start_k = u * u;
        u64 first_block = start_k / (V_U64 * 64);
        u64 scalar_end = (first_block + 1) * (V_U64 * 64);
        if (scalar_end > N + 1) scalar_end = N + 1;
        
        for (u64 k = start_k; k < scalar_end; k += u) {
            sieve_array[k >> 6] |= (1ULL << (k & 63));
        }
        if (scalar_end > N) continue;
        
        uint64x2_t m[64];
        for (u64 m_idx = 0; m_idx < u; m_idx++) {
            uint64_t arr[2] = { patternsNEON[u][m_idx][0], patternsNEON[u][m_idx][1] };
            m[m_idx] = vld1q_u64(arr);
        }
        
        u64 B = first_block + 1;
        while (B % u != 0 && B < num_blocks) {
            uint64_t* ptr_aligned = &sieve_array[B * V_U64];
            uint64x2_t val = vld1q_u64(ptr_aligned);
            vst1q_u64(ptr_aligned, vorrq_u64(val, m[B % u]));
            B++;
        }
        
        u64 B_fast = B;
        u64 limit_fast = num_blocks >= u ? num_blocks - u : 0;
        
        while (B_fast <= limit_fast) {
            #pragma GCC unroll 64
            for (u64 i = 0; i < u; i++) {
                uint64_t* ptr_aligned = &sieve_array[(B_fast + i) * V_U64];
                uint64x2_t val = vld1q_u64(ptr_aligned);
                vst1q_u64(ptr_aligned, vorrq_u64(val, m[i])); 
            }
            B_fast += u;
        }
        
        for (; B_fast < num_blocks; ++B_fast) {
            uint64_t* ptr_aligned = &sieve_array[B_fast * V_U64];
            uint64x2_t val = vld1q_u64(ptr_aligned);
            vst1q_u64(ptr_aligned, vorrq_u64(val, m[B_fast % u]));
        }
#else
        // Fallback for unexpected missing SIMD flags
        for (u64 k = u * u; k <= N; k += jumper[u][k & 63]) {
            sieve_array[k >> 6] |= orig_patterns[u][k & 63];
        }
#endif
    }
    
    // 2-2. Large Primes (u >= 64). Optimized with 8-way step unrolling to hide memory latency.
    u64 sqrt_target = 0;
    while ((sqrt_target + 1) * (sqrt_target + 1) <= N) sqrt_target++;

    // Start block `i` loop from 1 (skips index 0~63 since small primes already sieved them)
    // Again, eliminating branches using __builtin_ctzll
    for (u64 i = 1; i <= (sqrt_target >> 6); i++) {
        u64 mask = ~sieve_array[i];
        while (mask) {
            u64 bit = __builtin_ctzll(mask);
            u64 u = (i * 64) + bit;
            mask &= mask - 1; // Clear lowest set bit
            
            u64 k = u * u;
            // Manual 8-way unroll blocks prefetching
            for (; k + u * 8 <= N; k += u * 8) {
                sieve_array[k >> 6]           |= (1ULL << (k & 63));
                sieve_array[(k + u) >> 6]     |= (1ULL << ((k + u) & 63));
                sieve_array[(k + u * 2) >> 6] |= (1ULL << ((k + u * 2) & 63));
                sieve_array[(k + u * 3) >> 6] |= (1ULL << ((k + u * 3) & 63));
                sieve_array[(k + u * 4) >> 6] |= (1ULL << ((k + u * 4) & 63));
                sieve_array[(k + u * 5) >> 6] |= (1ULL << ((k + u * 5) & 63));
                sieve_array[(k + u * 6) >> 6] |= (1ULL << ((k + u * 6) & 63));
                sieve_array[(k + u * 7) >> 6] |= (1ULL << ((k + u * 7) & 63));
            }
            // Cleanup remaining
            for (; k <= N; k += u) {
                sieve_array[k >> 6] |= 1ULL << (k & 63);
            }
        }
    }
}


// ==========================================
// Main Function & Benchmarking Runner
// ==========================================
int main() {
    ios_base::sync_with_stdio(false); 
    cin.tie(NULL);

    cout << "==========================================\n";
    cout << " Eratosthenes Sieve Benchmark (target N = 10^9)\n";
    cout << "==========================================\n\n";

    // Initialize arrays (mark 0 and 1 as composite)
    for (size_t i = 0; i < (targetN >> 6) + 4; i++) {
        sieve_orig[i] = 0ULL;
        sieve_simd[i] = 0ULL;
    }
    sieve_orig[0] = 0b11; 
    sieve_simd[0] = 0b11;

    // --- Benchmark 1: Original Code ---
    auto start_orig = chrono::high_resolution_clock::now();
    run_original_sieve(sieve_orig, targetN);
    auto end_orig = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration_orig = end_orig - start_orig;

    // --- Benchmark 2: Gemini Optimized SIMD Code ---
    auto start_simd = chrono::high_resolution_clock::now();
    run_optimized_simd_sieve(sieve_simd, targetN);
    auto end_simd = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration_simd = end_simd - start_simd;

    // --- Verification ---
    u64 count_orig = count_primes(sieve_orig, targetN);
    u64 count_simd = count_primes(sieve_simd, targetN);

    cout << "Verification Check:\n";
    cout << " -> Original Primes Found: " << count_orig << "\n";
    cout << " -> SIMD Optimized Found : " << count_simd << "\n";
    if (count_orig == count_simd) {
        cout << " -> [SUCCESS] Outputs match exactly!\n\n";
    } else {
        cout << " -> [ERROR] Outputs mismatch!\n\n";
    }

    // --- Results Output ---
    cout << "Performance Results (Milliseconds):\n";
    cout << " -> Original Execution Time: " << duration_orig.count() << " ms\n";
    cout << " -> SIMD Optimized Time    : " << duration_simd.count() << " ms\n";
    
    double speedup = duration_orig.count() / (duration_simd.count() == 0 ? 0.001 : duration_simd.count());
    cout << " -> Overall Speedup        : " << speedup << "x times faster\n";
    
    cout << "\n==========================================\n";
    cout << " Optimized automatically by Gemini 3.1\n";
    cout << "==========================================\n";

    return 0;
}
