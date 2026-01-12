1. w/o features:
compile_template{rotation=true max_levels=3}:precompute_rotations{count=12}: close time.busy=10.7ms time.idle=4.08µs
compile_template{rotation=true max_levels=3}: close time.busy=11.3ms time.idle=6.71µs
coarse_to_fine{levels=3 parallel=false}:coarse_search{level=2 angles=12}: close time.busy=128ms time.idle=3.75µs
coarse_to_fine{levels=3 parallel=false}:refine_level{level=1 candidates=7}: close time.busy=88.1ms time.idle=4.33µs
coarse_to_fine{levels=3 parallel=false}:refine_level{level=0 candidates=8}: close time.busy=369ms time.idle=5.33µs
coarse_to_fine{levels=3 parallel=false}: close time.busy=585ms time.idle=2.29µs
final_refinement: close time.busy=536µs time.idle=3.12µs

2. rayon
compile_template{rotation=true max_levels=3}:precompute_rotations{count=12}: close time.busy=3.03ms time.idle=5.92µs
compile_template{rotation=true max_levels=3}: close time.busy=3.78ms time.idle=7.04µs
coarse_to_fine{levels=3 parallel=true}:coarse_search{level=2 angles=12 parallel=true}: close time.busy=21.9ms time.idle=4.08µs
coarse_to_fine{levels=3 parallel=true}:refine_level{level=1 candidates=7 parallel=true}: close time.busy=17.8ms time.idle=4.38µs
coarse_to_fine{levels=3 parallel=true}:refine_level{level=0 candidates=8 parallel=true}: close time.busy=53.8ms time.idle=4.46µs
coarse_to_fine{levels=3 parallel=true}: close time.busy=93.6ms time.idle=2.92µs
final_refinement: close time.busy=527µs time.idle=2.54µs

3. no rotation, no features:
compile_template{rotation=false max_levels=3}: close time.busy=426µs time.idle=9.12µs
coarse_to_fine{levels=3 parallel=false}:coarse_search{level=2 angles=1}: close time.busy=13.9ms time.idle=5.92µs
coarse_to_fine{levels=3 parallel=false}:refine_level{level=1 candidates=1}: close time.busy=4.92ms time.idle=3.12µs
coarse_to_fine{levels=3 parallel=false}:refine_level{level=0 candidates=2}: close time.busy=31.6ms time.idle=4.17µs
coarse_to_fine{levels=3 parallel=false}: close time.busy=57.0ms time.idle=3.00µs
final_refinement: close time.busy=491µs time.idle=2.63µs

3. no rotation, simd feature:
compile_template{rotation=false max_levels=3}: close time.busy=413µs time.idle=7.04µs
coarse_to_fine{levels=3 parallel=false}:coarse_search{level=2 angles=1}: close time.busy=19.7ms time.idle=4.25µs
coarse_to_fine{levels=3 parallel=false}:refine_level{level=1 candidates=1}: close time.busy=7.81ms time.idle=3.88µs
coarse_to_fine{levels=3 parallel=false}:refine_level{level=0 candidates=2}: close time.busy=51.5ms time.idle=4.42µs
coarse_to_fine{levels=3 parallel=false}: close time.busy=79.1ms time.idle=2.92µs
final_refinement: close time.busy=736µs time.idle=2.17µs

4. no rotation, rayon:
compile_template{rotation=false max_levels=3}: close time.busy=506µs time.idle=7.67µs
coarse_to_fine{levels=3 parallel=false}:coarse_search{level=2 angles=1}: close time.busy=15.7ms time.idle=5.25µs
coarse_to_fine{levels=3 parallel=false}:refine_level{level=1 candidates=1}: close time.busy=5.56ms time.idle=3.50µs
coarse_to_fine{levels=3 parallel=false}:refine_level{level=0 candidates=2}: close time.busy=35.5ms time.idle=3.33µs
coarse_to_fine{levels=3 parallel=false}: close time.busy=56.9ms time.idle=2.83µs
final_refinement: close time.busy=475µs time.idle=2.04µs

5. no rotation, rayon and simd:
compile_template{rotation=false max_levels=3}: close time.busy=453µs time.idle=7.92µs
coarse_to_fine{levels=3 parallel=false}:coarse_search{level=2 angles=1}: close time.busy=19.5ms time.idle=5.12µs
coarse_to_fine{levels=3 parallel=false}:refine_level{level=1 candidates=1}: close time.busy=8.04ms time.idle=4.08µs
coarse_to_fine{levels=3 parallel=false}:refine_level{level=0 candidates=2}: close time.busy=52.0ms time.idle=3.88µs
coarse_to_fine{levels=3 parallel=false}: close time.busy=79.7ms time.idle=3.04µs
final_refinement: close time.busy=741µs time.idle=2.29µs