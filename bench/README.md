#Build command
cmake -B <build_directory> -DBENCHMARKS=ON
cmake --build <build_directory> -j<n>

#Run command

##run with default gemm_bench_basic_config.yaml
./<build_directory>/bench/bench_gemm --benchmark_counters_tabular=true

##run with gemm_bench_exhaustive_config.yaml
./<build_directory>/bench/bench_gemm -f <project_root>/bench/configs/gemm_bench_exhaustive_config.yaml --benchmark_counters_tabular={true}

##help
./<build_directory>/bench/bench_gemm --help
