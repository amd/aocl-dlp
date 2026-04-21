# AOCL-DLP Benchmark Suite

## Build Command
```bash
cmake -B <build_directory> -DBUILD_BENCHMARKS=ON
cmake --build <build_directory> -j<n>
```

## Run Command

### Run with default gemm_bench_basic_config.yaml
```bash
./<build_directory>/bench/bench_gemm --benchmark_counters_tabular=true --benchmark_filter=<text_to_filter_without_quotes>
```

### Run with gemm_bench_exhaustive_config.yaml
```bash
./<build_directory>/bench/bench_gemm -f <project_root>/bench/configs/gemm_bench_exhaustive_config.yaml --benchmark_counters_tabular=true --benchmark_filter=<text_to_filter_without_quotes>
```

### Help
```bash
./<build_directory>/bench/bench_gemm --help
```

## Iteration Control with `-n` Flag

By default, each benchmark runs for a minimum of **3.0 seconds** (`MinTime(3.0)`). The framework automatically determines the number of iterations needed to fill this time. This can result in a very large number of iterations for small/fast operations.

The `-n` flag allows you to specify an exact number of **measured** iterations, overriding the default time-based approach.

> **Note:** The `-n` flag controls only the measured iterations reported by Google Benchmark. Before the measured loop, the benchmark runs **5 warmup iterations** to stabilize CPU frequency and populate caches. These warmup iterations are not included in the `-n` count. For example, `-n 100` will execute 5 warmup iterations + 100 measured iterations = 105 total kernel executions.

### Default Behavior (MinTime)
When `-n` is not specified, Google Benchmark uses an adaptive algorithm:
- For **small/fast inputs** (e.g., 64x64 matrix): May run thousands of iterations to fill 3.0 seconds
- For **large/slow inputs** (e.g., 4096x4096 matrix): May run only a few iterations to fill 3.0 seconds

### Using the `-n` Flag

#### Run exactly 1000 iterations per benchmark:
```bash
./<build_directory>/bench/bench_gemm -f input.yaml -n 1000
```

#### Run exactly 100 iterations with tabular output:
```bash
./<build_directory>/bench/bench_gemm -f input.yaml -n 100 --benchmark_counters_tabular=true
```

#### Run batch GEMM with fixed iterations:
```bash
./<build_directory>/bench/bench_batch_gemm -f batch_config.yaml -n 500
```

### When to Use `-n` Flag

| Scenario | Recommendation |
|----------|----------------|
| Quick smoke test | Use `-n 10` or `-n 100` for fast results |
| Consistent comparison across input sizes | Use fixed `-n` value for all inputs |
| Accurate performance measurement | Use default MinTime (no `-n` flag) |
| Debugging/verification | Use `-n 1` to run single iteration |

### Output Information
When the benchmark runs, it will display which mode is being used:
```
Benchmark mode: Fixed iterations (1000)
```
or
```
Benchmark mode: MinTime (3.0 seconds)
```
