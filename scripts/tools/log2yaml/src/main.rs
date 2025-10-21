/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

use anyhow::{Context, Result};
use clap::Parser;
use regex::Regex;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "log2yaml")]
#[command(about = "Parse test logs and generate YAML configs for failed tests", long_about = None)]
struct Args {
    /// Input log file path
    #[arg(value_name = "LOG_FILE")]
    log_file: PathBuf,

    /// Output YAML file path
    #[arg(
        short,
        long,
        value_name = "OUTPUT",
        default_value = "failed_tests_config.yaml"
    )]
    output: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Dry run - show what would be generated without writing file
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct FailedTest {
    test_name: String,
    m: u32,
    n: u32,
    k: u32,
    a_type: String,
    b_type: String,
    c_type: String,
    acc_type: String,
    storage_format: String,
    trans_a: bool,
    trans_b: bool,
    lda: u32,
    ldb: u32,
    ldc: u32,
    alpha: f64,
    beta: f64,
    reorder_a: bool,
    reorder_b: bool,
    pack_a: bool,
    pack_b: bool,
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
enum DimensionSpec {
    Value(u32),
    List(Vec<u32>),
    Range {
        lb: u32,
        ub: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        step: Option<i32>,
    },
}

#[derive(Debug, Serialize)]
struct YamlTestConfig {
    name: String,
    a_type: Vec<String>,
    b_type: Vec<String>,
    c_type: Vec<String>,
    acc_type: Vec<String>,
    storage_format: Vec<String>,
    #[serde(rename = "transA")]
    trans_a: Vec<bool>,
    #[serde(rename = "transB")]
    trans_b: Vec<bool>,
    m: DimensionSpec,
    n: DimensionSpec,
    k: DimensionSpec,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lda: Option<DimensionSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ldb: Option<DimensionSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ldc: Option<DimensionSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "mtagA")]
    mtag_a: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "mtagB")]
    mtag_b: Option<Vec<String>>,
    tolerances: Tolerances,
}

#[derive(Debug, Serialize)]
struct Tolerances {
    float: f64,
    bfloat16: f64,
    int8: i32,
}

#[derive(Debug, Serialize)]
struct YamlConfig {
    gemm_tests: Vec<YamlTestConfig>,
}

fn parse_log_file(content: &str, verbose: bool) -> Result<Vec<FailedTest>> {
    let mut failed_tests = Vec::new();

    // Regex to match the FAILED test line and extract test name
    let failed_re = Regex::new(
        r"\[  FAILED  \] YamlConfigurations/GemmParameterizedTest\.CompareImplementations/(\S+)",
    )?;

    // Regex patterns for parsing test details
    let dim_re = Regex::new(r"Matrix Dimensions: M=(\d+), N=(\d+), K=(\d+)")?;
    let types_re = Regex::new(r"Data Types: A=(\w+), B=(\w+), C=(\w+), ACC=(\w+)")?;
    let storage_re = Regex::new(r"Storage Format: ([\w-]+)")?;
    let trans_re = Regex::new(r"Transposition: transA=(\w+), transB=(\w+)")?;
    let ld_re = Regex::new(r"Leading Dimensions: lda=(\d+), ldb=(\d+), ldc=(\d+)")?;
    let alpha_beta_re = Regex::new(r"Alpha/Beta: alpha=([\d.+-]+), beta=([\d.+-]+)")?;
    let reorder_re = Regex::new(r"Reordering: reorderA=(\w+), reorderB=(\w+)")?;
    let pack_re = Regex::new(r"Packing: packA=(\w+), packB=(\w+)")?;

    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        if let Some(caps) = failed_re.captures(lines[i]) {
            let test_name = caps.get(1).unwrap().as_str().to_string();

            if verbose {
                println!("Found failed test: {}", test_name);
            }

            // Look ahead for the test details (they appear after the failure line)
            let mut test = FailedTest {
                test_name: test_name.clone(),
                m: 0,
                n: 0,
                k: 0,
                a_type: String::new(),
                b_type: String::new(),
                c_type: String::new(),
                acc_type: String::new(),
                storage_format: String::new(),
                trans_a: false,
                trans_b: false,
                lda: 0,
                ldb: 0,
                ldc: 0,
                alpha: 0.0,
                beta: 0.0,
                reorder_a: false,
                reorder_b: false,
                pack_a: false,
                pack_b: false,
            };

            // Parse the details from the preceding lines (Google Test prints details before FAILED)
            for j in (i.saturating_sub(20))..i {
                if j >= lines.len() {
                    continue;
                }
                let line = lines[j];

                if let Some(caps) = dim_re.captures(line) {
                    test.m = caps[1].parse()?;
                    test.n = caps[2].parse()?;
                    test.k = caps[3].parse()?;
                }
                if let Some(caps) = types_re.captures(line) {
                    test.a_type = caps[1].to_string();
                    test.b_type = caps[2].to_string();
                    test.c_type = caps[3].to_string();
                    test.acc_type = caps[4].to_string();
                }
                if let Some(caps) = storage_re.captures(line) {
                    test.storage_format = caps[1].to_string();
                }
                if let Some(caps) = trans_re.captures(line) {
                    test.trans_a = &caps[1] == "true";
                    test.trans_b = &caps[2] == "true";
                }
                if let Some(caps) = ld_re.captures(line) {
                    test.lda = caps[1].parse()?;
                    test.ldb = caps[2].parse()?;
                    test.ldc = caps[3].parse()?;
                }
                if let Some(caps) = alpha_beta_re.captures(line) {
                    test.alpha = caps[1].parse()?;
                    test.beta = caps[2].parse()?;
                }
                if let Some(caps) = reorder_re.captures(line) {
                    test.reorder_a = &caps[1] == "true";
                    test.reorder_b = &caps[2] == "true";
                }
                if let Some(caps) = pack_re.captures(line) {
                    test.pack_a = &caps[1] == "true";
                    test.pack_b = &caps[2] == "true";
                }
            }

            // Validate that we parsed meaningful data
            if test.m > 0 && test.n > 0 && test.k > 0 && !test.a_type.is_empty() {
                failed_tests.push(test);
            } else if verbose {
                println!(
                    "Warning: Could not parse complete details for {}",
                    test_name
                );
            }
        }
        i += 1;
    }

    Ok(failed_tests)
}

fn group_tests_by_pattern(tests: Vec<FailedTest>) -> Vec<Vec<FailedTest>> {
    // Group tests with identical parameters except for M, N, K
    let mut groups: HashMap<String, Vec<FailedTest>> = HashMap::new();

    for test in tests {
        // Create a key from all parameters except M, N, K
        let key = format!(
            "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}",
            test.a_type,
            test.b_type,
            test.c_type,
            test.acc_type,
            test.storage_format,
            test.trans_a,
            test.trans_b,
            test.alpha,
            test.beta,
            test.reorder_a,
            test.reorder_b,
            test.pack_a,
            test.pack_b,
            // Include lda, ldb, ldc in grouping if they're not default
            if test.lda > 0 {
                test.lda.to_string()
            } else {
                "auto".to_string()
            }
        );

        groups.entry(key).or_insert_with(Vec::new).push(test);
    }

    groups.into_values().collect()
}

fn create_dimension_spec(values: &[u32]) -> DimensionSpec {
    if values.is_empty() {
        return DimensionSpec::Value(0);
    }

    if values.len() == 1 {
        return DimensionSpec::Value(values[0]);
    }

    // Check if values form a simple range
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    if sorted.len() <= 3 {
        return DimensionSpec::List(sorted);
    }

    // Check for arithmetic progression
    let diffs: Vec<i32> = sorted
        .windows(2)
        .map(|w| w[1] as i32 - w[0] as i32)
        .collect();

    let all_same_diff = diffs.windows(2).all(|w| w[0] == w[1]);

    if all_same_diff && !diffs.is_empty() {
        let step = diffs[0];
        if step == 1 {
            DimensionSpec::Range {
                lb: sorted[0],
                ub: sorted[sorted.len() - 1],
                step: Some(-1), // -1 indicates step of 1 in the config format
            }
        } else {
            DimensionSpec::Range {
                lb: sorted[0],
                ub: sorted[sorted.len() - 1],
                step: Some(step),
            }
        }
    } else {
        DimensionSpec::List(sorted)
    }
}

fn generate_yaml_config(groups: Vec<Vec<FailedTest>>, verbose: bool) -> Result<YamlConfig> {
    let mut test_configs = Vec::new();

    for (idx, group) in groups.iter().enumerate() {
        if group.is_empty() {
            continue;
        }

        // Use first test as template
        let template = &group[0];

        // Collect unique values for M, N, K
        let mut m_values: Vec<u32> = group.iter().map(|t| t.m).collect();
        let mut n_values: Vec<u32> = group.iter().map(|t| t.n).collect();
        let mut k_values: Vec<u32> = group.iter().map(|t| t.k).collect();

        m_values.sort_unstable();
        m_values.dedup();
        n_values.sort_unstable();
        n_values.dedup();
        k_values.sort_unstable();
        k_values.dedup();

        if verbose {
            println!("Group {}: {} tests", idx, group.len());
            println!("  M values: {:?}", m_values);
            println!("  N values: {:?}", n_values);
            println!("  K values: {:?}", k_values);
        }

        // Collect unique alpha/beta values
        let alpha_values: HashSet<_> = group
            .iter()
            .map(|t| {
                // Round to avoid floating point issues
                (t.alpha * 10000.0).round() as i64
            })
            .collect();
        let alpha_vec: Vec<f64> = alpha_values.iter().map(|&v| v as f64 / 10000.0).collect();

        let beta_values: HashSet<_> = group
            .iter()
            .map(|t| (t.beta * 10000.0).round() as i64)
            .collect();
        let beta_vec: Vec<f64> = beta_values.iter().map(|&v| v as f64 / 10000.0).collect();

        // Collect unique trans values
        let trans_a_values: HashSet<_> = group.iter().map(|t| t.trans_a).collect();
        let trans_b_values: HashSet<_> = group.iter().map(|t| t.trans_b).collect();

        // Create descriptive name
        let type_str = format!("{}{}{}", template.a_type, template.b_type, template.c_type);
        let name = format!("failed_{}_group_{}", type_str, idx);

        // Check if lda, ldb, ldc vary or are constant
        let lda_values: HashSet<_> = group.iter().map(|t| t.lda).collect();
        let ldb_values: HashSet<_> = group.iter().map(|t| t.ldb).collect();
        let ldc_values: HashSet<_> = group.iter().map(|t| t.ldc).collect();

        let lda = if lda_values.len() > 1 {
            let mut vals: Vec<u32> = lda_values.into_iter().collect();
            vals.sort_unstable();
            Some(create_dimension_spec(&vals))
        } else {
            Some(DimensionSpec::Value(*lda_values.iter().next().unwrap()))
        };

        let ldb = if ldb_values.len() > 1 {
            let mut vals: Vec<u32> = ldb_values.into_iter().collect();
            vals.sort_unstable();
            Some(create_dimension_spec(&vals))
        } else {
            Some(DimensionSpec::Value(*ldb_values.iter().next().unwrap()))
        };

        let ldc = if ldc_values.len() > 1 {
            let mut vals: Vec<u32> = ldc_values.into_iter().collect();
            vals.sort_unstable();
            Some(create_dimension_spec(&vals))
        } else {
            Some(DimensionSpec::Value(*ldc_values.iter().next().unwrap()))
        };

        // Determine mtag based on reorder/pack flags
        let mtag_a = if template.reorder_a || template.pack_a {
            let mut tags = Vec::new();
            if template.pack_a {
                tags.push("pack".to_string());
            }
            if template.reorder_a {
                tags.push("reorder".to_string());
            }
            if tags.is_empty() {
                tags.push("none".to_string());
            }
            Some(tags)
        } else {
            Some(vec!["none".to_string()])
        };

        let mtag_b = if template.reorder_b || template.pack_b {
            let mut tags = Vec::new();
            if template.pack_b {
                tags.push("pack".to_string());
            }
            if template.reorder_b {
                tags.push("reorder".to_string());
            }
            if tags.is_empty() {
                tags.push("none".to_string());
            }
            Some(tags)
        } else {
            Some(vec!["none".to_string()])
        };

        let config = YamlTestConfig {
            name,
            a_type: vec![template.a_type.clone()],
            b_type: vec![template.b_type.clone()],
            c_type: vec![template.c_type.clone()],
            acc_type: vec![template.acc_type.clone()],
            storage_format: vec![template.storage_format.clone()],
            trans_a: trans_a_values.into_iter().collect(),
            trans_b: trans_b_values.into_iter().collect(),
            m: create_dimension_spec(&m_values),
            n: create_dimension_spec(&n_values),
            k: create_dimension_spec(&k_values),
            alpha: alpha_vec,
            beta: beta_vec,
            lda,
            ldb,
            ldc,
            mtag_a,
            mtag_b,
            tolerances: Tolerances {
                float: 1.0e-5,
                bfloat16: 1.0e-2,
                int8: 0,
            },
        };

        test_configs.push(config);
    }

    Ok(YamlConfig {
        gemm_tests: test_configs,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Read log file
    let content = fs::read_to_string(&args.log_file)
        .with_context(|| format!("Failed to read log file: {:?}", args.log_file))?;

    println!("Parsing log file: {:?}", args.log_file);

    // Parse failed tests
    let failed_tests = parse_log_file(&content, args.verbose)?;

    if failed_tests.is_empty() {
        println!("No failed tests found in the log file.");
        return Ok(());
    }

    println!("Found {} failed tests", failed_tests.len());

    if args.verbose {
        for test in &failed_tests {
            println!(
                "  - {} (M={}, N={}, K={})",
                test.test_name, test.m, test.n, test.k
            );
        }
    }

    // Group tests by similar patterns
    let groups = group_tests_by_pattern(failed_tests);
    println!("Grouped into {} test configuration(s)", groups.len());

    // Generate YAML configuration
    let yaml_config = generate_yaml_config(groups, args.verbose)?;

    // Serialize to YAML
    let yaml_string = serde_yaml::to_string(&yaml_config).context("Failed to serialize YAML")?;

    // Add header comment
    let output = format!(
        "# GEMM test configuration for failed tests\n# Generated from: {:?}\n# Total failed tests: {}\n\n{}",
        args.log_file,
        yaml_config.gemm_tests.iter().map(|_g| {
            // Estimate total tests per config
            1  // Simplified for now
        }).sum::<usize>(),
        yaml_string
    );

    if args.dry_run {
        println!("\n=== DRY RUN - Generated YAML ===");
        println!("{}", output);
    } else {
        // Write to output file
        fs::write(&args.output, output)
            .with_context(|| format!("Failed to write output file: {:?}", args.output))?;

        println!("\n✓ Successfully generated YAML config: {:?}", args.output);
        println!("  You can now run tests with this configuration to reproduce failures.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_dimension_spec_single_value() {
        let result = create_dimension_spec(&[10]);
        assert!(matches!(result, DimensionSpec::Value(10)));
    }

    #[test]
    fn test_create_dimension_spec_range() {
        let result = create_dimension_spec(&[10, 11, 12, 13, 14]);
        match result {
            DimensionSpec::Range { lb, ub, step } => {
                assert_eq!(lb, 10);
                assert_eq!(ub, 14);
                assert_eq!(step, Some(-1));
            }
            _ => panic!("Expected Range"),
        }
    }

    #[test]
    fn test_create_dimension_spec_list() {
        let result = create_dimension_spec(&[10, 20, 30]);
        assert!(matches!(result, DimensionSpec::List(_)));
    }
}
