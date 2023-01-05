use anyhow::{bail, Context, Result};
use clap::Parser;
use itertools::Itertools;
use no_drama::rank_bank;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    ///Yaml with array entry "rank_bank_function" and "row_function" containing one mask for each function bit
    #[clap(long, default_value = "dram-fns.yml")]
    dram_function_config_path: String,

    ///Choose verification mode from ["row-conflict"]
    #[clap(long)]
    mode: String,

    ///Path to csv file  with input data
    #[clap(short, long)]
    input: String,
}

impl CliArgs {
    fn validate(&self) -> Result<()> {
        if self.mode != "row-conflict" {
            bail!("mode argument must be \"row-conflict\"")
        }

        Ok(())
    }
}

#[derive(Deserialize, Debug)]
struct DRAMAddressing {
    rank_bank_function: Vec<u64>,
    row_function: Vec<u64>,
}

impl DRAMAddressing {
    fn bank_rank_addr(&self, paddr: u64) -> Vec<u64> {
        rank_bank::evaluate_addr_function(&self.rank_bank_function, paddr)
    }

    fn row_addr(&self, paddr: u64) -> Vec<u64> {
        rank_bank::evaluate_addr_function(&self.row_function, paddr)
    }
}

fn main() -> Result<()> {
    let args: CliArgs = CliArgs::parse();

    if let Err(e) = args.validate() {
        eprintln!("Invalid Config!");
        bail!(e);
    }

    let dram_fns = File::open(args.dram_function_config_path.clone()).with_context(|| {
        format!(
            "failed to open config file {}",
            args.dram_function_config_path
        )
    })?;
    let dram_fns: DRAMAddressing =
        serde_yaml::from_reader(dram_fns).with_context(|| "failed to parse dram address config")?;

    eprintln!("Dram address functions {:x?}", dram_fns);

    //
    // Program Logic
    //

    #[derive(Debug, Deserialize)]
    struct Entry {
        paddr_a: u64,
        paddr_b: u64,
        timing: u64,
    }

    let mut reader = csv::Reader::from_path(args.input.clone())
        .with_context(|| format!("failed to open input file {}", args.input))?;

    let mut row_conflict_tuples: Vec<Entry> = Vec::new();
    for result in reader.records().skip(1) {
        let entry: Entry = result?.deserialize(None)?;
        row_conflict_tuples.push(entry);
    }

    //check, that our addressing function also states "row conflict" for the found addrs
    eprintln!("Checking...");
    let mut error_counter = 0;
    let mut diff_set_bit_histogram = HashMap::new();
    let mut diff_set_bit_tuples_histogram = HashMap::new();
    let mut bank_fn_bit_mismatch_count = vec![0; dram_fns.rank_bank_function.len()];
    let mut bank_fn_bit_encounter_count = vec![0; dram_fns.rank_bank_function.len()];
    let bit_13_is_one = 0;
    let mut row_fn_bit_encounter_count = vec![0; dram_fns.row_function.len()];

    for entry in row_conflict_tuples.iter() {
        let bank_a = dram_fns.bank_rank_addr(entry.paddr_a);
        let bank_b = dram_fns.bank_rank_addr(entry.paddr_b);

        for (idx, bank_bit_fn) in dram_fns.rank_bank_function.iter().enumerate() {
            //if at least one bit from bank_bit_fn is set in the xor diff
            if ((entry.paddr_a ^ entry.paddr_b) & bank_bit_fn) != 0 {
                bank_fn_bit_encounter_count[idx] += 1;
            }
        }

        for (idx, row_bit_fn) in dram_fns.row_function.iter().enumerate() {
            if ((entry.paddr_a ^ entry.paddr_b) & row_bit_fn) != 0 {
                row_fn_bit_encounter_count[idx] += 1;
            }
        }

        if !bank_a.eq(&bank_b) {
            eprintln!(", {:x} and {:x} (xor diff {:09x})\t have row conflict timing {}, but address function says DIFFERENT SETs.",entry.paddr_a,entry.paddr_b,entry.paddr_a ^ entry.paddr_b,entry.timing);
            error_counter += 1;

            for (idx, (a, b)) in bank_a.iter().zip(bank_b).enumerate() {
                if *a != b {
                    bank_fn_bit_mismatch_count[idx] += 1
                }
            }

            //simple bits
            for bit_idx in 12..64 {
                let mask = 1_u64 << bit_idx;
                if (entry.paddr_a ^ entry.paddr_b) & mask != 0 {
                    if diff_set_bit_histogram.contains_key(&bit_idx) {
                        match diff_set_bit_histogram.get_mut(&bit_idx) {
                            None => bail! {"unexpected hashmap state"},
                            Some(v) => (*v) += 1,
                        }
                    } else {
                        diff_set_bit_histogram.insert(bit_idx, 0);
                    }
                }
            }

            // 2 tuples
            for combo_elems in (12..64).combinations(2) {
                let mut mask = 0;
                for e in &combo_elems {
                    mask |= 1_u64 << e
                }

                if (entry.paddr_a ^ entry.paddr_b) & mask != 0 {
                    if diff_set_bit_tuples_histogram.contains_key(&combo_elems) {
                        match diff_set_bit_tuples_histogram.get_mut(&combo_elems) {
                            None => bail! {"unexpected hashmap state"},
                            Some(v) => (*v) += 1,
                        }
                    } else {
                        diff_set_bit_tuples_histogram.insert(combo_elems, 0);
                    }
                }
            }
        } else {
            // a and b are in the same bank
            if dram_fns
                .row_addr(entry.paddr_a)
                .eq(&dram_fns.row_addr(entry.paddr_b))
            {
                eprintln!(
                    "{:x} and {:x} (xor diff {:09x})\t have row conflict timing {} but address function says SAME ROW ",
                    entry.paddr_a,entry.paddr_b,entry.paddr_a ^ entry.paddr_b,entry.timing
                );
                error_counter += 1;
            }
        }
    }
    eprintln!("done!");
    eprintln!(
        "Got {} percent errors",
        ((error_counter as f64) / (row_conflict_tuples.len() as f64)) * 100.0
    );
    for (bit_idx, &count) in diff_set_bit_histogram
        .iter()
        .sorted_by(|(_, v1), (_, v2)| Ord::cmp(v2, v1))
    {
        eprintln!(
            "bit_idx {} was in {:.4} percent of the  diffs",
            bit_idx,
            (count as f64) / (error_counter as f64) * 100.0
        )
    }

    for (bit_idx, &count) in diff_set_bit_tuples_histogram
        .iter()
        .sorted_by(|(_, val1), (_, val2)| Ord::cmp(val2, val1))
        .filter(|(k, _)| *k.iter().max().unwrap() < 29)
    //.filter(|(_, &v)| (v as f64) / (error_counter as f64) > 0.55)
    {
        eprintln!(
            "bit_idx {:?} was in {:.4} percent of the  diffs",
            bit_idx,
            (count as f64) / (error_counter as f64) * 100.0
        )
    }

    eprintln!(
        "bank addr values missmatches {:?}",
        bank_fn_bit_mismatch_count
            .iter()
            .copied()
            .map(|v| (v as f64) / (error_counter as f64) * 100.0)
            .collect::<Vec<f64>>()
    );
    eprintln!("bank_fn encounter count {:?}", bank_fn_bit_encounter_count);
    eprintln!("row_fn encounter count {:?}", row_fn_bit_encounter_count);

    eprintln!("Bit 13 is one: {}", bit_13_is_one);

    Ok(())
}
