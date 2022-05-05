use anyhow::{bail, Context, Result};
use clap::Parser;
use nix::sys::mman::{MapFlags, ProtFlags};
use no_drama::memory::LinuxPageMap;
use no_drama::memory::MemorySource;
use no_drama::{memory, rank_bank, DefaultMemoryTupleTimer, MemoryTupleTimer};
use serde::Deserialize;
use std::fs::File;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    #[clap(long, default_value = "dram-fns.yml")]
    dram_function_config_path: String,

    ///Alignment in bytes for sampled addresses
    #[clap(long, default_value = "64")]
    alignment: usize,

    ///Percent of allocated memory to cover during verification (order is randomized)
    #[clap(long, default_value = "10")]
    coverage_in_percent: usize,

    ///Size of the buffer from which addresses are sampled. If hugepages are used (default) this should be a multiple of a GiB.
    #[clap(long, default_value = "1024")]
    buffer_size_in_mb: usize,

    ///Access time above which we consider an access to be a row conflict
    #[clap(long)]
    conflict_threshold: u64,

    ///Access time below which we consider an access to be no row conflict (Must be <= conflict_threshold, allows to define a "dead zone")
    #[clap(long)]
    no_conflict_threshold: u64,

    ///Average a single time measurement over this many accesses
    #[clap(short, long, default_value = "1000")]
    rounds_per_measurement: usize,

    ///If set, the memory buffer from which addresses are samples won't be backed by hugepages
    #[clap(long)]
    dont_use_hugepages: bool,
}

impl CliArgs {
    fn validate(&self) -> Result<()> {
        if self.no_conflict_threshold > self.conflict_threshold {
            bail!(format!(
                "no_conflict_threshold ({}) must be <= conflict_threshold ({})",
                self.no_conflict_threshold, self.conflict_threshold
            ));
        }

        if self.coverage_in_percent <= 0 || self.coverage_in_percent > 100 {
            bail!("Coverage (in percent) must be > 0 and <= 0")
        }

        Ok(())
    }
}

#[derive(Deserialize, Debug)]
struct DRAMAddressing {
    rank_bank_function: Vec<u64>,
    row_function: Vec<u64>,
}

fn main() -> Result<()> {
    let args: CliArgs = CliArgs::parse();

    if let Err(e) = args.validate() {
        eprintln!("Invalid Config!");
        bail!(e);
    }

    let dram_fns =
        File::open(args.dram_function_config_path).with_context(|| "failed to open config file")?;
    let dram_fns: DRAMAddressing =
        serde_yaml::from_reader(dram_fns).with_context(|| "failed to parse dram address config")?;

    if dram_fns.row_function.len() == 1 {
        bail!("dram row function has only one element. Most likely you are using the wrong format");
    }

    eprintln!("Dram Config rank bank {:x?}", dram_fns);

    //
    //Setup
    //

    let alloc_flags;
    if args.dont_use_hugepages {
        alloc_flags = MapFlags::MAP_PRIVATE | MapFlags::MAP_ANONYMOUS | MapFlags::MAP_POPULATE;
    } else {
        alloc_flags = MapFlags::MAP_PRIVATE
            | MapFlags::MAP_ANONYMOUS
            | MapFlags::MAP_HUGETLB
            | MapFlags::MAP_POPULATE;
    }

    let virt_to_phys =
        Box::new(LinuxPageMap::new().with_context(|| "failed to instantiate virt_to_phys mapper")?);

    let mut buf = memory::MemoryBuffer::new(
        args.buffer_size_in_mb * 1024 * 1024,
        ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
        alloc_flags,
        virt_to_phys,
    )
    .with_context(|| "Failed to create buffer")?;

    let timer = Box::new(DefaultMemoryTupleTimer {});

    //
    // Program Logic
    //
    /*
       //
       // Test 1: Rows inside one bank
       //

       //get one representative for each row in a single bank, only considering bank addressing bits
       //up to max_bit argument
       let mut rows_in_bank = get_all_rows_in_bank_upto_bit_x_by_mask(
           &mut Box::new(&mut buf),
           &dram_fns.rank_bank_function,
           30,
           dram_fns.row_mask_candidates[0],
           args.alignment,
       )?;

       //check that first entry has row conflict with all other entries

       rows_in_bank.sort_by_key(|v| v.phys & dram_fns.row_mask_candidates[0]);
       let base = &rows_in_bank[0];
       for other in rows_in_bank.iter().skip(1) {
           let timing;
           unsafe {
               timing = timer.time_subsequent_access_from_ram(
                   base.ptr,
                   other.ptr,
                   args.rounds_per_measurement,
               );
           }
           assert_eq!(
               evaluate_addr_function(&dram_fns.rank_bank_function, base.phys),
               evaluate_addr_function(&dram_fns.rank_bank_function, other.phys)
           );
           assert_ne!(
               base.phys & dram_fns.row_mask_candidates[0],
               other.phys & dram_fns.row_mask_candidates[0]
           );
           /*eprintln!(
               "{:x} {:x}, Banks {:x?} {:x?}, Rows {:08x} {:08x}, timing {}",
               base.phys,
               other.phys,
               evaluate_addr_function(&dram_fns.rank_bank_function, base.phys),
               evaluate_addr_function(&dram_fns.rank_bank_function, other.phys),
               base.phys & dram_fns.row_mask_candidates[0],
               other.phys & dram_fns.row_mask_candidates[0],
               timing,
           );*/
           if timing < args.no_conflict_threshold {
               eprintln!(
                   "Expected conflict for {:x} and {:x} but not NO conflict timing {}",
                   base.phys, other.phys, timing
               );
           }
       }

       /* //check that all of these entries give a row conflict
       let total_work = binomial(rows_in_bank.len(), 2);
       let mut finished_work = 0;
       eprintln!("{} out of {}", finished_work, total_work);
       let mut counterexamples = Vec::new();
       for a in rows_in_bank.iter().combinations(2) {
           let timing;
           unsafe {
               timing = timer.time_subsequent_access_from_ram(
                   a[0].ptr,
                   a[1].ptr,
                   args.rounds_per_measurement,
               );
           }
           if timing < args.conflict_threshold {
               counterexamples.push((a[0].phys, a[1].phys, timing));
           }
           finished_work += 1;
           console::Term::stderr()
               .clear_last_lines(1)
               .expect("clear line failed");
           eprintln!(
               "{} out of {} , {} counterexamples",
               finished_work,
               total_work,
               counterexamples.len()
           );
       }
       eprintln!("Counterexamples {:x?}", counterexamples);

       */

       eprintln!(
           "Same set test is done! Got {} rows in total",
           rows_in_bank.len()
       );

    */

    //
    // Test 2: Random offset pairs
    //

    //sample offsets from buf for test
    let entry_count = buf.size_in_bytes() / args.alignment;
    let mut covered_entries =
        ((entry_count as f64) * (args.coverage_in_percent as f64 / 100.0)) as usize;
    if covered_entries == 1 || covered_entries == 0 {
        bail!("alignment {:x} combined with coverage {} leads to 0 entries assuming an event entry count, ",args.alignment,args.coverage_in_percent);
    }
    if (covered_entries % 2) != 0 {
        covered_entries -= 1;
    }
    let random_offsets = buf
        .get_random_offsets(args.alignment, covered_entries)
        .with_context(|| "failed to sample random offsets")?;
    let (offsets1, offsets2) = random_offsets.split_at(covered_entries / 2);

    //setup progress printing
    let total_work = (covered_entries / 2) as f64;
    let mut finished_work: f64 = 0.0;
    eprintln!(
        "[1/1] Checking address function on {} random tuples ...",
        covered_entries / 2
    );
    eprintln!("Progress: {:.1}%", (finished_work / total_work) * 100.0);

    //main measure loop
    let mut counter_examples: Vec<(u64, u64, u64)> = Vec::new();
    let mut row_conflict_counter = 0;
    for (off1, off2) in offsets1.iter().zip(offsets2.iter()) {
        let off_1_addr = buf
            .offset(*off1)
            .with_context(|| format!("failed to get offset {:x} from memory buffer", off1))?;
        let off_2_addr = buf
            .offset(*off2)
            .with_context(|| format!("failed to get offset {:x} from memory buffer", off2))?;

        let off1_rank_bank =
            rank_bank::evaluate_addr_function(&dram_fns.rank_bank_function, off_1_addr.phys);
        let off1_row = rank_bank::evaluate_addr_function(&dram_fns.row_function, off_1_addr.phys);

        let off2_rank_bank =
            rank_bank::evaluate_addr_function(&dram_fns.rank_bank_function, off_2_addr.phys);
        let off2_row = rank_bank::evaluate_addr_function(&dram_fns.row_function, off_2_addr.phys);

        /*
        eprintln!(
            "off1_row = {:?}, off1_rank_bank {:?}",
            off1_row, off1_rank_bank
        );
        eprintln!(
            "off2_row = {:?}, off2_rank_bank {:?}",
            off2_row, off2_rank_bank
        );
         */

        let time;
        unsafe {
            loop {
                time = timer.time_subsequent_access_from_ram(
                    off_1_addr.ptr,
                    off_2_addr.ptr,
                    args.rounds_per_measurement,
                );
                break;
                //allows to sample until we get a "good" measurement
                // if (time > args.conflict_threshold) || (time < args.no_conflict_threshold) {
                //    break;
                //}
            }
        }

        if time > args.conflict_threshold {
            row_conflict_counter += 1;
            //if here, timing says row conflict, check that this matches with our addressing function

            //we cannot have a row conflict, if we are in different banks
            if !off1_rank_bank.eq(&off2_rank_bank) {
                counter_examples.push((off_1_addr.phys, off_2_addr.phys, time));
                println!("accessing 0x{:x} and 0x{:x} (xor diff 0x{:09x}) gives row conflict timing {} but our function says different banks!\n",
                         off_1_addr.phys, off_2_addr.phys,off_1_addr.phys ^ off_2_addr.phys ,  time);
            }
            //we are in the same bank
            //we cannot have a row conflict if we are in the same bank and in the same row

            if off1_row.eq(&off2_row) {
                counter_examples.push((off_1_addr.phys, off_2_addr.phys, time));

                println!("accessing 0x{:x} and 0x{:x} (xor diff 0x{:09x}) gives row conflict timing {} but our function says same rank+bank+row!\n",
                         off_1_addr.phys, off_2_addr.phys,off_1_addr.phys ^ off_2_addr.phys, time);
            }
        }
        /*
        else if time < args.no_conflict_threshold {
            //if here, timing says no row conflict, check that this matches with our addressing function
            if (off1_rank_bank.eq(&off2_rank_bank)) && (off1_row == off2_row) {
                counter_examples.push((off_1_addr.phys, off_2_addr.phys, time));
                println!("accessing 0x{:x} and 0x{:x} gives NO row conflict timing {} but our function says but our functions says same bank different row!\n",
                         off_1_addr.phys, off_2_addr.phys, time);
            }
        } else {
            panic!("Got measurement in between conflict and no conflict threshold although we filtered for this")
        }
         */

        console::Term::stderr()
            .clear_last_lines(1)
            .expect("clear line failed");
        finished_work += 1.0;
        eprintln!("Progress: {:.1}%", (finished_work / total_work) * 100.0);
    }

    println!("Encountered {} row conflict accesses", row_conflict_counter);
    println!("Done!\nCounterexamples are : {:x?}", counter_examples);
    println!(
        "This is equal to {:.4}% of accesses",
        ((counter_examples.len() as f64) / total_work) * 100.0
    );

    Ok(())
}
