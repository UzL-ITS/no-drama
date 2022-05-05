use anyhow::{bail, Context, Result};
use clap::Parser;
use itertools::Itertools;
use nix::sys::mman::{MapFlags, ProtFlags};
use no_drama::memory::{LinuxPageMap, MemorySource};
use no_drama::{memory, DefaultMemoryTupleTimer, MemoryTupleTimer};
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashSet;
use std::fs::File;

#[derive(Parser, Debug, Serialize)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    ///Size of the buffer from which addresses are sampled. If hugepages are used (default) this should be a multiple of a GiB.
    #[clap(long, default_value = "5120")]
    buffer_size_in_mb: usize,

    ///Max amount of physical address bits a rank+bank function bit may depend on  
    #[clap(long, default_value = "30")]
    max_phy_addr_bit: usize,

    ///Amount of least significant bits that are ignored when searching for rank+bank functions
    #[clap(long, default_value = "6")]
    ignore_low_bits: usize,

    ///Access time above which we consider an access to be a row conflict
    #[clap(long)]
    conflict_threshold: u64,

    ///Access time below which we consider an access to be no row conflict (Must be <= conflict_threshold, allows to define a "deadzone")
    #[clap(long)]
    no_conflict_threshold: u64,

    ///Average a single time measurement over this many accesses
    #[clap(short, long, default_value = "4000")]
    rounds_per_measurement: usize,

    ///Yaml with array entry "rank_bank_function" and "row_function" containing one mask for each function bit
    #[clap(long, default_value = "dram-fns.yml")]
    dram_function_config_path: String,
}

impl CliArgs {
    fn validate(&self) -> Result<()> {
        if self.no_conflict_threshold > self.conflict_threshold {
            bail!(format!(
                "no_conflict_threshold ({}) must be <= conflict_threshold ({})",
                self.no_conflict_threshold, self.conflict_threshold
            ));
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

    eprintln!("Parsed config");
    //
    //Setup
    //

    let alloc_flags = MapFlags::MAP_PRIVATE
        | MapFlags::MAP_ANONYMOUS
        | MapFlags::MAP_HUGETLB
        | MapFlags::MAP_POPULATE;

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

    eprintln!("Conflict above {}", args.conflict_threshold);
    eprintln!("No conflict below {}", args.no_conflict_threshold);

    eprintln!("Build test environment");
    //
    // Program Logic
    //

    let dram_fns = File::open(args.dram_function_config_path.clone()).with_context(|| {
        format!(
            "failed to open config file {}",
            args.dram_function_config_path
        )
    })?;
    let dram_fns: DRAMAddressing =
        serde_yaml::from_reader(dram_fns).with_context(|| "failed to parse dram address config")?;

    eprintln!("Dram address functions {:x?}", dram_fns);

    //set with all physical address bits (indices)
    let set_v: HashSet<u64> = HashSet::from_iter(
        ((args.ignore_low_bits as u64)..(args.max_phy_addr_bit as u64)).into_iter(),
    );
    eprintln!(
        "Considered physical address bit indices: {:?}",
        set_v.iter().copied().sorted().collect::<Vec<u64>>()
    );

    //Built set of bits that are not used for row selection or as part of a rank two xor function

    let mut remaining_bits = set_v.clone();
    for fn_mask in dram_fns
        .rank_bank_function
        .iter()
        .merge(dram_fns.row_function.iter())
    {
        for idx in 0..64 {
            let mask = 1_u64 << idx;
            if (fn_mask & mask) != 0 {
                remaining_bits.remove(&idx);
            }
        }
    }

    eprintln!(
        "Testing the following bit indices: {:?}",
        remaining_bits
            .iter()
            .copied()
            .sorted()
            .collect::<Vec<u64>>()
    );

    //sample row conflict, see if toggling any bits from `remaining_bits` clears the row conflict

    const SAMPLE_SIZE: usize = 10;
    let mut sample_count = 0;

    while sample_count < SAMPLE_SIZE {
        let a = buf.get_random_address(64)?;
        let b = buf.get_random_address(64)?;

        let time;
        unsafe {
            time = timer.time_subsequent_access_from_ram(a.ptr, b.ptr, args.rounds_per_measurement);
        }

        if time < args.conflict_threshold {
            continue;
        }

        for bit_idx in remaining_bits.iter() {
            let flip_mask = 1_u64 << bit_idx;
            let b_flipped = ((b.ptr as u64) ^ flip_mask) as *mut u8;

            let time_flipped;
            unsafe {
                time_flipped = timer.time_subsequent_access_from_ram(
                    a.ptr,
                    b_flipped,
                    args.rounds_per_measurement,
                );
            }
            if time_flipped < args.no_conflict_threshold {
                eprintln!("Bit idx {} is rank/bank candidate", bit_idx)
            }
        }
        sample_count += 1;
    }

    Ok(())
}
