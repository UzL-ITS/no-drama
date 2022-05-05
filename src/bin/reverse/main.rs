use anyhow::{bail, Context, Result};
use clap::Parser;
use nix::sys::mman::{MapFlags, ProtFlags};
use no_drama::memory::MemorySource;
use no_drama::memory::{LinuxPageMap, MemoryAddress};
use no_drama::rank_bank::DramAnalyzer;
use no_drama::{construct_timer_from_cli_arg, memory, DefaultMemoryTupleTimer};
use serde::Serialize;
use std::collections::HashSet;
use std::fs::File;

#[derive(Parser, Debug, Serialize)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    ///Size of the buffer from which addresses are sampled. If hugepages are used (default) this should be a multiple of a GiB.
    #[clap(long, default_value = "5120")]
    buffer_size_in_mb: usize,

    ///Configures the amount of sets to search for. We define "sets" as the total number of banks, i.e. Rank * Banks per Rank
    #[clap(short, long)]
    set_count: usize,

    ///Controls how many elements are sampled per set. A higher value gives higher guarantees that the reverse functions are correct for all memory locations
    #[clap(long, default_value = "40")]
    elements_per_rank_bank_set: usize,

    ///Controls how many elements are sampled per set. A higher value gives higher guarantees that the reverse functions are correct for all memory locations
    #[clap(long, default_value = "10")]
    elements_per_same_row_set: usize,

    ///Max amount of physical address bits a rank+bank function bit may depend on  
    #[clap(long, default_value = "6")]
    max_rank_bank_function_bits: usize,

    ///Highest physical address bit a rank+bank function bit may depend on
    #[clap(long, default_value = "30")]
    msb_for_rank_bank_function: usize,

    ///Amount of least significant bits that are ignored when searching for rank+bank functions
    #[clap(long, default_value = "6")]
    ignore_low_bits_rank_bank: usize,

    /*
       ///Max amount of physical address bits a row function bit may depend on
       #[clap(long, default_value = "16")]
       max_row_function_bits: usize,

       ///Highest physical address bit a row function bit may depend on
       #[clap(long, default_value = "30")]
       msb_for_row_function: usize,

       ///Amount of least significant bits that are ignored when searching for row functions
       #[clap(long, default_value = "6")]
       ignore_low_bits_row: usize,

    */
    ///Access time above which we consider an access to be a row conflict
    #[clap(long)]
    conflict_threshold: u64,

    ///Access time below which we consider an access to be no row conflict (Must be <= conflict_threshold, allows to define a "deadzone")
    #[clap(long)]
    no_conflict_threshold: u64,

    ///Average a single time measurement over this many accesses
    #[clap(short, long, default_value = "4000")]
    rounds_per_measurement: usize,

    ///If set, the memory buffer from which addresses are samples won't be backed by hugepages
    #[clap(long)]
    dont_use_hugepages: bool,

    ///Select timer to time ram accesses.
    #[clap(long, default_value = "rdtsc")]
    timing_source: String,
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

#[derive(Serialize)]
struct DRAMAddressing {
    rank_bank_function: Vec<u64>,
    row_mask_candidates: Vec<u64>,
}

fn main() -> Result<()> {
    let args: CliArgs = CliArgs::parse();

    if let Err(e) = args.validate() {
        eprintln!("Invalid Config!");
        bail!(e);
    }

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

    let mut buf = Box::new(
        memory::MemoryBuffer::new(
            args.buffer_size_in_mb * 1024 * 1024,
            ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            alloc_flags,
            virt_to_phys,
        )
        .with_context(|| "Failed to create buffer")?,
    );

    let timer = construct_timer_from_cli_arg(&args.timing_source)
        .with_context(|| "failed to contruct timer")?;
    let _base_addr1 = buf.get_random_address(8192)?;

    let mut analyzer = DramAnalyzer::new(
        buf,
        timer,
        args.rounds_per_measurement,
        args.conflict_threshold,
        args.no_conflict_threshold,
    );

    //
    // Program Logic
    //

    const NUMBER_OF_TASKS: usize = 5;
    //Build address pool of row conflicts

    eprintln!(
        "[{}/{}] {}",
        1, NUMBER_OF_TASKS, "Searching row conflict sets ..."
    );
    let row_conflict_sets = analyzer
        .search_row_conflict_sets(args.set_count, args.elements_per_rank_bank_set)
        .with_context(|| "Search for row conflict sets failed")?;

    //Compute rank+bank functions

    eprintln!(
        "[{}/{}] {}",
        2, NUMBER_OF_TASKS, "Computing rank+bank functions ..."
    );
    let rank_bank_functions = analyzer
        .compute_rank_bank_functions(
            &row_conflict_sets,
            args.max_rank_bank_function_bits,
            args.msb_for_rank_bank_function - 1,
            args.ignore_low_bits_rank_bank,
        )
        .with_context(|| "Search for rank bank functions failed")?;

    println!("Rank+Bank Functions are {:x?}", rank_bank_functions);

    eprintln!(
        "[{}/{}] {}",
        3, NUMBER_OF_TASKS, "Searching same rank+bank+row sets ..."
    );
    //Select one address of the first 2 row_conflict_sets to search for same rank+bank+row addresses
    let mut same_rank_bank_row_base: Vec<MemoryAddress> = Vec::new();

    for set in row_conflict_sets.iter().take(2) {
        let addr = set.iter().take(1).collect::<Vec<&MemoryAddress>>()[0];
        same_rank_bank_row_base.push(addr.clone());
    }

    let same_rank_bank_rows = analyzer
        .search_same_bank_same_row_sets(
            &same_rank_bank_row_base,
            &rank_bank_functions,
            args.elements_per_same_row_set,
        )
        .with_context(|| "Search for entries with same rank+bank+row failed")?;

    eprintln!(
        "[{}/{}] {}",
        4, NUMBER_OF_TASKS, "Computing possible row masks  via flipping..."
    );
    //unlike for rank+bank, each mask on its own is a valid function
    let possible_row_mask_functions = analyzer
        .compute_row_masks_flipping(&same_rank_bank_rows, &rank_bank_functions)
        .with_context(|| "Search for row mask function via flipping failed")?;
    println!(
        "Possible solutions for the row mask are {:x?}",
        possible_row_mask_functions
    );

    eprintln!(
        "[{}/{}] {}",
        5, NUMBER_OF_TASKS, "Computing possible row masks via drama..."
    );
    //unlike for rank+bank, each mask on its own is a valid function
    let possible_row_mask_functions =
        DramAnalyzer::compute_row_masks_drama(&same_rank_bank_rows, &rank_bank_functions)
            .with_context(|| "Search for row mask function via drama method failed")?;
    if possible_row_mask_functions.len() > 50 {
        println!(
            "There are many possible row mask solutions, showing first 50, see log for the rest {:x?}",
            possible_row_mask_functions.iter().take(50).collect::<Vec<&u64>>()
        );
    } else {
        println!(
            "Possible solutions for the row mask are {:x?}",
            possible_row_mask_functions
        );
    }

    //Save results to files
    let file =
        File::create("reverse-output.yml").with_context(|| " failed to create output file")?;
    serde_yaml::to_writer(&file, &args)
        .with_context(|| "failed to serializes args struct to yml")?;

    let dram_addresses = DRAMAddressing {
        row_mask_candidates: possible_row_mask_functions,
        rank_bank_function: rank_bank_functions,
    };
    serde_yaml::to_writer(&file, &dram_addresses)
        .with_context(|| "failed to serialize dram addresses to yml")?;

    Ok(())
}
