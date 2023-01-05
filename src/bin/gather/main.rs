use anyhow::{bail, Context, Result};
use clap::Parser;
use nix::sys::mman::{MapFlags, ProtFlags};
use no_drama::memory;
use no_drama::memory::{
    LinuxPageMap, MemoryAddress, MemoryBuffer, MemorySource, VirtToPhysResolver,
};
use no_drama::MemoryTupleTimer;
use rand::{Rng, SeedableRng};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    ///Amount of memory tuples to sample
    #[clap(long, default_value = "10000")]
    sample_size: usize,

    //Output file path
    #[clap(short, long, default_value = "address-pairs.csv")]
    output: String,

    ///Alignment in bytes for sampled addresses
    #[clap(long, default_value = "64")]
    alignment: usize,

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

    ///Choose verification mode from ["row-conflict"]
    #[clap(long)]
    mode: String,

    ///Use fixed sequence. Created by fixed seed rng
    #[clap(long)]
    fixed_sequence: bool,
}

impl CliArgs {
    fn validate(&self) -> Result<()> {
        if self.no_conflict_threshold > self.conflict_threshold {
            bail!(format!(
                "no_conflict_threshold ({}) must be <= conflict_threshold ({})",
                self.no_conflict_threshold, self.conflict_threshold
            ));
        }

        if self.mode != "row-conflict" {
            bail!("mode argument must be \"row-conflict\"")
        }

        Ok(())
    }
}

fn get_pseudo_random_offset(
    rng: &mut rand::rngs::StdRng,
    buf_size_in_bytes: usize,
    alignment: usize,
) -> Result<usize> {
    if alignment >= buf_size_in_bytes {
        bail!(format!(
            "requested alignment {} is larger than buffer size {}",
            alignment, buf_size_in_bytes
        ))
    }
    let entry_count = buf_size_in_bytes / alignment;
    let index = rng.gen_range(0..entry_count);
    Ok(index * alignment)
}

fn sample_row_conflict_tuples(
    timer: Box<dyn MemoryTupleTimer>,
    buf: &mut Box<MemoryBuffer>,
    args: &CliArgs,
) -> Result<Vec<(MemoryAddress, MemoryAddress, u64)>> {
    let mut row_conflict_tuples: Vec<(MemoryAddress, MemoryAddress, u64)> = Vec::new();

    eprintln!(
        "Found {} out of {}",
        row_conflict_tuples.len(),
        args.sample_size
    );

    let mut pseudo_rng = rand::rngs::StdRng::seed_from_u64(41548468784856842);

    while row_conflict_tuples.len() < args.sample_size {
        let a: MemoryAddress;
        let b: MemoryAddress;
        if args.fixed_sequence {
            a = buf.offset(get_pseudo_random_offset(
                &mut pseudo_rng,
                buf.size_in_bytes(),
                args.alignment,
            )?)?;
            b = buf.offset(get_pseudo_random_offset(
                &mut pseudo_rng,
                buf.size_in_bytes(),
                args.alignment,
            )?)?;
        } else {
            a = buf.get_random_address(args.alignment)?;
            b = buf.get_random_address(args.alignment)?;
        }

        let time;
        unsafe {
            time = timer.time_subsequent_access_from_ram(a.ptr, b.ptr, args.rounds_per_measurement)
        }
        if time > args.conflict_threshold {
            row_conflict_tuples.push((a, b, time));
            console::Term::stderr()
                .clear_last_lines(1)
                .expect("clear line failed");
            eprintln!(
                "Found {} out of {}",
                row_conflict_tuples.len(),
                args.sample_size
            )
        }
    }

    Ok(row_conflict_tuples)
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

    let mut virt_to_phys =
        LinuxPageMap::new().with_context(|| "failed to instantiate virt_to_phys mapper")?;

    let buf = memory::MemoryBuffer::new(
        args.buffer_size_in_mb * 1024 * 1024,
        ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
        alloc_flags,
        Box::new(virt_to_phys),
    )
    .with_context(|| "Failed to create buffer")?;

    let timer = Box::new(
        no_drama::construct_timer_from_cli_arg("rdtsc")
            .with_context(|| "failed to instantiate timer")?,
    );

    //
    // Program Logic
    //

    if args.mode == "row-conflict" {
        let samples = sample_row_conflict_tuples(*timer, &mut Box::new(buf), &args)
            .with_context(|| "failed to samples row conflict tuples")?;

        let mut wtr =
            csv::Writer::from_path(args.output).with_context(|| " failed to create output file")?;

        wtr.write_record(&["addr a", "addr b", "timing in cycles"])?;

        for (a, b, timing) in samples.iter() {
            wtr.serialize(&[a.phys, b.phys, *timing])?
        }
    }

    Ok(())
}
