use anyhow::{Context, Result};
use clap::Parser;
use nix::sys::mman::{MapFlags, ProtFlags};
use no_drama::memory::LinuxPageMap;
use no_drama::memory::MemorySource;
use no_drama::{
    construct_timer_from_cli_arg, memory, CountingThreadTupleTimer, DefaultMemoryTupleTimer,
    MemoryTupleTimer,
};
use std::path::PathBuf;

///Program to sample the access time between random addresses from DRAM (i.e. the program
///ensures that the data is not in cache. Useful for figuring out the threshold for row conflicts.
///Also a good baseline to check if time measurements works
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    ///Perform this many individual measurements. Each measurement produces one output file
    #[clap(long = "series", default_value = "1")]
    measure_series_count: usize,

    ///Measure the access time between this many address pers per measurment series
    #[clap(long = "sample-size", default_value = "100000")]
    address_pairs_per_measurement: usize,

    ///Alignment of the sampled addresses in bytes
    #[clap(long = "alignment", default_value = "64")]
    alignment_off_addresses: usize,

    ///Average a single time measurement over this many accesses
    #[clap(short, long, default_value = "1000")]
    rounds_per_measurement: usize,

    ///Create the measurement files in this folder. Files will be named access-times-run-xy.csv where xy indicates the series
    #[clap(short, long, default_value = "./")]
    output_folder_path: String,

    ///Size of the buffer from which we sample the addresses. Unit is Mib (1024 MiB = 1GiB)
    #[clap(short, long, default_value = "1024")]
    buffer_size_in_mb: usize,

    ///If set, the buffer will not use hugepages for the backing buffer
    #[clap(long)]
    dont_use_hugepages: bool,

    ///Select timer to time ram accesses.
    #[clap(long, default_value = "rdtsc")]
    timing_source: String,
}

fn main() -> Result<(), anyhow::Error> {
    //parse cli args

    let args: CliArgs = CliArgs::parse();

    //allocate memory buffer used for the measurements

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

    /*runs the measurements: for each measurement series, sample args.address_pairs_per_measurement
    address pairs, measure the ram access time on back to back access and store the
    results in one csv file per series*/

    let timer = construct_timer_from_cli_arg(&args.timing_source)
        .with_context(|| "failed to contruct timer")?;
    for series_index in 0..args.measure_series_count {
        //get (distinct) random offsets in buf and split after first half to build tuples of random
        //offsets that where both entries are distinct from each other
        let random_offsets = buf.get_random_offsets(64, 2 * args.address_pairs_per_measurement)?;
        let (offsets1, offsets2) = random_offsets.split_at(args.address_pairs_per_measurement);

        let mut access_times = Vec::new();
        for (off1, off2) in offsets1.iter().zip(offsets2) {
            unsafe {
                access_times.push(
                    timer.time_subsequent_access_from_ram(
                        buf.offset(*off1)
                            .with_context(|| {
                                format!("failed to get offset {:x} from memory buffer", off1)
                            })?
                            .ptr,
                        buf.offset(*off2)
                            .with_context(|| {
                                format!("failed to get offset {:x} from memory buffer", off2)
                            })?
                            .ptr,
                        args.rounds_per_measurement,
                    ),
                );
            }
        }

        let output_file_path: PathBuf = [
            args.output_folder_path.clone(),
            format!("access-times-{}.csv", series_index),
        ]
        .iter()
        .collect();
        let mut wtr = csv::Writer::from_path(output_file_path)?;
        wtr.write_record(&["access time in cycles"])
            .with_context(|| format!("failed to write csv header for series {}", series_index))?;

        for entry in access_times.iter() {
            wtr.serialize(entry)
                .with_context(|| format!("failed to write csv entry in series {}", series_index))?;
        }
    }

    Ok(())
}
