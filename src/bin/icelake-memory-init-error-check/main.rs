use anyhow::{Context, Result};
use clap::Parser;
use nix::sys::mman::{MapFlags, ProtFlags};
use no_drama::memory::LinuxPageMap;
use no_drama::memory::MemorySource;
use no_drama::{memory, DefaultMemoryTimer, MemoryTimer};
use rand::prelude::*;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use std::path::PathBuf;
use std::ptr;

///Similar to bin/measure. However, here we fill the memory with a pseudo random pattern and check
/// if our accesses introduced any bit flips. This was intended to test some theory on the icelake
/// machine
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

    ///Create the measurement files in this folder. Files will be named access-times-run-xy.csv where xy indicates the series
    #[clap(short, long, default_value = "./")]
    output_folder_path: String,

    ///Size of the buffer from which we sample the addresses. Unit is Mib (1024 MiB = 1GiB)
    #[clap(short, long, default_value = "1024")]
    buffer_size_in_mb: usize,

    ///If set, the buffer will not use hugepages for the backing buffer
    #[clap(long)]
    dont_use_hugepages: bool,
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

    let timer = DefaultMemoryTimer {};
    for series_index in 0..args.measure_series_count {
        let random_offsets = buf.get_random_offsets(
            args.alignment_off_addresses,
            args.address_pairs_per_measurement,
        )?;

        let mut access_times = Vec::new();
        //phase 1 : fill memory with deterministic random sequence
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        for (_, off1) in random_offsets.iter().enumerate() {
            //fill memory and flush
            let a = buf
                .offset(*off1)
                .with_context(|| format!("failed to get offset {:x} from memory buffer", off1))?;
            unsafe {
                let fill_value = rng.gen();
                ptr::write_bytes(a.ptr, fill_value, args.alignment_off_addresses);
                timer.flush(a.ptr);
            }
        }

        eprintln!("Memory init done\n");
        rng = ChaCha20Rng::seed_from_u64(42);

        //phase 2 : measure access time AND check if value is still correct
        for (_, off1) in random_offsets.iter().enumerate() {
            //TODO: check value contained in ram
            let timing;
            let a = buf
                .offset(*off1)
                .with_context(|| format!("failed to get offset {:x} from memory buffer", off1))?;
            unsafe {
                timing = timer.time_access(a.ptr);
            }
            let want = rng.gen();
            for _ in 0..args.alignment_off_addresses {
                let got;
                unsafe { got = ptr::read(a.ptr) }
                if got != want {
                    eprintln!("Memory error, want {:x} got {:x}", want, got)
                }
            }
            access_times.push(timing);
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
