use anyhow::{bail, Context, Result};
use clap::Parser;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64;
use nix::sys::mman::{MapFlags, ProtFlags};
use no_drama::memory;
use no_drama::memory::{LinuxPageMap, MemorySource, VirtToPhysResolver};
use std::ptr;

/// Helper binary for recovering DRAM address with the oscilloscope. Given the physical address bit
/// indices in the `toggle_group` argument. We alternate between accessing the canonical address that
/// has all of these bits set to one and accessing the canonical address that has all of these bits
/// set to zero. While this program is running, we can do our oscilloscope measurements
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    ///Alignment in bytes for sampled addresses
    #[clap(long, default_value = "64")]
    alignment: usize,

    ///Size of the buffer from which addresses are sampled. If hugepages are used (default) this should be a multiple of a GiB.
    #[clap(long, default_value = "1024")]
    buffer_size_in_mb: usize,

    ///Comma separated list of phys addr indices that should we toggled together
    #[clap(long)]
    toggle_group: String,
}

impl CliArgs {
    fn validate(&self) -> Result<Vec<u64>> {
        let mut toggle_group = Vec::new();
        for v in str::split(self.toggle_group.as_str(), ",") {
            let as_number = v
                .parse()
                .with_context(|| "failed to convert input to int")?;
            toggle_group.push(as_number);
        }
        if toggle_group.len() == 0 {
            bail!("No indices for toggle_group specified");
        }
        Ok(toggle_group)
    }
}

fn access_addr(args: CliArgs, toggle_group: &Vec<u64>) -> Result<()> {
    //
    //Setup
    //

    let alloc_flags = MapFlags::MAP_PRIVATE
        | MapFlags::MAP_ANONYMOUS
        | MapFlags::MAP_HUGETLB
        | MapFlags::MAP_POPULATE;

    let mut virt_to_phys =
        LinuxPageMap::new().with_context(|| "failed to instantiate virt_to_phys mapper")?;

    let mut buf = memory::MemoryBuffer::new(
        args.buffer_size_in_mb * 1024 * 1024,
        ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
        alloc_flags,
        Box::new(virt_to_phys),
    )
    .with_context(|| "Failed to create buffer")?;

    //
    // Program Logic
    //

    eprintln!("Toggle group is {:?}", toggle_group);

    let base_addr = buf.offset(0).with_context(|| "failed to get base addr")?;

    //here we exploit that for our hugepages up to index 29 virt and phys are equal

    let mut virt_base = base_addr.ptr as u64;
    for idx in toggle_group {
        virt_base = virt_base | (1 << idx)
    }
    let addr_all_one = virt_base as *mut u8;
    let addr_all_zero = base_addr.ptr;
    eprintln!("addr_all_one is 0x{:x}", addr_all_one as u64);

    virt_to_phys =
        LinuxPageMap::new().with_context(|| "failed to instantiate virt_to_phys mapper")?;
    let addr_all_one_phys = virt_to_phys
        .get_phys(addr_all_one as u64)
        .with_context(|| "virt to phys for target_adrr failed:")?;
    println!(
        "RUNNING addr_all_zero {:x} addr_all_one {:x}",
        base_addr.phys, addr_all_one_phys
    );

    loop {
        unsafe {
            x86_64::_mm_clflush(addr_all_zero);
            x86_64::_mm_clflush(addr_all_one);
            x86_64::_mm_mfence();
            ptr::write_volatile(addr_all_zero, 1);
            ptr::write_volatile(addr_all_one, 1);
        }
    }
}

fn main() -> Result<()> {
    let args: CliArgs = CliArgs::parse();

    let parsed_toggle_group = args.validate();
    match parsed_toggle_group {
        Err(e) => {
            eprintln!("Invalid Config!");
            bail!(e);
        }
        Ok(v) => access_addr(args, &v),
    }
}
