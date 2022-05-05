use anyhow::{Context, Result};
use nix::sys::mman::{MapFlags, ProtFlags};
use no_drama::memory::{LinuxPageMap, MemorySource};
use no_drama::{memory, MemoryTupleTimer};
use std::time;

extern crate core_affinity;

fn main() -> Result<()> {
    core_affinity::set_for_current(core_affinity::CoreId { id: 0 });

    let alloc_flags = MapFlags::MAP_PRIVATE
        | MapFlags::MAP_ANONYMOUS
        | MapFlags::MAP_HUGETLB
        | MapFlags::MAP_POPULATE;

    let virt_to_phys =
        LinuxPageMap::new().with_context(|| "failed to instantiate virt_to_phys mapper")?;

    let mut buf = memory::MemoryBuffer::new(
        1024 * 1024 * 1024,
        ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
        alloc_flags,
        Box::new(virt_to_phys),
    )
    .with_context(|| "Failed to create buffer")?;

    let timer = no_drama::CountingThreadTupleTimer::new(0, 8);

    //sleep some time to allow counting thread to get going
    std::thread::sleep(time::Duration::from_secs(4));

    let addr1 = buf.offset(0)?;
    let addr2 = buf.offset(4096)?;

    for _i in 1..10 {
        let timing = unsafe { timer.time_subsequent_access_from_ram(addr1.ptr, addr2.ptr, 1000) };
        println!("Timing with counting thread is {}", timing);
    }

    //terminate counting thread
    std::mem::drop(timer);

    let timer = no_drama::DefaultMemoryTupleTimer {};
    for _i in 1..10 {
        let timing = unsafe { timer.time_subsequent_access_from_ram(addr1.ptr, addr2.ptr, 1000) };
        println!("Timing with intrinsics is {}", timing);
    }

    let timer = no_drama::AsmMemoryTupleTimer {};
    for _i in 1..10 {
        let timing = unsafe { timer.time_subsequent_access_from_ram(addr1.ptr, addr2.ptr, 1000) };
        println!("Timing with asm is {}", timing);
    }

    Ok(())
}
