pub mod memory;
pub mod rank_bank;
use anyhow::{bail, Result};

use std::string::String;

#[cfg(target_arch = "x86_64")]
#[cfg(target_arch = "x86_64")]
use {
    anyhow::Context,
    core::arch::x86_64,
    core::ptr,
    std::arch::asm,
    std::{sync, thread, time},
};

pub trait MemoryTimer {
    unsafe fn time_access(&self, a: *const u8) -> u64;
    unsafe fn flush(&self, a: *const u8);
}

#[cfg(target_arch = "x86_64")]
pub struct DefaultMemoryTimer {}

#[cfg(target_arch = "x86_64")]
impl MemoryTimer for DefaultMemoryTimer {
    unsafe fn time_access(&self, a: *const u8) -> u64 {
        let mut timing = 0;
        asm!(
            "mfence",
            "rdtsc",      /*writes to edx:eax*/
            "shl rdx, 32", /*shift low 32 bits in edx to high bits*/
            "or rdx,rax",  /*add low bits stored in eax*/
            "mov rcx, rdx", /*stash measurement in rcx*/
            "mov rax,  [{a}]",
            "lfence",
            "rdtsc",
            "shl rdx, 32", /*shift low 32 bits in edx to high bits*/
            "or rdx,rax",  /*add low bits stored in eax*/
            "sub rdx, rcx", /*calculdate diff*/
            "mov {timing}, rdx",
            a = in(reg) a as u64,
            timing = inout(reg) timing,
            out("rdx") _, /*mark rdx as clobbered*/
            out("rax") _, /*mark rax as clobbered*/
            out("rcx") _, /*mark rcx as clobbered*/

        );
        return timing;
    }

    unsafe fn flush(&self, a: *const u8) {
        asm!(
            "clflush [{a}]",
            a = in(reg) a as u64,
        );
    }
}

pub fn construct_timer_from_cli_arg(arg_string: &str) -> Result<Box<dyn MemoryTupleTimer>> {
    let arg_string = String::from(arg_string);
    let tokens: Vec<&str> = arg_string.split(",").collect();

    if tokens.len() < 1 {
        bail!("unknown timer");
    }

    #[cfg(target_arch = "x86_64")]
    match tokens[0] {
        "rdtsc" => Ok(Box::new(DefaultMemoryTupleTimer {})),
        "rdtsc-asm" => Ok(Box::new(AsmMemoryTupleTimer {})),
        "counting_thread" => {
            if tokens.len() != 3 {
                bail!("counting_thread usage: counting_thread,<main cpu>,<counting thread cpu>")
            }
            let main_cpu: usize = tokens[1]
                .parse()
                .with_context(|| format!("failed to parse {} to string", tokens[1]))?;
            let counting_thread_cpu: usize = tokens[2]
                .parse()
                .with_context(|| format!("failed to parse {} to string", tokens[2]))?;

            Ok(Box::new(CountingThreadTupleTimer::new(
                main_cpu,
                counting_thread_cpu,
            )))
        }
        _ => bail!("unknown timer on x86_64 target"),
    }

    #[cfg(target_arch = "aarch64")]
    match tokens[0] {
        _ => bail!("unknown timer on aarch64 target "),
    }
}

pub trait MemoryTupleTimer {
    unsafe fn time_subsequent_access_from_ram(
        &self,
        a: *const u8,
        b: *const u8,
        rounds: usize,
    ) -> u64;
}

#[cfg(target_arch = "x86_64")]
pub struct DefaultMemoryTupleTimer {}

#[cfg(target_arch = "x86_64")]
impl MemoryTupleTimer for DefaultMemoryTupleTimer {
    ///time_subsequent_access_from_ram measures the access time when accessing both memory locations back to back from ram.
    /// #Arguments
    /// * `a` pointer to first memory location
    /// * `b` pointer to second memory location
    /// * `rounds` average the access time over this many accesses
    unsafe fn time_subsequent_access_from_ram(
        &self,
        a: *const u8,
        b: *const u8,
        rounds: usize,
    ) -> u64 {
        let mut sum = 0;
        //flush data from cache
        x86_64::_mm_clflush(a);
        x86_64::_mm_clflush(b);

        for _run_idx in 1..rounds {
            x86_64::_mm_mfence(); //ensures clean slate memory access time wise
            let before = x86_64::_rdtsc(); // read timestamp counter
            x86_64::_mm_lfence(); //ensure rdtsc is done
            ptr::read_volatile(a);
            ptr::read_volatile(b);
            x86_64::_mm_lfence(); //ensure accesses are done
            let after = x86_64::_rdtsc(); //read second timestamp
            sum += after - before;
            //flush data from cache
            x86_64::_mm_clflush(a);
            x86_64::_mm_clflush(b);
        }

        return sum / rounds as u64;
    }
}

#[cfg(target_arch = "x86_64")]
pub struct AsmMemoryTupleTimer {}

#[cfg(target_arch = "x86_64")]
impl MemoryTupleTimer for AsmMemoryTupleTimer {
    unsafe fn time_subsequent_access_from_ram(
        &self,
        a: *const u8,
        b: *const u8,
        rounds: usize,
    ) -> u64 {
        let mut sum = 0;
        for _run_idx in 1..rounds {
            asm!(
                "mfence",
            "rdtsc",      /*writes to edx:eax*/
            "lfence",
                "shl rdx, 32", /*shift low 32 bits in edx to high bits*/
                "or rdx,rax",  /*add low bits stored in eax*/
                "mov rcx, rdx", /*stash measurement in rcx*/
                "mov rax,  [{a}]",
                "mov rax,  [{b}]",
                "lfence",
                "rdtsc",
                "shl rdx, 32", /*shift low 32 bits in edx to high bits*/
                "or rdx,rax",  /*add low bits stored in eax*/
                "sub rdx, rcx", /*calculdate diff*/
                "add {sum}, rdx",
                "clflush [{a}]",
                "clflush [{b}]",
                a = in(reg) a as u64,
                b = in(reg) b as u64,
                sum = inout(reg) sum,
                out("rdx") _, /*mark rdx as clobbered*/
                out("rax") _, /*mark rax as clobbered*/
                out("rcx") _, /*mark rcx as clobbered*/
            );
        }

        sum / rounds as u64
    }
}

//see https://users.rust-lang.org/t/how-to-have-shared-mutable-state-without-lock/3936
//for data structure "explanation"
#[cfg(target_arch = "x86_64")]
struct RacyCounter {
    counter: std::cell::UnsafeCell<usize>,
}
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for RacyCounter {}

#[cfg(target_arch = "x86_64")]
pub struct CountingThreadTupleTimer {
    counter: sync::Arc<RacyCounter>,
    thread_should_terminate: sync::Arc<sync::Mutex<bool>>,
}

#[cfg(target_arch = "x86_64")]
impl CountingThreadTupleTimer {
    ///Creates a counting thread based timer. Make sure to place the calling thread and
    /// the counting thread on cores that share at least a L2 cache.
    ///##Arguments
    ///* `core_id_main` core id to schedule the calling thread to
    ///* `core_id_counter` core id to schedule the counting thread to
    pub fn new(core_id_main: usize, core_id_counter: usize) -> CountingThreadTupleTimer {
        core_affinity::set_for_current(core_affinity::CoreId { id: core_id_main });
        let counter = sync::Arc::new(RacyCounter {
            counter: std::cell::UnsafeCell::new(0),
        });
        let thread_should_terminate = sync::Arc::new(sync::Mutex::new(false));

        let v = counter.clone();
        let vv = thread_should_terminate.clone();
        thread::spawn(move || {
            core_affinity::set_for_current(core_affinity::CoreId {
                id: core_id_counter,
            });

            //simple putting `*global_counter += 1` lead to quite high access time values.
            //I am not sure why, as locating the disassembly was a bit hard
            //We use a local counter, as this allows us to perform the memory write
            //with ptr::write_volatile
            let global_counter = v.counter.get();
            let mut local_counter: usize = 0;

            loop {
                //long counting sequence
                for _i in 1..1000000000000_u64 {
                    unsafe {
                        local_counter += 1;
                        ptr::write_volatile(global_counter, local_counter);
                    }
                }
                //check if we should terminate
                {
                    let should_terminate = vv.lock().unwrap();
                    if *should_terminate {
                        break;
                    }
                }
            }
        });

        //give timer some time to startup
        std::thread::sleep(time::Duration::from_secs(4));

        CountingThreadTupleTimer {
            counter,
            thread_should_terminate,
        }
    }
}
#[cfg(target_arch = "x86_64")]
impl Drop for CountingThreadTupleTimer {
    fn drop(&mut self) {
        let mut term = self.thread_should_terminate.lock().unwrap();
        *term = true;
    }
}
#[cfg(target_arch = "x86_64")]
impl MemoryTupleTimer for CountingThreadTupleTimer {
    unsafe fn time_subsequent_access_from_ram(
        &self,
        a: *const u8,
        b: *const u8,
        rounds: usize,
    ) -> u64 {
        let mut sum = 0;
        let counter_ptr = self.counter.counter.get();
        //flush data from cache
        x86_64::_mm_clflush(a);
        x86_64::_mm_clflush(b);

        for _run_idx in 1..rounds {
            x86_64::_mm_mfence(); //ensures clean slate memory access time wise
            let before = ptr::read_volatile(counter_ptr);
            x86_64::_mm_lfence(); //timer read is done
            ptr::read_volatile(a);
            ptr::read_volatile(b);
            x86_64::_mm_lfence(); //ensure accesses are done
            let after = ptr::read_volatile(counter_ptr); //read second timestamp
            sum += (after - before) as u64;
            //flush data from cache
            x86_64::_mm_clflush(a);
            x86_64::_mm_clflush(b);
        }

        return sum / rounds as u64;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
