pub mod memory;

use core::arch::x86_64;
use core::ptr;

///time_subsequent_access_from_ram measures the access time when accessing both memory locations back to back from ram.
/// #Arguments
/// * `a` pointer to first memory location
/// * `b` pointer to second memory location
/// * `rounds` average the access time over this many accesses
pub unsafe fn time_subsequent_access_from_ram(a: *const u8, b: *const u8, rounds: usize) -> u64 {
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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
