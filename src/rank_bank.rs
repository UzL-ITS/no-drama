use super::memory::MemoryAddress;
use anyhow::{bail, Context, Result};
use std::collections::hash_set::HashSet;
use std::collections::HashMap;

/// parity returns 1 if an odd number of bits is set, else 0
/// Taken from Hacker's delight 2nd edition, p. 96 by Henry S. Warren Jr.
const fn parity(value: u64) -> u64 {
    let mut y: u64 = value ^ (value >> 1);

    y ^= y >> 2;
    y ^= y >> 4;
    y ^= y >> 8;
    y ^= y >> 16;
    y ^= y >> 32;

    0b1 & y
}

///evaluate_addr_function evaluates the specified XOR function and returns the result bitwise
/// #Arguments
/// * `masks`  vector with one mask for each bit of the function. The mask selects the bits that should get XORed
/// for that specific function bit
/// * `value` value on which the function should be evaluated
///
/// #Example
/// ```
/// let masks :Vec<u64> = vec![0x1,0x2,0x4];
/// let value :u64 = 0x5;
/// ```
/// returns `[1,0,1]`
fn evaluate_addr_function(masks: &Vec<u64>, value: u64) -> Vec<u64> {
    masks.iter().map(|m| parity(value & m)).collect()
}

///all_sets_filled returns true if we have at least the specified number of sets and each set contains
/// at least the specified number of elements
/// #Arguments
/// * `sets` input upon which the criteria is evaluated
/// * `set_count` minimum number of entries for `sets`
/// * `elems_per_set` minimum number of entries for each entry of `sets`
fn all_sets_filled(
    sets: &HashMap<usize, HashSet<MemoryAddress>>,
    set_count: usize,
    elems_per_set: usize,
) -> bool {
    //check if we found enough sets
    if sets.len() < set_count {
        return false;
    }

    //check that all sets have at least elems_per_set entries
    match sets
        .iter()
        .map(|(_id, set)| set.len() >= elems_per_set)
        .reduce(|accum, item| accum & item)
    {
        None => false,
        Some(v) => v,
    }
}

///DramAnalyzer allows to reverse engineer the rank/bank and row mapping of DRAM memory
/// based on the timing side channel presented in the DRAMA paper
struct DramAnalyzer {
    memory_source: Box<dyn super::memory::MemorySource>,
    timer: Box<dyn super::MemoryTupleTimer>,
    ///average timing measurements over this many repetitions
    measure_rounds: usize,
    /// timing threshold for same rank/bank but different row
    conflict_threshold: u64,
}

impl DramAnalyzer {
    ///search_row_conflict_sets builds sets of memory addresses that are in the same rank/bank but
    ///in a different row. On real DRAM you should get RANK*BANK COUNT many conflict sets.
    /// This function may not terminate if the timing measurements are to noisy or the amount
    /// of required sets is too high.
    ///#Arguments
    ///* `set_count` amount of conflicting addresses sets to search for
    ///* `elems_per_set` require this many conflicting addresses in each set
    fn search_row_conflict_sets(
        &mut self,
        set_count: usize,
        elems_per_set: usize,
    ) -> Result<Vec<HashSet<MemoryAddress>>> {
        //alignment for the sampled addresses. Cache line bits should not influence the bank/rank functions
        const ADDRESS_ALIGNMENT_IN_BYTES: usize = 64;

        //stores the found sets
        let mut sets: HashMap<usize, HashSet<MemoryAddress>> = HashMap::new();

        //used to check if we have already used an address
        let mut used_physical_addrs: HashSet<u64> = HashSet::new();

        //we use this to get a unique id for each new set. Little bit hacky, but this
        //way we can manage them in a HashSet
        let mut set_id_counter = 0;

        while !all_sets_filled(&sets, set_count, elems_per_set) {
            let new_addr = self
                .memory_source
                .get_random_address(ADDRESS_ALIGNMENT_IN_BYTES)
                .with_context(|| "failed to sample new random address")?;

            if used_physical_addrs.contains(&new_addr.phys) {
                continue;
            }
            used_physical_addrs.insert(new_addr.phys);

            //
            //time new_addr against existing sets. If it has a high access time, we are in same rank/bank but different row
            //and add new_addr to this set. If we have low access time with all existing sets, we put new_addr in a new set.
            //To speed things up, we only compare against the first element of each set
            //

            //time new_addr against sets
            let mut ids_with_conflict_timing: Vec<usize> = Vec::new();
            for (set_id, set) in sets.iter() {
                if set.len() == 0 {
                    bail!("found set with zero elements, this should never happen");
                }
                let addr = set.iter().take(1).collect::<Vec<&MemoryAddress>>()[0];

                unsafe {
                    let timing = self.timer.time_subsequent_access_from_ram(
                        addr.ptr,
                        new_addr.ptr,
                        self.measure_rounds,
                    );

                    if timing > self.conflict_threshold {
                        ids_with_conflict_timing.push(set_id.clone());
                    }
                }
            }

            //insert logic, see comment above for loop
            if ids_with_conflict_timing.len() == 0 {
                let id_for_new_set = set_id_counter;
                set_id_counter += 1;
                sets.insert(id_for_new_set, HashSet::from([new_addr]));
            } else {
                match sets.get_mut(&ids_with_conflict_timing[0]) {
                    None => panic!("ids_with_conflict_timing[0] contained an invalid id"),
                    Some(set) => set.insert(new_addr),
                };
            }

            //found more then one matching set. This happens as same rank_bank and same row addrs always open
            // up a new set. But the probability is rather small. Other reason could be noisy measurements.
            // Delete all but the first
            //one so that the small sets/measurement outliers die and we populate one large set
            if ids_with_conflict_timing.len() > 1 {
                for key in ids_with_conflict_timing.iter().skip(1) {
                    sets.remove(key);
                }
            }
        }

        //normal assert is fine as this is no frequently called code path
        assert_eq!(
            sets.len(),
            set_count,
            "search_row_conflict_sets returned {} sets but was supposed to find only {}",
            sets.len(),
            set_count
        );

        let sets_as_vec = sets.into_values().collect();
        Ok(sets_as_vec)
    }
}

#[cfg(test)]
mod test {
    use super::super::memory::MemoryBuffer;
    use super::super::MemoryTupleTimer;
    use crate::memory;
    use crate::rank_bank::{evaluate_addr_function, parity};
    use nix::sys::mman::{MapFlags, ProtFlags};

    ///MockMemoryTimer evaluates the given rank_bank and row functions and only returns a high
    /// timing for same rank/bank but different row addresses. Useful for testing only.
    struct MockMemoryTimer {
        conflict_threshold: u64,
        emulated_rank_bank_function: Vec<u64>,
        emulated_row_function: Vec<u64>,
    }

    impl MemoryTupleTimer for MockMemoryTimer {
        unsafe fn time_subsequent_access_from_ram(
            &self,
            a: *const u8,
            b: *const u8,
            _rounds: usize,
        ) -> u64 {
            //evaluate emulated rank_bank addresses for bath input addresses
            let rank_bank_addr_a =
                evaluate_addr_function(&self.emulated_rank_bank_function, a as u64);
            let row_addr_a = evaluate_addr_function(&self.emulated_row_function, a as u64);

            let rank_bank_addr_b =
                evaluate_addr_function(&self.emulated_rank_bank_function, b as u64);
            let row_addr_b = evaluate_addr_function(&self.emulated_row_function, b as u64);

            //if our emulated functions place the two addrs in same rank/bank but a different row return high timing, else low
            return if rank_bank_addr_a.eq(&rank_bank_addr_b) && !row_addr_a.eq(&row_addr_b) {
                self.conflict_threshold + 10
            } else {
                self.conflict_threshold - 50
            };
        }
    }

    #[test]
    fn test_parity() {
        let inputs: Vec<u64> = vec![0x1, 0x3, 0x11, 0x33, 0x100];
        let want_parity: Vec<u64> = vec![1, 0, 0, 0, 1];

        assert_eq!(inputs.len(), want_parity.len());

        for (input, want) in inputs.iter().zip(want_parity) {
            let got = parity(input.clone());
            assert_eq!(
                want.clone(),
                got,
                "Input 0x{:x} : Want {} got {}",
                input,
                want,
                got
            );
        }
    }

    #[test]
    fn test_addr_evaluation() {
        let addr = 0xf00;
        let masks = vec![0x100, 0x200];
        let want_addr = vec![1, 1];
        assert_eq!(want_addr, evaluate_addr_function(&masks, addr));
    }

    #[test]
    fn find_two_sets() {
        const CONFLICT_THRESHOLD: u64 = 330;

        struct TestCase {
            description: &'static str,
            want_sets: usize,
            want_elems_per_set: usize,
            rank_bank_function: Vec<u64>,
            row_function: Vec<u64>,
        }

        let test_cases = vec![
            TestCase {
                description: "Simple, 1 bit rank_bank, 2 two bit row",
                want_sets: 2,
                want_elems_per_set: 10,
                rank_bank_function: vec![0x100],
                row_function: vec![0x200, 0x400],
            },
            TestCase {
                description: "Large, 4 bit rank_bank, 4 bit row",
                want_sets: 16,
                want_elems_per_set: 40,
                rank_bank_function: vec![0x100, 0x200, 0x400, 0x800],
                row_function: vec![0x1000, 0x2000, 0x4000, 0x8000],
            },
        ];

        for test_case in test_cases.into_iter() {
            //
            // Setup test environment
            //

            //mock mapper just returns the virtual address as the physical
            let mock_virt_to_phys = Box::new(memory::LinearMockMapper {});

            //alloc real memory buffer but without hugepages
            let memory_source = MemoryBuffer::new(
                1024 * 1024 * 200,
                ProtFlags::PROT_READ,
                MapFlags::MAP_PRIVATE | MapFlags::MAP_ANONYMOUS | MapFlags::MAP_POPULATE,
                mock_virt_to_phys,
            )
            .expect("failed to init buffer for testing");
            //Build mock timer that returns timings according to the given rank/bank and row function
            let mock_timer = MockMemoryTimer {
                conflict_threshold: CONFLICT_THRESHOLD,
                emulated_rank_bank_function: test_case.rank_bank_function,
                emulated_row_function: test_case.row_function,
            };

            //
            // Test function
            //

            let mut config = super::DramAnalyzer {
                measure_rounds: 1,
                memory_source: Box::new(memory_source),
                timer: Box::new(mock_timer),
                conflict_threshold: CONFLICT_THRESHOLD,
            };

            //Check that we got the desired amount of groups
            let got_sets = config
                .search_row_conflict_sets(test_case.want_sets, test_case.want_elems_per_set)
                .expect("unexpected error in search_row_conflict_sets");

            //
            // Evaluate results
            //

            assert_eq!(
                got_sets.len(),
                test_case.want_sets,
                "Test {} : expected {} sets, got {}",
                test_case.description,
                test_case.want_sets,
                got_sets.len()
            );
            for (idx, set) in got_sets.iter().enumerate() {
                assert!(
                    set.len() >= test_case.want_elems_per_set,
                    "Test {} : Set {}, wanted at least {} elems, got only {}",
                    test_case.description,
                    idx,
                    test_case.want_elems_per_set,
                    set.len()
                );
            }
        }
    }
}
