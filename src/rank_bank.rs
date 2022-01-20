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
pub fn evaluate_addr_function(masks: &Vec<u64>, value: u64) -> Vec<u64> {
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

struct XBitPermutationIter {
    last_mask: u64,
    current_mask: u64,
    inited: bool,
}

impl XBitPermutationIter {
    ///msb is *NOT* index but logical counting (i.e. first is 1)
    fn new(bit_count: usize, msb: usize, initial_shift: usize) -> XBitPermutationIter {
        let first_mask = (1u64 << bit_count) - 1;
        let last_mask = first_mask << (msb - bit_count);

        let first_mask = first_mask << initial_shift;
        let last_mask = last_mask << initial_shift;

        XBitPermutationIter {
            current_mask: first_mask,
            last_mask,
            inited: false,
        }
    }
}

impl Iterator for XBitPermutationIter {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_mask == self.last_mask {
            return None;
        }
        if !self.inited {
            self.inited = true;
            return Some(self.current_mask);
        }

        let t = self.current_mask | self.current_mask.wrapping_sub(1);
        self.current_mask = t.wrapping_add(1)
            | ((!t & ((!t).overflowing_neg()).0)
                .wrapping_sub(1)
                .overflowing_shr(self.current_mask.trailing_zeros() + 1))
            .0;

        Some(self.current_mask)
    }
}
#[cfg(test)]
mod test_x_bit_permutation_iter {
    use super::XBitPermutationIter;
    #[test]
    fn test_permutations_simple() {
        let perm_iter = XBitPermutationIter::new(2, 3, 0);
        let want_values: Vec<u64> = vec![0x3, 0x5, 0x6];
        let got_values = perm_iter.collect::<Vec<u64>>();
        assert_eq!(
            want_values, got_values,
            "Wanted {:x?} got {:x?},",
            want_values, got_values
        );
    }

    #[test]
    fn test_permutations_large() {
        let bit_count = 3;
        let msb = 10;
        let perm_iter = XBitPermutationIter::new(bit_count, msb, 0);
        //there should be "bit_count choose msb" many results
        let want_result_count = 120;

        let got_result_count = perm_iter.count();

        assert_eq!(
            want_result_count, got_result_count,
            "Unexpected number of results, wanted {} got {}",
            want_result_count, got_result_count
        )
    }
}

///DramAnalyzer allows to reverse engineer the rank/bank and row mapping of DRAM memory
/// based on the timing side channel presented in the DRAMA paper
pub struct DramAnalyzer {
    memory_source: Box<dyn super::memory::MemorySource>,
    timer: Box<dyn super::MemoryTupleTimer>,
    ///average timing measurements over this many repetitions
    measure_rounds: usize,
    /// timing threshold for same rank/bank but different row
    conflict_threshold: u64,
}

impl DramAnalyzer {
    /// Removes linear dependents masks from the input and returns results in a new vec
    ///
    ///Directly taken from trrespass drama code
    /// Original comment: https://www.cs.umd.edu/~gasarch/TOPICS/factoring/fastgauss.pdf gaussian elimination in GF2
    fn remove_linear_dependent_masks(masks: &Vec<u64>) -> Result<Vec<u64>> {
        //compute matrix dimensions
        let height = masks.len();
        let width = match masks.iter().map(|v| 64 - v.leading_zeros()).max() {
            None => bail!("masks vec may not be empty"),
            Some(v) => v as usize,
        };
        //fill matrix
        let mut matrix = vec![vec![false; width]; height];

        for i in 0..height {
            for j in 0..width {
                // != 0 to convert to bool as C would do
                matrix[i][width - j - 1] = (masks[i] & (1_u64 << j)) != 0;
            }
        }

        //also build transposed matrix. N.B: suffix _t on the names
        let height_t = width;
        let width_t = height;
        let mut matrix_t = vec![vec![false; width_t]; height_t];

        for i in 0..height {
            for j in 0..width {
                // != 0 to convert to bool
                matrix_t[j][i] = matrix[i][j];
            }
        }

        //i guess row reduce algorithm to get rid of linear combinations?

        let mut pvt_col = 0;
        let mut filtered_masks = Vec::new();

        while pvt_col < width_t {
            for row in 0..height_t {
                if matrix_t[row][pvt_col] {
                    filtered_masks.push(masks[pvt_col]);
                    for c in 0..width_t {
                        if c == pvt_col {
                            continue;
                        }
                        if !(matrix_t[row][c]) {
                            continue;
                        }

                        //original comment : column sum
                        for r in 0..height_t {
                            //Compute "XOR" of the two bool values. N.B. that "!=" on bool equals XOR
                            //on GF(2)
                            matrix_t[r][c] = matrix_t[r][c] != matrix_t[r][pvt_col];
                        }
                    }
                    break;
                }
            }
            pvt_col += 1;
        }

        Ok(filtered_masks)
    }

    /// Selects one representative for each set and checks that the rest of the values inside
    /// the set have the same value when ANDed with mask. Returns the number of entries that fail this test.
    ///
    /// #Arguments
    /// * `row_conflict_sets` sets of address that have a row conflict (same rank+bank but different row)
    /// * `rank_bank_function_candidate` candidate for a single bit of the rank bank function in bitmask form.
    ///
    /// We could also check all pairs inside a set, but a single representative should be fine
    /// The reasoning for this test is to find rank_bank address masks matching all the observations that we made
    fn check_conflict_sets_against_function(
        row_conflict_sets: &Vec<HashSet<MemoryAddress>>,
        rank_bank_function_candidate: u64,
    ) -> Result<usize> {
        let mut counter_examples = 0;
        for set in row_conflict_sets.iter() {
            if set.len() == 0 {
                bail!("found set with zero elements, this should never happen");
            }
            let set_representative = set.iter().take(1).collect::<Vec<&MemoryAddress>>()[0];
            let set_representative_fn_value =
                &parity(set_representative.phys & rank_bank_function_candidate);
            counter_examples += set
                .iter()
                .skip(1)
                .map(|e| parity(e.phys & rank_bank_function_candidate))
                .filter(|v| v != set_representative_fn_value)
                .count();
        }

        return Ok(counter_examples);
    }

    /// Selects all functions from the configured search space that "explains" the conflict sets passed to this function
    /// (the functions are evaluated on the physical address).
    /// The result still contains linear dependent function masks
    /// For the search space, we assume a XOR function. As we wan to "explain" the observations the function needs
    /// log2(row_conflict_sets.len()) bits. Each bit of the function may depend on at most `max_function_bits`
    /// which we will represent a AND bit mask. The MSB of each mask is restricted by `msb_index_for_function`.
    /// The first `ignore_low_bits` bits of each address are ignored
    ///
    /// #Arguments
    /// *`row_conflict_sets` : sets of address that have a row conflict (same rank+bank but different row)
    ///  * `max_function_bits` : max amount of bits for a single mask
    /// * `msb_index_for_function` highest bit (INDEX) of the physical address that is still considered/included in the masks
    /// * `ignore_low_bits` ignore this many low bits of the physical address
    fn compute_rank_bank_function_candidates(
        row_conflict_sets: &Vec<HashSet<MemoryAddress>>,
        max_function_bits: usize,
        msb_index_for_function: usize,
        ignore_low_bits: usize,
    ) -> Result<Vec<u64>> {
        let mut candidates: Vec<u64> = Vec::new();

        //iterate over functions with increasing bit complexity
        for current_max_bits in 1..=max_function_bits {
            //iterate over permutations with current_max_bits set (lower than  msb_index_for_function)
            for mask in XBitPermutationIter::new(
                current_max_bits,
                msb_index_for_function + 1,
                ignore_low_bits,
            ) {
                let counter_examples =
                    DramAnalyzer::check_conflict_sets_against_function(&row_conflict_sets, mask)
                        .with_context(|| "failed to evaluate function on row conflict sets")?;
                if counter_examples == 0 {
                    candidates.push(mask);
                }
            }
        }

        match candidates.len() {
            0 => bail!("Did not find any matching functions"),
            _ => Ok(candidates),
        }
    }

    /// Selects all functions from the configured search space that "explains" the conflict sets passed to this function
    /// (the functions are evaluated on the physical address).
    /// For the search space, we assume a XOR function. As we wan to "explain" the observations the function needs
    /// log2(row_conflict_sets.len()) bits. Each bit of the function may depend on at most `max_function_bits`
    /// which we will represent a AND bit mask. The MSB of each mask is restricted by `msb_index_for_function`.
    /// The first `ignore_low_bits` bits of each address are ignored
    ///
    /// #Arguments
    /// *`row_conflict_sets` : sets of address that have a row conflict (same rank+bank but different row)
    ///  * `max_function_bits` : max amount of bits for a single mask
    /// * `msb_index_for_function` highest bit (INDEX) of the physical address that is still considered/included in the masks
    /// * `ignore_low_bits` ignore this many low bits of the physical address
    pub fn compute_rank_bank_functions(
        &self,
        row_conflict_sets: &Vec<HashSet<MemoryAddress>>,
        max_function_bits: usize,
        msb_index_for_function: usize,
        ignore_low_bits: usize,
    ) -> Result<Vec<u64>> {
        let function_candidates = DramAnalyzer::compute_rank_bank_function_candidates(
            row_conflict_sets,
            max_function_bits,
            msb_index_for_function,
            ignore_low_bits,
        )
        .with_context(|| "compute_rank_bank_function_candidates failed")?;

        DramAnalyzer::remove_linear_dependent_masks(&function_candidates)
            .with_context(|| "remove_linear_dependent_masks failed")
    }

    ///search_row_conflict_sets builds sets of memory addresses that are in the same rank/bank but
    ///in a different row. On real DRAM you should get RANK*BANK COUNT many conflict sets.
    /// This function may not terminate if the timing measurements are to noisy or the amount
    /// of required sets is too high.
    ///#Arguments
    ///* `set_count` amount of conflicting addresses sets to search for
    ///* `elems_per_set` require this many conflicting addresses in each set
    pub fn search_row_conflict_sets(
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
    use crate::memory::MemoryAddress;
    use crate::rank_bank::{evaluate_addr_function, parity, DramAnalyzer};
    use nix::sys::mman::{MapFlags, ProtFlags};
    use std::collections::HashSet;

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

    #[test]
    fn test_remove_linear_dependent_masks() {
        let input = vec![0x811, 0x422, 0xc33, 0x200];
        //N.B. changing the order in input may change the result. Reduction should normally be from
        //"left to right"
        let want = vec![0x811, 0x422, 0x200];

        let got = DramAnalyzer::remove_linear_dependent_masks(&input)
            .expect("remove_linear_dependent_masks failed");

        assert_eq!(
            want, got,
            "On input {:x?} we expect reduced maks {:x?} but got {:x?}",
            input, want, got
        );
    }

    #[test]
    fn test_compute_rank_bank_function_candidates_minimal_example() {
        let max_function_bits = 1;
        let msb_index_for_function = 1;
        let ignore_first_bits = 0;

        let bank_0_samples = HashSet::from([
            MemoryAddress {
                phys: 0x0,
                virt: 0x0,
                ptr: std::ptr::null_mut(),
            },
            MemoryAddress {
                phys: 0x1,
                virt: 0x1,
                ptr: std::ptr::null_mut(),
            },
        ]);
        let bank_1_samples = HashSet::from([
            MemoryAddress {
                phys: 0x2,
                virt: 0x2,
                ptr: std::ptr::null_mut(),
            },
            MemoryAddress {
                phys: 0x3,
                virt: 0x3,
                ptr: std::ptr::null_mut(),
            },
        ]);
        //the function search space configured above leaves us with 0x1 and 0x2 as function masks
        //in the input above the first set contains 0x0 and 0x1, thus the mask 0x1 cannot be right
        //the second set also has one entry with bit index 0 set and one with bit index zero unset.
        //For the second bit however, the entries are homogenous. Thus this is a valid mask
        let want_rank_bank_function = vec![0x2];

        let input_sets = vec![bank_0_samples, bank_1_samples];

        let got_rank_bank_candidates = super::DramAnalyzer::compute_rank_bank_function_candidates(
            &input_sets,
            max_function_bits,
            msb_index_for_function,
            ignore_first_bits,
        )
        .expect("compute_rank_bank_function_candidates failed");

        assert_eq!(
            got_rank_bank_candidates, want_rank_bank_function,
            "Unexpected rank_bank function wanted {:x?} got {:x?},",
            want_rank_bank_function, got_rank_bank_candidates,
        )
    }
}
