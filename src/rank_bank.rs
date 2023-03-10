use std::collections::hash_set::HashSet;
use std::collections::HashMap;

use anyhow::{bail, Context, Result};

use crate::MemoryTupleTimer;

use super::memory::{MemoryAddress, MemoryBuffer, MemorySource};

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
/// * `physical_address` address on which the function should be evaluated
///
/// #Example
/// ```
/// let masks :Vec<u64> = vec![0x1,0x2,0x4];
/// let value :u64 = 0x5;
/// ```
/// returns `[1,0,1]`
pub fn evaluate_addr_function(masks: &Vec<u64>, physical_address: u64) -> Vec<u64> {
    masks.iter().map(|m| parity(physical_address & m)).collect()
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
    //print status updates
    console::Term::stderr()
        .clear_last_lines(1)
        .expect("clear line failed");
    eprintln!(
        "Progress : touched {} out of {} sets, {} addrs missing",
        sets.len(),
        set_count,
        (set_count * elems_per_set)
            - sets
                .iter()
                .take(set_count)
                .map(|(_id, set)| std::cmp::min(elems_per_set, set.len()))
                .sum::<usize>()
    );

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
    shift: usize,
}

impl XBitPermutationIter {
    /// Create an iterator that returns all permutations with the given amount of bits, that
    /// uses at most the specified MSB. Furthermore you can specify to shift the values such that a certain
    /// amount of low bits is ignored. Thus the returned iterator has (`bit_count` out of (`msb`-`excluded_low_bits`) values)
    /// #Arguments
    /// * `bit_count` maximal amount of bits that may be set at once (`bit_count  out of n`)
    /// * `msb` highest bit that may be set (logical counting, i.e. first bit has the number 1)
    /// * `excluded_low_bits` exclude this many lsb bits
    /// ```
    fn new(bit_count: usize, msb: usize, excluded_low_bits: usize) -> XBitPermutationIter {
        let msb = msb - excluded_low_bits;

        let first_mask = (1u64 << bit_count) - 1;
        let last_mask = first_mask << (msb - bit_count);

        XBitPermutationIter {
            current_mask: first_mask,
            last_mask,
            inited: false,
            shift: excluded_low_bits,
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
            return Some(self.current_mask << self.shift);
        }

        let t = self.current_mask | self.current_mask.wrapping_sub(1);
        self.current_mask = t.wrapping_add(1)
            | ((!t & ((!t).overflowing_neg()).0)
                .wrapping_sub(1)
                .overflowing_shr(self.current_mask.trailing_zeros() + 1))
            .0;

        Some(self.current_mask << self.shift)
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
        let perms = XBitPermutationIter::new(3, 8, 4);
        let got: Vec<u64> = perms.collect();
        let want = vec![0x70, 0xb0, 0xd0, 0xe0];
        assert_eq!(got, want, "Got {:x?} want {:x?}", got, want);
    }
}

///Get a address for each row using the given alignment (only considering max_bit bits of the row address) in a single bank.
/// The bank is selected at random. DOES NOT CHECK IF WE HAVE AN ACTUAL ROW CONFLICT. This function only
/// uses the given input args to find addrs that should have a row conflict. It does not do any validation
/// #Arguments
/// * `max_bit` NO index, i.e. first bit is 1 and so on.
///
pub fn get_all_rows_in_bank_upto_bit_x_by_mask(
    memory_source: &mut MemoryBuffer,
    rank_bank_function: &Vec<u64>,
    max_bit: usize,
    row_mask: u64,
    alignment: usize,
) -> Result<Vec<MemoryAddress>> {
    //select an address from a random set as our base value
    let base_addr = memory_source.get_random_address(alignment)?;
    let base_bank_addr = evaluate_addr_function(&rank_bank_function, base_addr.phys);

    //throw away all bits above `max_bit` in row mask
    let row_mask = row_mask & ((1_u64 << max_bit) - 1);
    eprintln!("Row mask after truncating high bits {:x}", row_mask);
    //use number of remaining row bits, to calc number of rows that we should find if we only
    //change address bits up to max_bit
    let want_row_count = 2_usize.pow(row_mask.count_ones());

    let mut found_row_mask_values = HashSet::new();
    found_row_mask_values.insert(base_addr.phys & row_mask);

    let mut found_rows = Vec::new();
    found_rows.push(base_addr);

    let entry_count = memory_source.size_in_bytes() / alignment;
    eprintln!(
        "Iterating over {} entries with alignment {}",
        entry_count, alignment
    );
    eprintln!(
        "(Assuming Hugeapges): smallest paddr 0x{:x} ,largest paddr 0x{:x}",
        memory_source.offset(0)?.phys,
        memory_source.offset((entry_count - 1) * alignment)?.phys
    );

    //this needs to be last print to stderr before loop entry. Otherwise progress printing eats up lines
    eprintln!(
        "Found {} out of {} addresses",
        found_rows.len(),
        want_row_count
    );
    for index in 0..entry_count {
        let offset = index * alignment;
        let candidate = memory_source.offset(offset)?;

        //candidate must be in same bank as base_addr
        let candidate_bank_addr = evaluate_addr_function(&rank_bank_function, candidate.phys);
        if !base_bank_addr.eq(&candidate_bank_addr) {
            continue;
        }

        //skip candidate if we already have an address for that row mask value
        let candidate_row_mask_value = candidate.phys & row_mask;
        if found_row_mask_values.contains(&candidate_row_mask_value) {
            continue;
        }

        //if we come here, we are in the same bank and did not yet see the row mask value
        found_row_mask_values.insert(candidate_row_mask_value);
        found_rows.push(candidate);

        console::Term::stderr()
            .clear_last_lines(1)
            .expect("clear line failed");
        eprintln!(
            "Found {} out of {} addresses",
            found_row_mask_values.len(),
            want_row_count,
        )
    }

    if found_rows.len() < want_row_count {
        bail!(format!(
            "Wanted {} rows but got only {}",
            want_row_count,
            found_rows.len()
        ));
    }

    Ok(found_rows)
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
    ///timing threshold for no row conflict. Can be equal to `conflict_threshold`. Setting it to
    /// a slightly lower value might prevent errors due to noisy measurements
    no_conflict_threshold: u64,
}

impl DramAnalyzer {
    pub fn new(
        memory_source: Box<dyn super::memory::MemorySource>,
        timer: Box<dyn super::MemoryTupleTimer>,
        measure_rounds: usize,
        conflict_threshold: u64,
        no_conflict_threshold: u64,
    ) -> DramAnalyzer {
        DramAnalyzer {
            memory_source,
            timer,
            measure_rounds,
            conflict_threshold,
            no_conflict_threshold,
        }
    }

    pub fn get_all_rows_in_bank_by_timing(
        &mut self,
        base_addr: &MemoryAddress,
        alignment: usize,
    ) -> Result<Vec<MemoryAddress>> {
        let mut found_rows = Vec::new();
        found_rows.push(base_addr.clone());

        let entry_count = self.memory_source.size_in_bytes() / alignment;
        eprintln!(
            "Iterating over {} entries with alignment {}",
            entry_count, alignment
        );
        eprintln!("Found {} rows", found_rows.len(),);
        for index in 0..entry_count {
            let offset = index * alignment;
            let candidate = self.memory_source.offset(offset)?;

            let timing;
            unsafe {
                timing = self.timer.time_subsequent_access_from_ram(
                    base_addr.ptr.clone(),
                    candidate.ptr,
                    self.measure_rounds,
                );
            }

            if timing > self.conflict_threshold {
                found_rows.push(candidate);
            }

            console::Term::stderr()
                .clear_last_lines(1)
                .expect("clear line failed");
            eprintln!("Found {} rows", found_rows.len())
        }
        Ok(found_rows)
    }

    pub fn latency_xiao_general(&mut self, flip_mask: &u64) -> Result<bool> {
        let mut addrs = Vec::new();
        const SAMPLE_SIZE: usize = 400;

        while addrs.len() < SAMPLE_SIZE {
            let candidate = self
                .memory_source
                .get_random_address(64)
                .with_context(|| "get_random_address failed ")?;
            if (candidate.phys & flip_mask) == 0 {
                addrs.push(candidate);
            }
        }

        //measure timing between each addr in addrs against addr with bits from flip_mask flipped
        let mut high_access_time_count = 0;
        for addr in addrs.iter() {
            let base_ptr = addr.ptr;
            //only works if we are in hugepage in idx_for_flip < 29
            let flipped = ((base_ptr as u64) ^ flip_mask) as *const u8;
            unsafe {
                let time = self.timer.time_subsequent_access_from_ram(
                    base_ptr,
                    flipped,
                    self.measure_rounds,
                );
                if time > self.conflict_threshold {
                    high_access_time_count += 1;
                }
            }
        }

        //return true if high access times are in majority
        return Ok(high_access_time_count > (addrs.len() / 2));
    }

    pub fn latency_xiao(&mut self, idx_for_flip: u64) -> Result<bool> {
        //sample 100 addrs
        let mut addrs = Vec::new();
        for _i in 0..100 {
            addrs.push(
                self.memory_source
                    .get_random_address(64)
                    .with_context(|| "get_random_address failed ")?,
            );
        }

        //measure timing between each addr in addrs against addr with bit at idx_for_flip flipped
        //increase high_access_time_count if access time was high
        let flip_mask = 1_u64 << idx_for_flip;
        let mut high_access_time_count = 0;
        for addr in addrs.iter() {
            let base_ptr = addr.ptr;
            //only works if we are in hugepage in idx_for_flip < 29
            let flipped = ((base_ptr as u64) ^ flip_mask) as *const u8;
            unsafe {
                let time = self.timer.time_subsequent_access_from_ram(
                    base_ptr,
                    flipped,
                    self.measure_rounds,
                );
                if time > self.conflict_threshold {
                    high_access_time_count += 1;
                }
            }
        }

        //return true if high access times are in majority
        return Ok(high_access_time_count > (addrs.len() / 2));
    }

    pub fn compute_row_mask_xiao(&mut self) -> Result<u64> {
        let mut row_mask: u64 = 0;
        for bit_idx in 0..=29 {
            if self
                .latency_xiao(bit_idx)
                .with_context(|| format!("latency_xia failed for bit_idx {}", bit_idx))?
            {
                row_mask |= 1_u64 << bit_idx;
            }
        }

        Ok(row_mask)
    }

    pub fn compute_row_masks_flipping(
        &self,
        same_rank_bank_row_sets: &Vec<HashSet<MemoryAddress>>,
        rank_bank_function: &Vec<u64>,
    ) -> Result<Vec<u64>> {
        let rank_bank_mask: u64 = match rank_bank_function
            .iter()
            .copied()
            .reduce(|accum, value| accum | value)
        {
            None => bail!("rank_bank_mask is empty!"),
            Some(v) => v,
        };
        eprintln!("RankBank mask is 0x{:x}", rank_bank_mask);
        let base_addr = same_rank_bank_row_sets[0]
            .iter()
            .take(1)
            .collect::<Vec<&MemoryAddress>>()[0];
        let mut row_mask: u64 = 0;
        for bit_idx in 0..=29 {
            let flip_mask = 1_u64 << bit_idx;

            /*if (rank_bank_mask & flip_mask) != 0 {
                continue;
            }*/

            unsafe {
                let ptr_with_flip = (base_addr.virt ^ flip_mask) as *const u8;
                let time = self.timer.time_subsequent_access_from_ram(
                    base_addr.ptr,
                    ptr_with_flip,
                    self.measure_rounds,
                );
                if time > self.conflict_threshold {
                    row_mask |= flip_mask;
                }
            }
        }

        Ok(vec![row_mask])
    }

    pub fn compute_row_masks_drama(
        same_rank_bank_row_sets: &Vec<HashSet<MemoryAddress>>,
        _rank_bank_function: &Vec<u64>,
    ) -> Result<Vec<u64>> {
        let row_function_candidates =
            DramAnalyzer::brute_force_matching_functions(same_rank_bank_row_sets, 12, 12, 30, 6)?;

        /*for rank_bank_mask in rank_bank_function.iter() {
            let lsb = 1_u64 << (rank_bank_mask.trailing_zeros() + 1);
            for row_mask in row_function_candidates.iter_mut() {
                if (*row_mask & lsb) != 0 {
                    *row_mask = (*row_mask) ^ (1_u64 << rank_bank_mask.trailing_zeros());
                }
            }
        }*/

        Ok(row_function_candidates)
    }
    /// Find addresses that are in the same rank+bank+row as the supplied addresses.
    /// For rank+bank, we use the supplied address function. For same row we use timing
    /// #Arguments
    /// * `addresses` search for same rank+bank+row addresses for each element
    /// * `rank_bank_function` reverse engineered address function for rank+bank.
    /// * `elems_per_set` amount of same rank+bank+row addresses to search for each supplied address
    pub fn search_same_bank_same_row_sets(
        &mut self,
        addresses: &Vec<MemoryAddress>,
        rank_bank_function: &Vec<u64>,
        elems_per_set: usize,
    ) -> Result<Vec<HashSet<MemoryAddress>>> {
        //alignment for the sampled addresses. Cache line bits should not influence the row function
        //For performance optimization, we might increase this to 4096 as usually all addrs wihting
        //a page are in the same row
        const ADDRESS_ALIGNMENT_IN_BYTES: usize = 64;

        //stores the found sets
        let mut same_bank_same_row: Vec<HashSet<MemoryAddress>> = Vec::new();
        //used to check if we have already used an address
        let mut used_physical_addrs: HashSet<u64> = HashSet::new();
        addresses.iter().for_each(|e| {
            used_physical_addrs.insert(e.phys);
        });

        let total_work = addresses.len() * elems_per_set;
        let mut finished_work = 0;

        eprintln!(""); //required for line clearing logic for progress printing

        for base_addr in addresses.iter() {
            let base_addr_rank_bank = evaluate_addr_function(rank_bank_function, base_addr.phys);

            let mut same_as_base = HashSet::new();

            while same_as_base.len() < elems_per_set {
                let candidate_addr = self
                    .memory_source
                    .get_random_address(ADDRESS_ALIGNMENT_IN_BYTES)
                    .with_context(|| "failed to sample memory address")?;
                //sample new addr if not in same bank than base_addr
                if !evaluate_addr_function(rank_bank_function, candidate_addr.phys)
                    .eq(&base_addr_rank_bank)
                {
                    continue;
                }
                if used_physical_addrs.contains(&candidate_addr.phys) {
                    continue;
                }
                used_physical_addrs.insert(candidate_addr.phys);

                //use timing to check if we are in same row
                unsafe {
                    let time = self.timer.time_subsequent_access_from_ram(
                        base_addr.ptr,
                        candidate_addr.ptr,
                        self.measure_rounds,
                    );
                    if time > self.no_conflict_threshold {
                        continue;
                    }
                }

                same_as_base.insert(candidate_addr);
                finished_work += 1;
                console::Term::stderr()
                    .clear_last_lines(1)
                    .expect("clear line failed");
                eprintln!("Found {} out of {} addresses", finished_work, total_work)
            }
            same_bank_same_row.push(same_as_base);
        }

        Ok(same_bank_same_row)
    }

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
    /// * `observations` sets of addresses. We validate if the given function matches these observations
    /// * `rank_bank_function_candidate` candidate for a single bit of the searched function in bitmask form.
    ///
    /// We could also check all pairs inside a set, but a single representative should be fine
    /// The reasoning for this test is to find rank_bank address masks matching all the observations that we made
    fn check_observations_against_function(
        observations: &Vec<HashSet<MemoryAddress>>,
        function_candidate: u64,
    ) -> Result<usize> {
        let mut counter_examples = 0;
        for set in observations.iter() {
            if set.len() == 0 {
                bail!("found set with zero elements, this should never happen");
            }
            let set_representative = set.iter().take(1).collect::<Vec<&MemoryAddress>>()[0];
            let set_representative_fn_value = &parity(set_representative.phys & function_candidate);
            counter_examples += set
                .iter()
                .skip(1)
                .map(|e| parity(e.phys & function_candidate))
                .filter(|v| v != set_representative_fn_value)
                .count();
        }

        return Ok(counter_examples);
    }

    /// Selects all functions from the configured search space that "explains" the observations passed to this function
    /// (the functions are evaluated on the physical address).
    /// The result still contains linear dependent function masks
    /// For the search space, we assume a XOR function. As we want to "explain" the observations the function needs
    /// log2(row_conflict_sets.len()) bits. Each bit of the function may depend on at most `max_function_bits`
    /// which we will represent a AND bit mask. The MSB of each mask is restricted by `msb_index_for_function`.
    /// The first `ignore_low_bits` bits of each address are ignored
    ///
    /// #Arguments
    /// *`observations` : sets of addresses. We search for all functions that would group addresses like in this observation
    ///  * `max_function_bits` : max amount of bits for a single mask
    /// * `msb_index_for_function` highest bit (INDEX) of the physical address that is still considered/included in the masks
    /// * `ignore_low_bits` ignore this many low bits of the physical address
    fn brute_force_matching_functions(
        observations: &Vec<HashSet<MemoryAddress>>,
        min_function_bits: usize,
        max_function_bits: usize,
        msb_index_for_function: usize,
        ignore_low_bits: usize,
    ) -> Result<Vec<u64>> {
        let mut candidates: Vec<u64> = Vec::new();

        //compute "work steps" for progress indications
        let total_work: usize = (1..=max_function_bits)
            .map(|k| num_integer::binomial(msb_index_for_function + 1 - ignore_low_bits, k))
            .sum();
        let bar = indicatif::ProgressBar::new(total_work as u64);

        //iterate over functions with increasing bit complexity
        for current_max_bits in min_function_bits..=max_function_bits {
            //iterate over permutations with current_max_bits set (lower than  msb_index_for_function)
            for mask in XBitPermutationIter::new(
                current_max_bits,
                msb_index_for_function + 1,
                ignore_low_bits,
            ) {
                bar.inc(1);
                let counter_examples =
                    DramAnalyzer::check_observations_against_function(&observations, mask)
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
        let function_candidates = DramAnalyzer::brute_force_matching_functions(
            row_conflict_sets,
            1,
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

        //required for progress printing terminal line clearing logic in all_sets_filled
        eprintln!("");
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
                let addr = match set.iter().nth(0) {
                    None => bail!("found set with zero elements, this should never happen"),
                    Some(v) => v,
                };

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

///MockMemoryTimer evaluates the given rank_bank and row functions and only returns a high
/// timing for same rank/bank but different row addresses. Useful for testing only.
pub struct MockMemoryTimer {
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
        let rank_bank_addr_a = evaluate_addr_function(&self.emulated_rank_bank_function, a as u64);
        let row_addr_a = evaluate_addr_function(&self.emulated_row_function, a as u64);

        let rank_bank_addr_b = evaluate_addr_function(&self.emulated_rank_bank_function, b as u64);
        let row_addr_b = evaluate_addr_function(&self.emulated_row_function, b as u64);

        //if our emulated functions place the two addrs in same rank/bank but a different row return high timing, else low
        return if rank_bank_addr_a.eq(&rank_bank_addr_b) && !row_addr_a.eq(&row_addr_b) {
            self.conflict_threshold + 10
        } else {
            self.conflict_threshold - 50
        };
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use nix::sys::mman::{MapFlags, ProtFlags};

    use crate::memory;
    use crate::memory::MemoryAddress;
    use crate::rank_bank::{evaluate_addr_function, parity, DramAnalyzer};

    use super::super::memory::MemoryBuffer;
    use super::super::MemoryTupleTimer;
    use super::MockMemoryTimer;

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
    fn test_search_same_bank_same_row_sets() {
        const CONFLICT_THRESHOLD: u64 = 330;
        const _NO_CONFLICT_THRESHOLD: u64 = 330;

        struct TestCase {
            description: &'static str,
            want_elems_per_set: usize,
            rank_bank_function: Vec<u64>,
            row_function: Vec<u64>,
            //as we use a mock timing function, these don't need to be valid addrs
            input_addrs: Vec<MemoryAddress>,
        }

        let test_cases = vec![TestCase {
            description: "Simple, 1 bit rank_bank, 2 two bit row",
            want_elems_per_set: 10,
            rank_bank_function: vec![0x100],
            row_function: vec![0x200],
            input_addrs: vec![MemoryAddress {
                phys: 0x100,
                virt: 0x100,
                ptr: std::ptr::null_mut(),
            }],
        }];

        for test_case in test_cases.into_iter() {
            //
            // Setup test environment
            //

            //mock mapper just returns the virtual address as the physical
            let mock_virt_to_phys = Box::new(memory::LinearMockMapper {});

            //alloc real memory buffer but without hugepages
            let memory_source = MemoryBuffer::new(
                1024 * 1024 * 200,
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_PRIVATE | MapFlags::MAP_ANONYMOUS | MapFlags::MAP_POPULATE,
                mock_virt_to_phys,
            )
            .expect("failed to init buffer for testing");
            //Build mock timer that returns timings according to the given rank/bank and row function
            let mock_timer = MockMemoryTimer {
                conflict_threshold: CONFLICT_THRESHOLD,
                emulated_rank_bank_function: test_case.rank_bank_function.clone(),
                emulated_row_function: test_case.row_function.clone(),
            };

            //
            // Test function
            //
            let mut config = super::DramAnalyzer {
                measure_rounds: 1,
                memory_source: Box::new(memory_source),
                timer: Box::new(mock_timer),
                conflict_threshold: CONFLICT_THRESHOLD,
                no_conflict_threshold: CONFLICT_THRESHOLD,
            };

            let got_addrs = config
                .search_same_bank_same_row_sets(
                    &test_case.input_addrs,
                    &test_case.rank_bank_function,
                    test_case.want_elems_per_set,
                )
                .expect("unexpected error in search_same_bank_same_row_sets");

            //Evaluate results
            assert_eq!(
                got_addrs.len(),
                test_case.input_addrs.len(),
                "Wanted results for all {} input addrs, only got results for {}",
                test_case.input_addrs.len(),
                got_addrs.len()
            );
            for (idx, set) in got_addrs.iter().enumerate() {
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
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
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
                no_conflict_threshold: CONFLICT_THRESHOLD,
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

        let got_rank_bank_candidates = super::DramAnalyzer::brute_force_matching_functions(
            &input_sets,
            1,
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
