use anyhow::{bail, Context, Result};
use clap::Parser;
use itertools::Itertools;
use nix::sys::mman::{MapFlags, ProtFlags};
use no_drama::memory::LinuxPageMap;
use no_drama::rank_bank::DramAnalyzer;
use no_drama::{construct_timer_from_cli_arg, memory, DefaultMemoryTupleTimer};
use petgraph::dot::{Config, Dot};
use petgraph::{Graph, Undirected};
use serde::Serialize;
use std::collections::HashSet;

#[derive(Parser, Debug, Serialize)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    ///Size of the buffer from which addresses are sampled. If hugepages are used (default) this should be a multiple of a GiB.
    #[clap(long, default_value = "5120")]
    buffer_size_in_mb: usize,

    ///Max amount of physical address bits a rank+bank function bit may depend on  
    #[clap(long, default_value = "30")]
    max_phy_addr_bit: usize,

    ///Amount of least significant bits that are ignored when searching for rank+bank functions
    #[clap(long, default_value = "6")]
    ignore_low_bits: usize,

    ///Access time above which we consider an access to be a row conflict
    #[clap(long)]
    conflict_threshold: u64,

    ///Access time below which we consider an access to be no row conflict (Must be <= conflict_threshold, allows to define a "deadzone")
    #[clap(long)]
    no_conflict_threshold: u64,

    ///Average a single time measurement over this many accesses
    #[clap(short, long, default_value = "4000")]
    rounds_per_measurement: usize,

    ///Select timer to time ram accesses.
    #[clap(long, default_value = "rdtsc")]
    timing_source: String,
}

impl CliArgs {
    fn validate(&self) -> Result<()> {
        if self.no_conflict_threshold > self.conflict_threshold {
            bail!(format!(
                "no_conflict_threshold ({}) must be <= conflict_threshold ({})",
                self.no_conflict_threshold, self.conflict_threshold
            ));
        }

        Ok(())
    }
}

fn into_sorted_vec<I, T>(vals: I) -> Vec<T>
where
    I: Iterator<Item = T>,
    T: Copy + Ord,
{
    let mut v: Vec<T> = Vec::new();
    for e in vals {
        v.push(e);
    }
    v.sort();
    v
}

fn run(analyzer: &mut DramAnalyzer, args: &CliArgs) -> Result<()> {
    const TASK_COUNT: usize = 4;

    //set with all physical address bits (indices)
    let set_v: HashSet<u64> = HashSet::from_iter(
        ((args.ignore_low_bits as u64)..(args.max_phy_addr_bit as u64)).into_iter(),
    );
    eprintln!(
        "Considered physical address bit indices: {:?}",
        into_sorted_vec(set_v.iter())
    );

    //Step 1 : Detect exclusive row bits (not overlapping with rank/bank/channel) bits
    eprintln!("Task [{}/{}]: find exclusive row bits", 1, TASK_COUNT);
    let mut set_r: HashSet<u64> = HashSet::new();
    for &bit_idx in set_v.iter() {
        let flip_mask = 1_u64 << bit_idx;
        if analyzer.latency_xiao_general(&flip_mask).with_context(|| {
            format!(
                "error finding exclusive row bits: latency_xiao_general for bit_idx {} failed",
                bit_idx
            )
        })? {
            set_r.insert(bit_idx);
        }
    }

    //Step 2 : Detect column bits
    eprintln!("Task [{}/{}]: find exclusive col bits", 2, TASK_COUNT);
    let mut set_c = HashSet::new();

    for r_bit_idx in set_r.iter() {
        for non_r_bit_idx in set_v.difference(&set_r) {
            let flip_mask = (1_u64 << r_bit_idx) | (1_u64 << non_r_bit_idx);
            if analyzer.latency_xiao_general(&flip_mask).with_context(|| {
                format!(
                    "error finding column bits: latency_xiao_general for ({},{}) failed",
                    r_bit_idx, non_r_bit_idx
                )
            })? {
                set_c.insert(non_r_bit_idx);
            }
        }
    }

    //Step 3: 2 XOR BANK + Complex Row bits
    eprintln!("Task [{}/{}]: build G1 graph", 3, TASK_COUNT);
    let mut search_space: HashSet<u64> = set_v.clone();
    search_space.retain(|v| !set_r.contains(v) && !set_c.contains(v));

    let mut graph_g1: Graph<u64, (), Undirected> = Graph::new_undirected();
    for &bit_idx in search_space.iter() {
        graph_g1.add_node(bit_idx);
    }

    for (n1, n2) in graph_g1.node_indices().tuple_combinations::<(_, _)>() {
        let b_i = match graph_g1.node_weight(n1) {
            None => bail!("Node without weight"),
            Some(v) => v,
        };
        let b_j = match graph_g1.node_weight(n2) {
            None => bail!("Node without weight"),
            Some(v) => v,
        };

        let flip_mask = (1_u64 << b_i) | (1_u64 << b_j);
        //eprintln!("Paddr tuple ({},{})", b_i, b_j);
        if analyzer.latency_xiao_general(&flip_mask).with_context(|| {
            format!("error finding edges in G1 graph on tuple ({},{})", b_i, b_j)
        })? {
            graph_g1.add_edge(n1, n2, ());
        }
    }

    //Step Double used bank bits? + Complex Row bits
    eprintln!("Task [{}/{}]: build G2 graph", 4, TASK_COUNT);
    let mut graph_g2: Graph<u64, (), Undirected> = Graph::new_undirected();
    for &bit_idx in search_space.iter() {
        graph_g2.add_node(bit_idx);
    }

    for (n1, n2, n3) in graph_g2.node_indices().tuple_combinations::<(_, _, _)>() {
        let b_i = match graph_g2.node_weight(n1) {
            None => bail!("Node without weight"),
            Some(v) => v,
        };
        let b_j = match graph_g2.node_weight(n2) {
            None => bail!("Node without weight"),
            Some(v) => v,
        };
        let b_m = match graph_g2.node_weight(n3) {
            None => bail!("Node without weight"),
            Some(v) => v,
        };

        let flip_mask = (1_u64 << b_i) | (1_u64 << b_j) | (1_u64 << b_m);
        if analyzer.latency_xiao_general(&flip_mask).with_context(|| {
            format!("error finding edges in G1 graph on tuple ({},{})", b_i, b_j)
        })? {
            graph_g2.add_edge(n1, n2, ());
            graph_g2.add_edge(n1, n3, ());
            graph_g2.add_edge(n2, n3, ());
        }
    }

    //
    // Export Data
    //

    let mut r_as_vec: Vec<u64> = Vec::from_iter(set_r.iter().copied());
    r_as_vec.sort();
    println!("Simple row bits: {:?}", r_as_vec);

    let mut c_as_vec: Vec<&u64> = Vec::from_iter(set_c.iter().copied());
    c_as_vec.sort();
    println!("Column bits : {:?}", c_as_vec);

    println!(
        "Graphviz for G1\n{:?}",
        Dot::with_config(&graph_g1, &[Config::EdgeNoLabel])
    );

    println!(
        "Graphviz for G2\n{:?}",
        Dot::with_config(&graph_g2, &[Config::EdgeNoLabel])
    );

    Ok(())
}

fn main() -> Result<()> {
    let args: CliArgs = CliArgs::parse();

    if let Err(e) = args.validate() {
        eprintln!("Invalid Config!");
        bail!(e);
    }

    eprintln!("Parsed config");
    //
    //Setup
    //

    let alloc_flags = MapFlags::MAP_PRIVATE
        | MapFlags::MAP_ANONYMOUS
        | MapFlags::MAP_HUGETLB
        | MapFlags::MAP_POPULATE;

    let virt_to_phys =
        Box::new(LinuxPageMap::new().with_context(|| "failed to instantiate virt_to_phys mapper")?);

    let buf = Box::new(
        memory::MemoryBuffer::new(
            args.buffer_size_in_mb * 1024 * 1024,
            ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            alloc_flags,
            virt_to_phys,
        )
        .with_context(|| "Failed to create buffer")?,
    );

    let timer = construct_timer_from_cli_arg(&args.timing_source)
        .with_context(|| "failed to contruct timer")?;
    eprintln!("Conflict above {}", args.conflict_threshold);
    eprintln!("No conflict below {}", args.no_conflict_threshold);
    let mut analyzer = DramAnalyzer::new(
        buf,
        timer,
        args.rounds_per_measurement,
        args.conflict_threshold,
        args.no_conflict_threshold,
    );

    eprintln!("Build test environment");
    //
    // Program Logic
    //

    run(&mut analyzer, &args)
}
/*
#[cfg(test)]
mod test {
    use no_drama::rank_bank::MockMemoryTimer;

    #[test]
    fn full_test() {
        const IGNORED_LSBS: usize = 6;
        const PADDR_LEN: usize = IGNORED_LSBS + 3;
        const CONFLICT_THRESH = 400;

        let bank_fn

        let mock_timer = MockMemoryTimer{}

    }
}
*/
