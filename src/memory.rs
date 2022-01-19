use anyhow::{bail, Context, Result};
use nix::libc::c_void;
use nix::sys::mman::{MapFlags, ProtFlags};
use pagemap;
use rand::Rng;

///MemoryBuffer wraps a raw memory buffer and allows to easily access certain locations
/// of it as well as getting both virtual and physical addresses.
/// It is the central struct for all of the analysis functionality
pub struct MemoryBuffer {
    buf: *mut u8,
    size_in_bytes: usize,
    pagemapper: pagemap::PageMap,
}

///MemoryAddress bundles the virtual and physical address of a memory location, toghether
/// with a pointer to said memory location
pub struct MemoryAddress {
    pub virt: u64,
    pub phys: u64,
    pub ptr: *mut u8,
}

impl MemoryBuffer {
    ///new
    /// #Arguments
    /// * `size_in_bytes` allocates this many bytes
    /// * `prot` protection flags, see mmap manual
    /// * `flags` allocation flags, see mmap manual
    pub fn new(
        size_in_bytes: usize,
        prot: ProtFlags,
        flags: MapFlags,
    ) -> Result<MemoryBuffer, anyhow::Error> {
        //allocate the memory
        let ptr: *mut c_void;
        unsafe {
            let nullptr: *mut nix::libc::c_void = std::ptr::null_mut();
            ptr = nix::sys::mman::mmap(nullptr, size_in_bytes, prot, flags, -1, 0)
                .with_context(|| format!("failed to mmap memory"))?;
        }
        if ptr.is_null() {
            anyhow::bail!("allocation failed");
        }

        //get pagemap for our process
        let pid = std::process::id();
        let our_pagemap =
            pagemap::PageMap::new(pid as u64).with_context(|| "failed to open pagemap")?;

        return Ok(MemoryBuffer {
            buf: ptr.cast(),
            size_in_bytes,
            pagemapper: our_pagemap,
        });
    }

    ///offset returns of MemoryAddress struct for the given offset in the buffer
    /// #Arguments
    /// * `byte_offset` offset in bytes
    pub unsafe fn offset(&mut self, byte_offset: usize) -> Result<MemoryAddress, anyhow::Error> {
        if byte_offset > self.size_in_bytes {
            bail!(
                format! {"out off bounds, requested offset {} >= {}",byte_offset,self.size_in_bytes}
            )
        }
        let ptr_to_start = self.buf.add(byte_offset);

        //
        //get phys address
        //

        //calc virtual address of page containing ptr_to_start
        let vaddr_start_page = ptr_to_start as u64 & !0xFFF;
        let vaddr_end_page = vaddr_start_page + 4095;

        //query pagemap
        let memory_region = pagemap::MemoryRegion::from((vaddr_start_page, vaddr_end_page));
        let entry = self
            .pagemapper
            .pagemap_region(&memory_region)
            .with_context(|| {
                format!(
                    "failed to query pagemap for memory region {:?}",
                    memory_region
                )
            })?;
        if entry.len() != 1 {
            bail!(format! {
            "Got {} pagemap entries for virtual address 0x{:x}, expected exactly one",
            entry.len(),
            ptr_to_start as u64})
        }
        if entry[0].pfn()? == 0 {
            bail!(format! {
                "Got invalid PFN 0 for virtual address 0x{:x}. Are we root?",
                ptr_to_start as u64,
            })
        }

        //assumes page offset of 12 bits
        let pfn = entry[0]
            .pfn()
            .with_context(|| format!("failed to get PFN for pagemap entry {:?}", entry[0]))?;
        let phys_addr = (pfn << 12) | ((ptr_to_start as u64) & 0xFFF);

        Ok(MemoryAddress {
            phys: phys_addr,
            virt: ptr_to_start as u64,
            ptr: ptr_to_start,
        })
    }

    /// Returns distinct random offsets inside the buffer
    /// # Arguments
    /// * `alignment` alignment of the returned offsets in bytes
    /// * `count` amount of offsets to return
    pub fn get_random_offsets(
        &self,
        alignment: usize,
        count: usize,
    ) -> Result<Vec<usize>, anyhow::Error> {
        if alignment == self.size_in_bytes {
            bail!(format!(
                "requested alignment {} is larger than buffer size {}",
                alignment, self.size_in_bytes
            ))
        }
        let last_valid_index = self.size_in_bytes / alignment;

        Ok(
            rand::seq::index::sample(&mut rand::thread_rng(), last_valid_index + 1, count)
                .iter()
                .map(|index| index * alignment)
                .collect(),
        )
    }

    /// get_random_address returns a random address from the buffer with the given alignment
    /// #Arguments
    /// * `alignment` alignment of the returned address in bytes
    pub unsafe fn get_random_address(
        &mut self,
        alignment: usize,
    ) -> Result<MemoryAddress, anyhow::Error> {
        if alignment == self.size_in_bytes {
            bail!(format!(
                "requested alignment {} is larger than buffer size {}",
                alignment, self.size_in_bytes
            ))
        }
        let last_valid_index = self.size_in_bytes / alignment;
        let index = rand::thread_rng().gen_range(0..=last_valid_index);
        let off = index * alignment;

        return self.offset(off);
    }

    ///size_in_bytes returns the size of the memory buffer in bytes
    pub fn size_in_bytes(&self) -> usize {
        self.size_in_bytes
    }
}

impl Drop for MemoryBuffer {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = nix::sys::mman::munmap(self.buf.cast(), self.size_in_bytes) {
                panic!("MyBuffer de-allocation failed with {}", e.to_string());
            }
        }
    }
}
