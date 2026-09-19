// Precomputed zero-copy binary tables for 5-card lookup (~48 KB, fits in L1/L2 cache).
pub const M0: u32 = 0x9e3779b9;
pub const M1: u32 = 0x85ebca6b;
pub const M2: u32 = 0x7feb352d;

#[repr(align(64))]
pub struct AlignedBytes<const N: usize>(pub [u8; N]);

pub static LOOKUP_BYTES: AlignedBytes<49152> = AlignedBytes(*include_bytes!("lookup.bin"));

#[inline(always)]
pub fn flush_table() -> &'static [u16; 8192] {
    let ptr = LOOKUP_BYTES.0.as_ptr();
    debug_assert_eq!(
        ptr as usize % std::mem::align_of::<u16>(),
        0,
        "flush_table pointer must be aligned to u16"
    );
    unsafe { &*ptr.cast::<[u16; 8192]>() }
}

#[inline(always)]
pub fn displacement_table() -> &'static [u16; 8192] {
    let ptr = unsafe { LOOKUP_BYTES.0.as_ptr().add(16384) };
    debug_assert_eq!(
        ptr as usize % std::mem::align_of::<u16>(),
        0,
        "displacement_table pointer must be aligned to u16"
    );
    unsafe { &*ptr.cast::<[u16; 8192]>() }
}

#[inline(always)]
pub fn no_flush_table() -> &'static [u16; 8192] {
    let ptr = unsafe { LOOKUP_BYTES.0.as_ptr().add(32768) };
    debug_assert_eq!(
        ptr as usize % std::mem::align_of::<u16>(),
        0,
        "no_flush_table pointer must be aligned to u16"
    );
    unsafe { &*ptr.cast::<[u16; 8192]>() }
}
