//! Compiles the C engine into the binary. The engine is the gated part of the project and stays
//! C; this crate only links it through engine/api/cachepotato.h.
//!
//! CACHEPOTATO_MARCH picks the instruction set: `native` (default) for a local build, and
//! `x86-64-v3` (AVX2 + FMA, what the kernels need) for a release binary that must run on other
//! machines. Bit-identity is per machine either way, so neither choice changes a claim.

use std::path::PathBuf;

const ENGINE_SOURCES: &[&str] = &[
    "api/cachepotato.c",
    "blocks/qwen3_moe.c",
    "kernels/kernels.c",
    "kernels/parallel.c",
    "pager/pager.c",
    "core/weights.c",
    "core/weights_v4.c",
    "tokenizers/tokenizer.c",
    "tokenizers/tokenizer_qwen.c",
];

fn main() {
    let engine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine");
    let march = std::env::var("CACHEPOTATO_MARCH").unwrap_or_else(|_| "native".into());

    let mut build = cc::Build::new();
    build
        .include(&engine)
        .std("c11")
        .opt_level(3)
        .flag(format!("-march={march}"))
        .warnings(false);
    for src in ENGINE_SOURCES {
        build.file(engine.join(src));
    }
    build.compile("cachepotato_engine");

    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rerun-if-env-changed=CACHEPOTATO_MARCH");
    println!("cargo:rerun-if-changed={}", engine.display());
}
