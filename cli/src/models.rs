//! The models the CLI knows how to fetch, and where they live on disk.
//!
//! A model is `name[:tag]`, e.g. `qwen3-30b-a3b` (the default tag) or `qwen3-30b-a3b:q8`.
//! Files are stored under `<home>/models/<name>/` with the same relative paths as in the
//! Hugging Face repo, so the engine finds the tokenizer in `tokenizer/` next to the weights.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

pub struct Variant {
    pub tag: &'static str,
    pub file: &'static str,
    pub bytes: u64,
    pub summary: &'static str,
}

pub struct ModelSpec {
    pub name: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub description: &'static str,
    pub variants: &'static [Variant],
    /// Files every variant needs, besides its weights. SHA256SUMS is fetched separately.
    pub shared_files: &'static [&'static str],
}

pub const CHECKSUMS: &str = "SHA256SUMS";

pub static MODELS: &[ModelSpec] = &[ModelSpec {
    name: "qwen3-30b-a3b",
    repo: "ryanlevey/cachepotato-qwen3-30b-a3b",
    revision: "main",
    description: "Qwen3-30B-A3B (MoE, 3.3B active), Apache-2.0",
    variants: &[
        Variant {
            tag: "q4",
            file: "qwen3-30b-a3b-q4attn.v4cp",
            bytes: 17_530_739_904,
            summary: "4-bit experts + attention; fastest (1.2 tok/s on a 16 GB T480), passed the quality gate",
        },
        Variant {
            tag: "q8",
            file: "qwen3-30b-a3b-q8.v4cp",
            bytes: 32_479_239_360,
            summary: "8-bit; the exact reference (0.8 tok/s on a 16 GB T480)",
        },
    ],
    shared_files: &["tokenizer/qwen2.5-tokenizer.bin", "tokenizer/qwen2.5-unicode.bin"],
}];

pub struct Resolved {
    pub spec: &'static ModelSpec,
    pub variant: &'static Variant,
}

impl Resolved {
    pub fn display(&self) -> String {
        format!("{}:{}", self.spec.name, self.variant.tag)
    }
    pub fn dir(&self, home: &Path) -> PathBuf {
        home.join("models").join(self.spec.name)
    }
    pub fn weights_path(&self, home: &Path) -> PathBuf {
        self.dir(home).join(self.variant.file)
    }
    pub fn url(&self, file: &str) -> String {
        format!("https://huggingface.co/{}/resolve/{}/{}", self.spec.repo, self.spec.revision, file)
    }
    /// Weights + shared files, in download order (small files first).
    pub fn files(&self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = self.spec.shared_files.to_vec();
        v.push(self.variant.file);
        v
    }
}

pub fn resolve(name: &str) -> Result<Resolved> {
    let (base, tag) = match name.split_once(':') {
        Some((b, t)) => (b, Some(t)),
        None => (name, None),
    };
    let Some(spec) = MODELS.iter().find(|m| m.name == base) else {
        let known: Vec<_> = MODELS.iter().map(|m| m.name).collect();
        bail!("unknown model '{base}'. Known models: {}", known.join(", "));
    };
    let variant = match tag {
        None => &spec.variants[0],
        Some(t) => match spec.variants.iter().find(|v| v.tag == t) {
            Some(v) => v,
            None => {
                let tags: Vec<_> = spec.variants.iter().map(|v| v.tag).collect();
                bail!("'{base}' has no tag '{t}'. Tags: {}", tags.join(", "));
            }
        },
    };
    Ok(Resolved { spec, variant })
}

/// `$CACHEPOTATO_HOME`, else `$XDG_DATA_HOME/cachepotato`, else `~/.local/share/cachepotato`.
pub fn home() -> Result<PathBuf> {
    if let Some(h) = std::env::var_os("CACHEPOTATO_HOME") {
        return Ok(PathBuf::from(h));
    }
    if let Some(x) = std::env::var_os("XDG_DATA_HOME").filter(|x| !x.is_empty()) {
        return Ok(PathBuf::from(x).join("cachepotato"));
    }
    match std::env::var_os("HOME") {
        Some(h) => Ok(PathBuf::from(h).join(".local/share/cachepotato")),
        None => bail!("cannot find a home directory; set CACHEPOTATO_HOME"),
    }
}

pub fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}
