//! `cachepotato doctor`: what this machine is, and what it can run.
//!
//! The engine's speed is bytes divided by bandwidth on both the disk and the RAM side, so the
//! numbers that matter are free memory (the pager budget), the disk type, and the cores. Every
//! check here is something a user would otherwise discover as "it is broken".

use std::fs;
use std::path::Path;

use anyhow::Result;

use crate::ffi;
use crate::models::{self, gb, MODELS};
use crate::pull;

pub struct Machine {
    pub cpu: String,
    pub avx2_fma: bool,
    pub cores: usize,
    pub mem_total_mb: Option<u64>,
    pub mem_available_mb: Option<u64>,
    pub disk_free: Option<u64>,
    pub disk_kind: String,
}

fn meminfo(key: &str) -> Option<u64> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    text.lines()
        .find_map(|l| l.strip_prefix(key)?.trim().strip_suffix("kB")?.trim().parse::<u64>().ok())
        .map(|kb| kb / 1024)
}

fn cpu_name() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| t.lines().find_map(|l| l.strip_prefix("model name")?.split_once(':').map(|(_, v)| v.trim().to_string())))
        .unwrap_or_else(|| std::env::consts::ARCH.to_string())
}

#[cfg(target_arch = "x86_64")]
fn has_avx2_fma() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}
#[cfg(not(target_arch = "x86_64"))]
fn has_avx2_fma() -> bool {
    false
}

/// NVMe / SATA SSD / HDD for the filesystem holding `path`, best effort: the mount's source
/// device from /proc/self/mounts, then that block device's queue/rotational in sysfs.
fn disk_kind(path: &Path) -> String {
    let mut p = path.to_path_buf();
    while !p.exists() {
        match p.parent() {
            Some(q) => p = q.to_path_buf(),
            None => return "unknown".into(),
        }
    }
    let p = fs::canonicalize(&p).unwrap_or(p);
    let Ok(mounts) = fs::read_to_string("/proc/self/mounts") else { return "unknown".into() };
    let dev = mounts
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let (src, mnt) = (it.next()?, it.next()?);
            p.starts_with(mnt).then(|| (mnt.len(), src.to_string()))
        })
        .max_by_key(|(n, _)| *n)
        .map(|(_, s)| s);
    let Some(dev) = dev.and_then(|d| d.strip_prefix("/dev/").map(str::to_string)) else {
        return "unknown".into();
    };
    let sys = Path::new("/sys/class/block").join(&dev);
    let sys = fs::canonicalize(&sys).unwrap_or(sys);
    // a partition's queue/ lives on its parent disk
    let rot = fs::read_to_string(sys.join("queue/rotational"))
        .or_else(|_| fs::read_to_string(sys.join("../queue/rotational")));
    let disk = sys.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or(dev);
    match rot.as_deref().map(str::trim) {
        Ok("1") => format!("HDD ({disk})"),
        Ok("0") if disk.starts_with("nvme") => format!("NVMe SSD ({disk})"),
        Ok("0") => format!("SSD ({disk})"),
        _ => format!("unknown ({disk})"),
    }
}

pub fn probe(home: &Path) -> Machine {
    Machine {
        cpu: cpu_name(),
        avx2_fma: has_avx2_fma(),
        cores: ffi::physical_cores(),
        mem_total_mb: meminfo("MemTotal:"),
        mem_available_mb: ffi::mem_available_mb(),
        disk_free: pull::free_space(home),
        disk_kind: disk_kind(home),
    }
}

/// Problems that stop the engine from running at all.
pub fn blockers(m: &Machine) -> Vec<String> {
    let mut v = vec![];
    if !cfg!(target_os = "linux") {
        v.push("CachePotato currently runs on Linux only".into());
    }
    if !m.avx2_fma {
        v.push("this CPU lacks AVX2/FMA, which the engine's kernels require".into());
    }
    v
}

pub fn run() -> Result<()> {
    let home = models::home()?;
    let m = probe(&home);
    let mb = |v: Option<u64>| v.map_or("unknown".to_string(), |x| format!("{:.1} GB", x as f64 / 1024.0));

    println!("CachePotato {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("  CPU        {}", m.cpu);
    println!("  AVX2/FMA   {}", if m.avx2_fma { "yes" } else { "NO" });
    println!("  cores      {} physical (threads used by default)", m.cores);
    println!("  memory     {} total, {} available now", mb(m.mem_total_mb), mb(m.mem_available_mb));
    println!("  models in  {}", home.display());
    println!("  disk       {}, {} free", m.disk_kind, m.disk_free.map_or("unknown".into(), gb));
    println!();

    let blockers = blockers(&m);
    for b in &blockers {
        println!("  ✗ {b}");
    }
    if m.disk_kind.starts_with("HDD") {
        println!("  ! models are on a spinning disk: paging will be very slow. Set CACHEPOTATO_HOME to an SSD.");
    }

    println!("Models:");
    for spec in MODELS {
        for v in spec.variants {
            let r = models::resolve(&format!("{}:{}", spec.name, v.tag))?;
            let installed = pull::is_installed(&r, &home);
            let budget = ffi::auto_budget_mb(v.bytes);
            let fits_disk = installed || m.disk_free.is_some_and(|f| f > pull::remaining_bytes(&r, &home) + 1_000_000_000);
            let verdict = if !blockers.is_empty() {
                "cannot run here".to_string()
            } else if budget.is_none() {
                "not enough free memory right now (needs ~4 GB available)".to_string()
            } else if !fits_disk {
                format!("not enough disk space ({} needed)", gb(v.bytes))
            } else {
                let b = budget.unwrap();
                let pages = if b * 1024 * 1024 < v.bytes { "pages from disk" } else { "fits in RAM" };
                format!("ok, {:.1} GB pager budget ({pages})", b as f64 / 1024.0)
            };
            println!(
                "  {:<20} {:>8}  {:<10} {}",
                r.display(),
                gb(v.bytes),
                if installed { "installed" } else { "" },
                verdict
            );
        }
    }
    println!();
    println!("Speed is set by disk and memory bandwidth; reference: 1.2 tok/s for {}:q4 on a", MODELS[0].name);
    println!("ThinkPad T480 (16 GB, SATA SSD). An NVMe SSD or more RAM is faster.");
    Ok(())
}
