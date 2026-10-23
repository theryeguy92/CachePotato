//! Download a model from Hugging Face: resumable, verified against the repo's SHA256SUMS.
//!
//! Each file is downloaded to `<file>.part` and renamed only after its SHA-256 matches. A
//! verified file gets a `<file>.sha256` marker holding its hash, so later runs can trust a
//! 17 GB file without re-reading it, and a marker that disagrees with SHA256SUMS (the repo
//! was updated) sends the file back through download.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use reqwest::header::RANGE;
use reqwest::StatusCode;
use sha2::{Digest, Sha256};

use crate::models::{gb, Resolved, CHECKSUMS};

const BUF: usize = 1 << 20;

fn client() -> Result<Client> {
    Ok(Client::builder()
        .user_agent(concat!("cachepotato/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        // per connect/read/write operation in the blocking client, so it catches a stalled
        // connection without capping a download that takes hours
        .timeout(Duration::from_secs(120))
        .build()?)
}

fn parse_checksums(text: &str) -> HashMap<String, String> {
    text.lines()
        .filter_map(|l| {
            let (hash, name) = l.split_once(char::is_whitespace)?;
            Some((name.trim().trim_start_matches('*').to_string(), hash.to_lowercase()))
        })
        .collect()
}

fn marker(path: &Path) -> PathBuf {
    let mut m = path.as_os_str().to_owned();
    m.push(".sha256");
    PathBuf::from(m)
}

fn part(path: &Path) -> PathBuf {
    let mut m = path.as_os_str().to_owned();
    m.push(".part");
    PathBuf::from(m)
}

/// The local SHA256SUMS, if the model was ever pulled.
fn local_checksums(dir: &Path) -> Option<HashMap<String, String>> {
    fs::read_to_string(dir.join(CHECKSUMS)).ok().map(|t| parse_checksums(&t))
}

/// True when every file of this variant is present and its marker matches SHA256SUMS.
pub fn is_installed(r: &Resolved, home: &Path) -> bool {
    let dir = r.dir(home);
    let Some(sums) = local_checksums(&dir) else { return false };
    r.files().iter().all(|f| {
        let p = dir.join(f);
        let ok_marker = fs::read_to_string(marker(&p)).map(|m| Some(m.trim()) == sums.get(*f).map(|s| s.as_str()));
        p.is_file() && matches!(ok_marker, Ok(true))
    })
}

/// Bytes still to download for this variant (what is already verified counts as done).
pub fn remaining_bytes(r: &Resolved, home: &Path) -> u64 {
    if is_installed(r, home) {
        return 0;
    }
    let done = fs::metadata(part(&r.weights_path(home))).map(|m| m.len()).unwrap_or(0);
    r.variant.bytes.saturating_sub(done)
}

pub fn free_space(dir: &Path) -> Option<u64> {
    let mut p = dir.to_path_buf();
    while !p.exists() {
        p = p.parent()?.to_path_buf();
    }
    let c = std::ffi::CString::new(p.as_os_str().as_encoded_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    (unsafe { libc::statvfs(c.as_ptr(), &mut st) } == 0).then(|| st.f_bavail as u64 * st.f_frsize as u64)
}

pub fn pull(r: &Resolved, home: &Path) -> Result<()> {
    let dir = r.dir(home);
    fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;

    let need = remaining_bytes(r, home);
    if let Some(free) = free_space(&dir) {
        let margin = 1_000_000_000;
        if need + margin > free {
            bail!(
                "not enough disk space in {}: {} needed (+1 GB margin), {} free. \
                 Free some space or set CACHEPOTATO_HOME to another disk.",
                dir.display(),
                gb(need),
                gb(free)
            );
        }
    }

    let http = client()?;
    let sums_text = http
        .get(r.url(CHECKSUMS))
        .send()
        .and_then(|x| x.error_for_status())
        .and_then(|x| x.text())
        .with_context(|| format!("cannot fetch {} from {}", CHECKSUMS, r.spec.repo))?;
    let sums = parse_checksums(&sums_text);

    for f in r.files() {
        let Some(expected) = sums.get(f) else {
            bail!("{} in {} has no entry for {f}", CHECKSUMS, r.spec.repo);
        };
        let dest = dir.join(f);
        if dest.is_file() && fs::read_to_string(marker(&dest)).map(|m| m.trim() == expected).unwrap_or(false) {
            continue;
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        download(&http, &r.url(f), &dest, expected, f)?;
    }
    // written last: its presence means "this directory was pulled from these sums"
    fs::write(dir.join(CHECKSUMS), sums_text)?;
    Ok(())
}

fn bar(len: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template("{msg:28!} [{bar:30}] {bytes}/{total_bytes} {binary_bytes_per_sec} eta {eta}")
            .unwrap()
            .progress_chars("=> "),
    );
    pb.set_message(msg.to_string());
    pb
}

fn download(http: &Client, url: &str, dest: &Path, expected: &str, label: &str) -> Result<()> {
    let tmp = part(dest);
    let have = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
    let mut hasher = Sha256::new();

    // a resumed download re-hashes what is already on disk, so the final check covers all of it
    if have > 0 {
        let pb = bar(have, &format!("checking {label}"));
        let mut f = File::open(&tmp)?;
        let mut buf = vec![0u8; BUF];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            pb.inc(n as u64);
        }
        pb.finish_and_clear();
    }

    let mut req = http.get(url);
    if have > 0 {
        req = req.header(RANGE, format!("bytes={have}-"));
    }
    let mut resp = req.send().with_context(|| format!("cannot download {url}"))?;
    // 416: the part file already holds every byte, so there is nothing to read
    let complete = have > 0 && resp.status() == StatusCode::RANGE_NOT_SATISFIABLE;
    if !complete {
        resp = resp.error_for_status().with_context(|| format!("cannot download {url}"))?;
    }
    let total = have + if complete { 0 } else { resp.content_length().unwrap_or(0) };
    let pb = bar(total, label);
    pb.set_position(have);

    // nothing is opened for writing when the part file is already complete
    if !complete {
        let mut out = if resp.status() == StatusCode::PARTIAL_CONTENT {
            OpenOptions::new().append(true).open(&tmp)?
        } else {
            // the server ignored the range: start over
            hasher = Sha256::new();
            pb.set_length(resp.content_length().unwrap_or(0));
            pb.set_position(0);
            File::create(&tmp)?
        };
        let mut buf = vec![0u8; BUF];
        loop {
            let n = match resp.read(&mut buf) {
                Ok(n) => n,
                Err(e) => {
                    pb.abandon();
                    bail!("download of {label} interrupted ({e}). Run the same command again to resume.");
                }
            };
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            pb.inc(n as u64);
        }
        out.sync_all()?;
    }
    pb.finish_and_clear();

    let got = format!("{:x}", hasher.finalize());
    if got != expected {
        fs::remove_file(&tmp).ok();
        bail!("{label}: checksum mismatch (expected {expected}, got {got}); the partial file was deleted, run again to retry");
    }
    fs::rename(&tmp, dest)?;
    fs::write(marker(dest), format!("{got}\n"))?;
    eprintln!("  {label}: verified");
    Ok(())
}
