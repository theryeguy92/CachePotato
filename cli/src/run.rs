//! `cachepotato run`: make sure the model is here, open it, and chat in the terminal.
//!
//! Generated text goes to stdout and nothing else does; prompts, banners and timings go to
//! stderr. That split is what lets the FFI gate compare this binary's stdout byte for byte
//! with tools/qwen3_chat.c.

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::doctor;
use crate::ffi::{self, Model, Options, Stop};
use crate::models::{self, gb};
use crate::pull;

pub struct RunArgs {
    pub model: String,
    pub max_tokens: usize,
    pub budget_mb: Option<u64>,
    pub threads: Option<usize>,
    pub pin_cores: bool,
    pub yes: bool,
}

/// Ask a yes/no question on stderr; Enter means yes. Without a terminal there is nobody to
/// ask, so the answer is no unless --yes was given.
pub fn confirm(question: &str, yes: bool) -> bool {
    if yes {
        return true;
    }
    if !io::stdin().is_terminal() {
        eprintln!("{question} [Y/n] — no terminal to answer on; pass --yes to accept");
        return false;
    }
    eprint!("{question} [Y/n] ");
    io::stderr().flush().ok();
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line).ok();
    matches!(line.trim().to_lowercase().as_str(), "" | "y" | "yes")
}

/// A model name is fetched if missing; a path to a .v4cp file is used as it is.
pub fn locate(model: &str, yes: bool) -> Result<PathBuf> {
    let as_path = Path::new(model);
    if as_path.is_file() {
        return Ok(as_path.to_path_buf());
    }
    let r = models::resolve(model)?;
    let home = models::home()?;
    if !pull::is_installed(&r, &home) {
        let need = pull::remaining_bytes(&r, &home);
        eprintln!("{} is not downloaded yet.", r.display());
        eprintln!("  {}", r.variant.summary);
        eprintln!("  {} from huggingface.co/{} into {}", gb(need), r.spec.repo, r.dir(&home).display());
        if !confirm("Download it now?", yes) {
            bail!("not downloaded. Run `cachepotato pull {}` when ready.", r.display());
        }
        pull::pull(&r, &home)?;
    }
    Ok(r.weights_path(&home))
}

pub fn run(a: RunArgs) -> Result<()> {
    let home = models::home()?;
    let machine = doctor::probe(&home);
    if let Some(b) = doctor::blockers(&machine).first() {
        bail!("{b} (see `cachepotato doctor`)");
    }
    let path = locate(&a.model, a.yes)?;

    eprintln!("loading {} …", path.display());
    let mut model = Model::open(
        &path,
        &Options {
            budget_mb: a.budget_mb,
            threads: a.threads,
            pin_cores: Some(a.pin_cores),
            tokenizer_dir: None,
        },
    )?;
    let s = model.stats();
    let budget = if s.budget_mb < 0 { "no pager".to_string() } else { format!("{:.1} GB pager budget", s.budget_mb as f64 / 1024.0) };
    let pinned = match s.pinned() {
        [] if a.pin_cores => " (not pinned: CPU topology unreadable)".to_string(),
        [] => String::new(),
        cpus => {
            let list = cpus.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
            if cpus.len() < s.threads as usize {
                format!(" ({} pinned to CPUs {list}, one per physical core)", cpus.len())
            } else {
                format!(" pinned to CPUs {list}")
            }
        }
    };
    eprintln!(
        "{} layers, {} experts (top-{}), {}, {} threads{}, {}-token context.",
        s.n_layers, s.n_experts, s.topk, budget, s.threads, pinned, s.max_ctx
    );
    eprintln!("Greedy decoding. /reset starts over, /stats shows the pager, exit or ctrl-D leaves.");
    eprintln!("The first answer is slow while the pager reads weights from disk.");

    let interactive = io::stdin().is_terminal();
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        eprint!("\n>>> ");
        io::stderr().flush().ok();
        let Some(line) = lines.next() else { break };
        let line = line?;
        let msg = line.trim_end_matches(['\r', '\n']);
        match msg.trim() {
            "" => continue,
            "exit" | "quit" | "/exit" | "/bye" => break,
            "/reset" => {
                model.reset();
                eprintln!("(conversation cleared)");
                continue;
            }
            "/stats" => {
                print_stats(&model.stats());
                continue;
            }
            _ => {}
        }
        let mut out = io::stdout().lock();
        let turn = model.chat(msg, a.max_tokens, |bytes| {
            out.write_all(bytes).and_then(|_| out.flush()).is_ok()
        })?;
        drop(out);
        if turn.context_reset != 0 {
            eprintln!("\n(the conversation was full, so it was restarted before this message)");
        }
        let rate = if turn.decode_s > 0.0 { turn.generated_tokens as f64 / turn.decode_s } else { 0.0 };
        let why = match turn.stop {
            Stop::MaxTokens => " — stopped at --max-tokens",
            Stop::Context => " — context full",
            _ => "",
        };
        eprintln!(
            "\n[{} prompt tok in {:.1}s | {} tok at {:.2} tok/s | context {}{}]",
            turn.prompt_tokens, turn.prefill_s, turn.generated_tokens, rate, turn.context_tokens, why
        );
        if !interactive {
            io::stdout().flush().ok();
        }
    }
    Ok(())
}

fn print_stats(s: &ffi::Stats) {
    if s.budget_mb < 0 {
        eprintln!("no pager attached");
        return;
    }
    let acc = s.hits + s.misses;
    eprintln!(
        "pager: {:.1} of {:.1} GB resident | {} accesses, {:.1}% hits | {:.2} GB read from disk | {} evictions",
        s.resident_bytes as f64 / (1u64 << 30) as f64,
        s.budget_mb as f64 / 1024.0,
        acc,
        if acc > 0 { 100.0 * s.hits as f64 / acc as f64 } else { 0.0 },
        s.bytes_fetched as f64 / (1u64 << 30) as f64,
        s.evictions
    );
}
