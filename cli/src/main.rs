//! cachepotato — run Mixture-of-Experts models larger than RAM on a laptop.
//!
//! A front-end over the C engine in ../engine (linked by build.rs through
//! engine/api/cachepotato.h). The engine does the arithmetic and is gated; this crate finds,
//! downloads and configures models, and talks to the user.

mod doctor;
mod ffi;
mod models;
mod pull;
mod run;
mod serve;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "cachepotato", version, about = "Run Mixture-of-Experts models larger than RAM on a laptop")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Chat with a model in the terminal, downloading it first if needed
    Run {
        /// Model name (`qwen3-30b-a3b`, `qwen3-30b-a3b:q8`) or a path to a .v4cp file
        model: String,
        /// Maximum tokens per answer
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// Pager budget in MB (default: from available memory)
        #[arg(long)]
        budget: Option<u64>,
        /// Threads (default: one per physical core)
        #[arg(long)]
        threads: Option<usize>,
        /// Do not pin each thread to its own physical core (pinning is ~7% faster, same output)
        #[arg(long)]
        no_pin_cores: bool,
        /// Download without asking
        #[arg(short, long)]
        yes: bool,
    },
    /// Serve a model: browser UI and an OpenAI-compatible API (http://localhost:8080)
    Serve {
        /// Model name (`qwen3-30b-a3b`, `qwen3-30b-a3b:q8`) or a path to a .v4cp file
        model: String,
        /// Address to listen on. Anything but loopback requires --api-key
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Require `Authorization: Bearer <key>` on the API
        #[arg(long, env = "CACHEPOTATO_API_KEY")]
        api_key: Option<String>,
        /// Answer length when a request does not set max_tokens
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        /// Pager budget in MB (default: from available memory)
        #[arg(long)]
        budget: Option<u64>,
        /// Threads (default: one per physical core)
        #[arg(long)]
        threads: Option<usize>,
        /// Do not pin each thread to its own physical core
        #[arg(long)]
        no_pin_cores: bool,
        /// Download without asking
        #[arg(short, long)]
        yes: bool,
    },
    /// Download a model
    Pull {
        /// Model name, e.g. `qwen3-30b-a3b` or `qwen3-30b-a3b:q8`
        model: String,
        /// Do not ask before downloading
        #[arg(short, long)]
        yes: bool,
    },
    /// List the models CachePotato knows and which are downloaded
    List,
    /// Check this machine: CPU, memory, disk, and what it can run
    Doctor,
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Run { model, max_tokens, budget, threads, no_pin_cores, yes } => {
            run::run(run::RunArgs { model, max_tokens, budget_mb: budget, threads, pin_cores: !no_pin_cores, yes })
        }
        Cmd::Serve { model, host, port, api_key, max_tokens, budget, threads, no_pin_cores, yes } => {
            serve::serve(serve::ServeArgs {
                model,
                host,
                port,
                api_key,
                max_tokens,
                budget_mb: budget,
                threads,
                pin_cores: !no_pin_cores,
                yes,
            })
        }
        Cmd::Pull { model, yes } => {
            let r = models::resolve(&model)?;
            let home = models::home()?;
            if pull::is_installed(&r, &home) {
                println!("{} is already downloaded and verified.", r.display());
                return Ok(());
            }
            let need = pull::remaining_bytes(&r, &home);
            eprintln!("{}: {}", r.display(), r.variant.summary);
            let q = format!("Download {} into {}?", models::gb(need), r.dir(&home).display());
            if !run::confirm(&q, yes) {
                return Ok(());
            }
            pull::pull(&r, &home)?;
            println!("{} is ready: cachepotato run {}", r.display(), r.display());
            Ok(())
        }
        Cmd::List => {
            let home = models::home()?;
            for spec in models::MODELS {
                println!("{} — {}", spec.name, spec.description);
                for v in spec.variants {
                    let r = models::resolve(&format!("{}:{}", spec.name, v.tag))?;
                    let mark = if pull::is_installed(&r, &home) { "installed" } else { "" };
                    println!("  {:<22} {:>8}  {:<10} {}", r.display(), models::gb(v.bytes), mark, v.summary);
                }
            }
            Ok(())
        }
        Cmd::Doctor => doctor::run(),
    }
}
