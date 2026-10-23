//! Bindings to engine/api/cachepotato.h, and a safe wrapper over them.
//!
//! The structs mirror the header field for field; a change on either side must be made on both.

use std::ffi::{c_char, c_int, c_long, c_longlong, c_void, CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

use anyhow::{anyhow, bail, Result};

pub const BUDGET_AUTO: c_long = 0;

#[repr(C)]
struct CpOptions {
    budget_mb: c_long,
    threads: c_int,
    pin_cores: c_int,
    max_ctx: c_int,
    tokenizer_dir: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // constructed by C
pub enum Stop {
    Eos = 0,
    MaxTokens = 1,
    Context = 2,
    Caller = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Turn {
    pub prompt_tokens: c_int,
    pub context_tokens: c_int,
    pub generated_tokens: c_int,
    pub context_reset: c_int,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub stop: Stop,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub n_layers: c_int,
    pub d_model: c_int,
    pub n_experts: c_int,
    pub topk: c_int,
    pub vocab: c_int,
    pub max_ctx: c_int,
    pub threads: c_int,
    pub budget_mb: c_long,
    pub file_bytes: c_longlong,
    pub hits: c_long,
    pub misses: c_long,
    pub evictions: c_long,
    pub bytes_fetched: c_longlong,
    pub resident_bytes: c_longlong,
    pub n_pinned: c_int,
    pub pinned_cpus: [c_int; 64],
}

impl Stats {
    /// The logical CPUs the engine's threads are pinned to, thread 0 first; empty if unpinned.
    pub fn pinned(&self) -> &[c_int] {
        &self.pinned_cpus[..(self.n_pinned.clamp(0, 64) as usize)]
    }
}

#[repr(C)]
struct CpModel {
    _private: [u8; 0],
}

type TokenFn = unsafe extern "C" fn(*const c_char, usize, *mut c_void) -> c_int;
type ProgressFn = unsafe extern "C" fn(c_int, c_int, c_int, *mut c_void);

extern "C" {
    fn cp_options_default(opt: *mut CpOptions);
    fn cp_open(path: *const c_char, opt: *const CpOptions, err: *mut c_char, errlen: usize) -> *mut CpModel;
    fn cp_close(m: *mut CpModel);
    fn cp_chat(
        m: *mut CpModel,
        msg: *const c_char,
        max_new: c_int,
        on_token: Option<TokenFn>,
        user: *mut c_void,
        turn: *mut Turn,
    ) -> c_int;
    fn cp_reset(m: *mut CpModel);
    fn cp_prefill_text(m: *mut CpModel, text: *const c_char) -> c_int;
    fn cp_context_tokens(m: *const CpModel) -> c_int;
    fn cp_set_progress(m: *mut CpModel, f: Option<ProgressFn>, user: *mut c_void);
    fn cp_get_stats(m: *const CpModel, out: *mut Stats);
    fn cp_physical_cores() -> c_int;
    fn cp_mem_available_mb() -> c_long;
    fn cp_auto_budget_mb(model_file_bytes: c_longlong) -> c_long;
}

pub fn physical_cores() -> usize {
    unsafe { cp_physical_cores() }.max(1) as usize
}

/// MemAvailable in MB, if /proc/meminfo is readable.
pub fn mem_available_mb() -> Option<u64> {
    let v = unsafe { cp_mem_available_mb() };
    (v >= 0).then_some(v as u64)
}

/// The budget `cp_open` would choose for a file this size, or None when memory is too tight.
pub fn auto_budget_mb(model_file_bytes: u64) -> Option<u64> {
    let v = unsafe { cp_auto_budget_mb(model_file_bytes as c_longlong) };
    (v >= 0).then_some(v as u64)
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    /// None = automatic from MemAvailable.
    pub budget_mb: Option<u64>,
    /// None = one per physical core.
    pub threads: Option<usize>,
    /// None = CP_PIN_CORES from the environment. Pinning is per physical core and never
    /// changes the output.
    pub pin_cores: Option<bool>,
    pub tokenizer_dir: Option<std::path::PathBuf>,
}

/// Where a forward pass is: `layer` of `n_layers` done, over `tokens` tokens (> 1 = a prompt
/// being prefilled). `stats` is filled during a prefill and on the last layer of a generated
/// token, and is None otherwise (reading it walks every region, so not on every layer).
pub struct Progress {
    pub layer: usize,
    pub n_layers: usize,
    pub tokens: usize,
    pub stats: Option<Stats>,
}

struct ProgressHook {
    model: *const CpModel,
    f: Box<dyn FnMut(Progress)>,
}

/// An open model. One per process: the engine's pager is process-global. Not Send: the
/// engine pins the thread that opened it, so it is used where it was opened.
pub struct Model {
    ptr: NonNull<CpModel>,
    progress: Option<Box<ProgressHook>>,
}

impl Model {
    pub fn open(path: &Path, o: &Options) -> Result<Model> {
        let cpath = CString::new(path.as_os_str().as_encoded_bytes())?;
        let tdir = o
            .tokenizer_dir
            .as_ref()
            .map(|p| CString::new(p.as_os_str().as_encoded_bytes()))
            .transpose()?;
        let mut opt = std::mem::MaybeUninit::<CpOptions>::uninit();
        let mut opt = unsafe {
            cp_options_default(opt.as_mut_ptr());
            opt.assume_init()
        };
        opt.budget_mb = o.budget_mb.map_or(BUDGET_AUTO, |b| b as c_long);
        if let Some(t) = o.threads {
            opt.threads = t as c_int;
        }
        if let Some(p) = o.pin_cores {
            opt.pin_cores = p as c_int;
        }
        if let Some(t) = &tdir {
            opt.tokenizer_dir = t.as_ptr();
        }
        let mut err = [0 as c_char; 512];
        let p = unsafe { cp_open(cpath.as_ptr(), &opt, err.as_mut_ptr(), err.len()) };
        match NonNull::new(p) {
            Some(ptr) => Ok(Model { ptr, progress: None }),
            None => {
                let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
                Err(anyhow!(msg))
            }
        }
    }

    /// One user turn. `on_token` gets each token's raw bytes (possibly a partial UTF-8
    /// sequence) and returns `false` to stop after that token.
    pub fn chat<F: FnMut(&[u8]) -> bool>(&mut self, msg: &str, max_new: usize, mut on_token: F) -> Result<Turn> {
        unsafe extern "C" fn trampoline<F: FnMut(&[u8]) -> bool>(
            bytes: *const c_char,
            len: usize,
            user: *mut c_void,
        ) -> c_int {
            let f = &mut *(user as *mut F);
            let s = std::slice::from_raw_parts(bytes as *const u8, len);
            // a panic must not unwind into C; treat it as "stop"
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(s))) {
                Ok(true) => 0,
                _ => 1,
            }
        }
        let cmsg = CString::new(msg).map_err(|_| anyhow!("message contains a NUL byte"))?;
        let mut turn = std::mem::MaybeUninit::<Turn>::zeroed();
        let rc = unsafe {
            cp_chat(
                self.ptr.as_ptr(),
                cmsg.as_ptr(),
                max_new.min(c_int::MAX as usize) as c_int,
                Some(trampoline::<F>),
                &mut on_token as *mut F as *mut c_void,
                turn.as_mut_ptr(),
            )
        };
        match rc {
            0 => Ok(unsafe { turn.assume_init() }),
            -2 => bail!("the message could not be tokenized"),
            _ => bail!("the forward pass failed (engine error {rc})"),
        }
    }

    pub fn reset(&mut self) {
        unsafe { cp_reset(self.ptr.as_ptr()) }
    }

    /// Append raw text (ChatML included) to the conversation without generating.
    /// Returns the tokens added.
    pub fn prefill_text(&mut self, text: &str) -> Result<usize> {
        let c = CString::new(text).map_err(|_| anyhow!("text contains a NUL byte"))?;
        match unsafe { cp_prefill_text(self.ptr.as_ptr(), c.as_ptr()) } {
            n if n >= 0 => Ok(n as usize),
            -2 => bail!("the text could not be tokenized"),
            -4 => bail!("the conversation does not fit in the context"),
            rc => bail!("the forward pass failed (engine error {rc})"),
        }
    }

    pub fn context_tokens(&self) -> usize {
        unsafe { cp_context_tokens(self.ptr.as_ptr()) }.max(0) as usize
    }

    /// Called after every layer of every forward pass, on this thread.
    pub fn set_progress<F: FnMut(Progress) + 'static>(&mut self, f: F) {
        unsafe extern "C" fn trampoline(layer: c_int, n: c_int, t: c_int, user: *mut c_void) {
            let hook = &mut *(user as *mut ProgressHook);
            let stats = (t > 1 || layer == n).then(|| {
                let mut s = std::mem::MaybeUninit::<Stats>::zeroed();
                cp_get_stats(hook.model, s.as_mut_ptr());
                s.assume_init()
            });
            let p = Progress { layer: layer as usize, n_layers: n as usize, tokens: t as usize, stats };
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (hook.f)(p)));
        }
        let mut hook = Box::new(ProgressHook { model: self.ptr.as_ptr(), f: Box::new(f) });
        let user = &mut *hook as *mut ProgressHook as *mut c_void;
        unsafe { cp_set_progress(self.ptr.as_ptr(), Some(trampoline), user) };
        self.progress = Some(hook);
    }

    pub fn stats(&self) -> Stats {
        let mut s = std::mem::MaybeUninit::<Stats>::zeroed();
        unsafe {
            cp_get_stats(self.ptr.as_ptr(), s.as_mut_ptr());
            s.assume_init()
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            cp_set_progress(self.ptr.as_ptr(), None, std::ptr::null_mut());
            cp_close(self.ptr.as_ptr())
        }
    }
}
