//! Same-process native continuation. TensorFS owns loading; this DSO owns graph execution.
use super::*;
use pyo3::types::PyCapsule;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Duration;

const OK: u32 = 0;
const CANCELED: u32 = 1;
const INVALID: u32 = 2;
const CAPACITY: u32 = 3;
const CUDA: u32 = 4;
const INTERNAL: u32 = 5;
const UNPROVEN: u32 = 6;
const ARMED: u32 = 0;
const RUNNING: u32 = 1;
const DONE: u32 = 2;
const FAILED: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy)]
struct Status {
    kind: u32,
    code: i32,
    detail_len: u32,
    detail: [u8; 244],
}
impl Status {
    fn new(kind: u32, code: i32, message: &str) -> Self {
        let mut detail = [0; 244];
        let mut length = message.len().min(detail.len());
        while !message.is_char_boundary(length) {
            length -= 1;
        }
        detail[..length].copy_from_slice(&message.as_bytes()[..length]);
        Self {
            kind,
            code,
            detail_len: length as u32,
            detail,
        }
    }
    fn ok() -> Self {
        Self::new(OK, 0, "")
    }
    fn invalid(message: &str) -> Self {
        Self::new(INVALID, 0, message)
    }
    fn cuda(code: i32) -> Self {
        if code < 0 {
            return Self::invalid("native continuation is ineligible or closed");
        }
        Self::new(
            if code == 2 { CAPACITY } else { CUDA },
            code,
            &format!("native continuation: CUDA error {code}"),
        )
    }
    fn unproven() -> Self {
        Self::new(UNPROVEN, 999, "native continuation completion is unproven")
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Progress {
    phase: u32,
    completed_steps: u32,
    canceled: u32,
    done: u32,
    epoch: u64,
}

type Operation = unsafe extern "C" fn(*mut c_void) -> Status;
type Observe = unsafe extern "C" fn(*mut c_void, *mut Progress) -> Status;
type StartAfter = unsafe extern "C" fn(*mut c_void, *const usize, u32, u64) -> Status;
#[repr(C)]
struct Api {
    abi_version: u32,
    struct_size: u32,
    flags: u64,
    device: i32,
    reserved: u32,
    pid: u64,
    context: *mut c_void,
    retain: Operation,
    release: Operation,
    start: Operation,
    poll: Observe,
    cancel: Operation,
    wait: Observe,
    start_after: StartAfter,
}
// SAFETY: context is an opaque Arc<Continuation>; only its owning DSO dereferences it.
unsafe impl Send for Api {}

#[repr(C)]
struct Capsule {
    api: Api, // The checked C table is the capsule's first field, never the private Arc layout.
    _owner: Arc<Continuation>,
}

struct Continuation {
    owner: Arc<RepeatOwner>,
    state: Mutex<Progress>,
    gate: AtomicU8, // 0: armed, 1: start committed, 2: canceled before start
    cancel_requested: AtomicBool,
    cancel_issued: AtomicBool,
    input_invalid: AtomicBool,
    // A worker may outlive every public Python definition. Keep its provider module loaded.
    _module: Py<PyAny>,
}

// Keep the fixed, bounded ABI status by value. In particular, cancel/drop must not allocate
// a boxed Rust error whose ownership is unrelated to the capability's native-only contract.
#[allow(clippy::result_large_err)]
impl Continuation {
    fn checked(&self) -> Result<std::sync::MutexGuard<'_, Progress>, Status> {
        if self.owner.pid != std::process::id() {
            return Err(Status::invalid(
                "native continuation cannot be used after fork",
            ));
        }
        self.state.lock().map_err(|_| Status::unproven())
    }
    fn core(&self, epoch: u64) -> Result<std::sync::MutexGuard<'_, State>, Status> {
        let state = self.owner.state.lock().map_err(|_| Status::unproven())?;
        if state.pointer == 0 {
            return Err(Status::invalid("native repeat was closed"));
        }
        if state.failed {
            return Err(Status::unproven());
        }
        if state.epoch != epoch || state.pending != Some(epoch) {
            return Err(Status::invalid("stale native continuation epoch"));
        }
        Ok(state)
    }
    fn start(&self) -> Result<(), Status> {
        self.start_input(None)
    }
    fn start_input(&self, input: Option<(&[usize], u64)>) -> Result<(), Status> {
        if self.owner.pid != std::process::id() {
            return Err(Status::invalid(
                "native continuation cannot be used after fork",
            ));
        }
        // Atomic commitment: cancel never waits for this mutex or a CUDA driver call.
        if let Err(gate) = self
            .gate
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
        {
            return Err(if gate == 2 {
                Status::new(CANCELED, 0, "native continuation canceled before launch")
            } else {
                Status::invalid("native continuation start is one-shot")
            });
        }
        let mut progress = self.checked()?;
        let mut core = match self.core(progress.epoch) {
            Ok(core) => core,
            Err(status) => {
                // A checked closed/stale definition refused before any launch by this epoch.
                progress.phase = if status.kind == INVALID { DONE } else { FAILED };
                progress.done = u32::from(status.kind == INVALID);
                return Err(status);
            }
        };
        // SAFETY: reserved epoch exclusively owns this handle; its caller-ready event is recorded.
        let code = unsafe {
            match input {
                None => cozy_repeat_launch_armed(core.pointer as *mut c_void),
                Some((streams, error)) => cozy_repeat_launch_after(
                    core.pointer as *mut c_void,
                    streams.as_ptr(),
                    streams.len() as u32,
                    error,
                ),
            }
        };
        if code != 0 {
            core.failed = true;
            progress.phase = FAILED;
            return Err(Status::cuda(code));
        }
        core.launched = true;
        core.launches += 1;
        progress.phase = RUNNING;
        // A cancellation during initialization remains in cancel_requested. poll/wait emits
        // its GPU control write only after initialization, never before the memset resets it.
        Ok(())
    }
    fn cancel(&self) -> Result<(), Status> {
        if self.owner.pid != std::process::id() {
            return Err(Status::invalid(
                "native continuation cannot be used after fork",
            ));
        }
        self.cancel_requested.store(true, Ordering::Release);
        let _ = self
            .gate
            .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire);
        Ok(())
    }
    fn observe(&self, snapshot: bool) -> Result<Progress, Status> {
        let mut progress = self.checked()?;
        if progress.phase == DONE {
            return Ok(*progress);
        }
        if progress.phase == ARMED {
            if self.gate.load(Ordering::Acquire) == 2 {
                let mut core = self.owner.state.lock().map_err(|_| Status::unproven())?;
                if core.pending == Some(progress.epoch) {
                    core.pending = None;
                }
                // The atomic gate proves this epoch never launched, even if its definition
                // was subsequently closed. Do not touch a later epoch's reservation.
                progress.phase = DONE;
                progress.done = 1;
                progress.canceled = 1;
            }
            return Ok(*progress);
        }
        if progress.phase == FAILED {
            return Err(Status::unproven());
        }
        let mut core = self.core(progress.epoch)?;
        if self.cancel_requested.load(Ordering::Acquire)
            && !self.cancel_issued.swap(true, Ordering::AcqRel)
        {
            // SAFETY: start has completed initialization and published RUNNING under this lock.
            let code = unsafe { cozy_repeat_cancel(core.pointer as *mut c_void) };
            if code != 0 {
                core.failed = true;
                progress.phase = FAILED;
                return Err(Status::cuda(code));
            }
        }
        let (mut completed, mut canceled, mut done) = (0, 0, 0);
        // Internal completion waiting queries only the event. GPU progress snapshots are
        // requested by public observers, never generated every millisecond by the worker.
        // SAFETY: stack outputs and one owned running epoch; producer and graph owners remain held.
        let mut code = if snapshot {
            unsafe {
                cozy_repeat_poll(
                    core.pointer as *mut c_void,
                    &mut completed,
                    &mut canceled,
                    &mut done,
                )
            }
        } else {
            completed = progress.completed_steps;
            canceled = progress.canceled;
            unsafe { cozy_repeat_ready(core.pointer as *mut c_void, &mut done) }
        };
        if code == 0 && done != 0 {
            // The execution event is already complete; obtain a final checked snapshot.
            code = unsafe {
                cozy_repeat_wait(core.pointer as *mut c_void, &mut completed, &mut canceled)
            };
        }
        if code != 0 {
            core.failed = true;
            progress.phase = FAILED;
            return Err(Status::cuda(code));
        }
        progress.completed_steps = completed;
        progress.canceled = canceled;
        if done != 0 {
            let (mut entered, mut invalid, mut checked) = (0, 0, 0);
            // SAFETY: wait above completed the owned epoch and populated its host snapshot.
            unsafe {
                cozy_repeat_input_status(
                    core.pointer as *mut c_void,
                    &mut entered,
                    &mut invalid,
                    &mut checked,
                );
            }
            if checked == 0 {
                return Err(Status::unproven());
            }
            self.input_invalid.store(invalid != 0, Ordering::Release);
            progress.phase = DONE;
            progress.done = 1;
            core.pending = None;
        }
        Ok(*progress)
    }
    fn poll(&self) -> Result<Progress, Status> {
        self.observe(true)
    }
    fn wait(&self) -> Result<Progress, Status> {
        loop {
            let progress = self.observe(false)?;
            if progress.done != 0 || progress.phase == ARMED {
                return Ok(progress);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Drop for Continuation {
    fn drop(&mut self) {
        if self.owner.pid != std::process::id() {
            // Never acquire inherited CUDA/mutex state. Retain the real definition after fork.
            std::mem::forget(self.owner.clone());
            return;
        }
        // An abandoned native reference must not permit reuse while its graph still runs.
        if self
            .cancel()
            .and_then(|()| self.wait().map(|_| ()))
            .is_err()
        {
            std::mem::forget(self.owner.clone());
        }
    }
}

fn boundary(
    context: *mut c_void,
    call: impl FnOnce(&Continuation) -> Result<(), Status>,
) -> Status {
    if context.is_null() {
        return Status::invalid("null native continuation");
    }
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the named capsule contract requires a retained context from this DSO.
        let continuation = unsafe { &*context.cast::<Continuation>() };
        call(continuation)
    })) {
        Ok(Ok(())) => Status::ok(),
        Ok(Err(status)) => status,
        Err(_) => Status::new(
            INTERNAL,
            999,
            "native continuation panicked; retain its owners",
        ),
    }
}

unsafe extern "C" fn retain(context: *mut c_void) -> Status {
    boundary(context, |owner| {
        if owner.owner.pid != std::process::id() {
            return Err(Status::invalid(
                "native continuation cannot be retained after fork",
            ));
        }
        // SAFETY: capsule or a prior retain keeps this provider-allocated Arc alive.
        unsafe {
            Arc::increment_strong_count(context.cast::<Continuation>());
        }
        Ok(())
    })
}
unsafe extern "C" fn release(context: *mut c_void) -> Status {
    if context.is_null() {
        return Status::invalid("null native continuation");
    }
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller owns one prior retain. End the borrowed inspection before dropping it.
        let pid = unsafe { (&*context.cast::<Continuation>()).owner.pid };
        if pid != std::process::id() {
            return Status::invalid("native continuation cannot be released after fork");
        }
        // SAFETY: consumes precisely one prior retain, in its allocating DSO.
        unsafe {
            drop(Arc::from_raw(context.cast::<Continuation>()));
        }
        Status::ok()
    })) {
        Ok(status) => status,
        Err(_) => Status::new(
            INTERNAL,
            999,
            "native continuation release panicked; retain its owners",
        ),
    }
}
unsafe extern "C" fn start(context: *mut c_void) -> Status {
    boundary(context, Continuation::start)
}
unsafe extern "C" fn start_after(
    context: *mut c_void,
    streams: *const usize,
    count: u32,
    error: u64,
) -> Status {
    if streams.is_null()
        || count == 0
        || count > 4
        || !(streams as usize).is_multiple_of(std::mem::align_of::<usize>())
    {
        return Status::invalid("event input requires one to four owned native streams");
    }
    // SAFETY: the retained same-process capability requires count readable native handles.
    let streams = unsafe { std::slice::from_raw_parts(streams, count as usize) };
    if streams.iter().any(|stream| *stream <= 2) || error % 4 != 0 {
        return Status::invalid("event input has a special stream or unaligned error word");
    }
    boundary(context, |owner| owner.start_input(Some((streams, error))))
}
unsafe extern "C" fn cancel(context: *mut c_void) -> Status {
    boundary(context, Continuation::cancel)
}
unsafe extern "C" fn poll(context: *mut c_void, out: *mut Progress) -> Status {
    if out.is_null() {
        return Status::invalid("null native progress output");
    }
    boundary(context, |owner| {
        let progress = owner.poll()?;
        // SAFETY: importer supplies one writable ABI-sized progress value.
        unsafe {
            out.write(progress);
        }
        if progress.done != 0 && owner.input_invalid.load(Ordering::Acquire) {
            return Err(Status::invalid(
                "native decoder rejected input values before body entry",
            ));
        }
        Ok(())
    })
}
unsafe extern "C" fn wait(context: *mut c_void, out: *mut Progress) -> Status {
    if out.is_null() {
        return Status::invalid("null native progress output");
    }
    boundary(context, |owner| {
        let progress = owner.wait()?;
        // SAFETY: importer supplies one writable ABI-sized progress value.
        unsafe {
            out.write(progress);
        }
        if progress.done != 0 && owner.input_invalid.load(Ordering::Acquire) {
            return Err(Status::invalid(
                "native decoder rejected input values before body entry",
            ));
        }
        Ok(())
    })
}

pub(super) fn export(
    py: Python<'_>,
    owner: Arc<RepeatOwner>,
    producer: usize,
) -> PyResult<(Py<PyAny>, u64)> {
    owner.check_process()?;
    // Take the module owner before arming; no Python allocation failure may strand an epoch.
    let module = py
        .import("cozy_runtime._kernels._execution")?
        .into_any()
        .unbind();
    // If completion becomes unproven and the core is retained, retain its code module too.
    owner
        .owners
        .lock()
        .map_err(|_| PyRuntimeError::new_err("prepared owners poisoned"))?
        .push(module.clone_ref(py));
    let epoch = py
        .detach(|| {
            let mut core = owner.state.lock().map_err(|_| 999)?;
            if core.pointer == 0 {
                return Err(-2);
            }
            if core.failed {
                return Err(999);
            }
            if core.pending.is_some() {
                return Err(-4);
            }
            let epoch = core.epoch.checked_add(1).ok_or(999)?;
            // SAFETY: runs on the submitting OS thread; owns the definition and actual caller stream.
            let code = unsafe { cozy_repeat_arm(core.pointer as *mut c_void, producer) };
            if code != 0 {
                if code >= 0 {
                    core.failed = true;
                }
                return Err(code);
            }
            core.epoch = epoch;
            core.pending = Some(epoch);
            Ok(epoch)
        })
        .map_err(|code| cuda_error(py, "arm native load continuation", code))?;
    let owner = Arc::new(Continuation {
        owner,
        state: Mutex::new(Progress {
            phase: ARMED,
            completed_steps: 0,
            canceled: 0,
            done: 0,
            epoch,
        }),
        gate: AtomicU8::new(0),
        cancel_requested: AtomicBool::new(false),
        cancel_issued: AtomicBool::new(false),
        input_invalid: AtomicBool::new(false),
        _module: module,
    });
    let context = Arc::as_ptr(&owner) as *mut c_void;
    let api = Api {
        abi_version: 1,
        struct_size: std::mem::size_of::<Api>() as u32,
        flags: 15,
        device: owner.owner.device,
        reserved: 0,
        pid: u64::from(owner.owner.pid),
        context,
        retain,
        release,
        start,
        poll,
        cancel,
        wait,
        start_after,
    };
    let capsule = PyCapsule::new_with_value(
        py,
        Capsule { api, _owner: owner },
        c"cozy.execution.continuation.v1",
    )?;
    Ok((capsule.into_any().unbind(), epoch))
}

const _: () = assert!(std::mem::size_of::<Status>() == 256);
const _: () = assert!(std::mem::size_of::<Progress>() == 24);

#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<Api>() == 96);
