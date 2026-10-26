//! An ordered native consumer of the existing same-process continuation capabilities.
//! One worker advances the accepted stages; it never calls Python or owns a second queue.
use super::continuation::{
    CANCELED, CAPACITY, CUDA, DONE, INTERNAL, INVALID, OK, Progress, Status, UNPROVEN,
};
use super::*;
use pyo3::ffi;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Condvar;
use std::sync::atomic::{AtomicBool, Ordering};

type Operation = unsafe extern "C" fn(*mut c_void) -> Status;
type Observe = unsafe extern "C" fn(*mut c_void, *mut Progress) -> Status;

#[repr(C)]
#[derive(Clone, Copy)]
struct Api {
    version: u32,
    size: u32,
    flags: u64,
    device: i32,
    reserved: u32,
    pid: u64,
    context: *mut c_void,
    retain: Option<Operation>,
    release: Option<Operation>,
    start: Option<Operation>,
    poll: Option<Observe>,
    cancel: Option<Operation>,
    wait: Option<Observe>,
}

#[repr(C)]
struct ReusableApi {
    base: Api,
    start_after: Option<unsafe extern "C" fn(*mut c_void, *const usize, u32, u64) -> Status>,
    rearm: Option<Operation>,
    finish: Option<Operation>,
}

struct Input {
    api: Api,
    owner: Option<Py<PyAny>>,
    quiescent: AtomicBool,
    rearm: Option<Operation>,
    finish: Option<Operation>,
    finished: AtomicBool,
}
// SAFETY: v1 requires retained same-process, native-only thread-safe operations.
unsafe impl Send for Input {}
// SAFETY: the provider serializes observe and atomically arbitrates start/cancel.
unsafe impl Sync for Input {}

impl Input {
    fn import(value: &Bound<'_, PyAny>, device: i32) -> PyResult<Self> {
        let invalid = || PyRuntimeError::new_err("invalid native sequence continuation v1");
        // SAFETY: validate CPython's exact named capsule before reading its C prefix.
        let pointer = unsafe {
            if ffi::PyCapsule_IsValid(value.as_ptr(), c"cozy.execution.continuation.v1".as_ptr())
                != 1
            {
                return Err(invalid());
            }
            ffi::PyCapsule_GetPointer(value.as_ptr(), c"cozy.execution.continuation.v1".as_ptr())
        };
        if pointer.is_null() || !(pointer as usize).is_multiple_of(std::mem::align_of::<Api>()) {
            return Err(invalid());
        }
        // SAFETY: the named capability guarantees a live fixed header; check its size first.
        let (version, size) = unsafe { (*pointer.cast::<u32>(), *pointer.cast::<u32>().add(1)) };
        if version != 1 || size < std::mem::size_of::<Api>() as u32 {
            return Err(invalid());
        }
        // SAFETY: the validated size covers every field copied below, including nullable fns.
        let api = unsafe { *pointer.cast::<Api>() };
        if api.flags & 7 != 7
            || api.pid != u64::from(std::process::id())
            || api.device != device
            || api.reserved != 0
            || api.context.is_null()
            || [api.retain, api.release, api.start, api.cancel]
                .iter()
                .any(Option::is_none)
            || api.poll.is_none()
            || api.wait.is_none()
        {
            return Err(invalid());
        }
        let owner = value.clone().unbind();
        let (rearm, finish) = if api.flags & 16 != 0 {
            if size < std::mem::size_of::<ReusableApi>() as u32 {
                return Err(invalid());
            }
            // SAFETY: capability flag and size cover the optional same-process v1 tail.
            let extended = unsafe { &*pointer.cast::<ReusableApi>() };
            (
                Some(extended.rearm.ok_or_else(invalid)?),
                Some(extended.finish.ok_or_else(invalid)?),
            )
        } else {
            (None, None)
        };
        // SAFETY: one matching provider-owned retain; keep the actual capsule/code alive.
        let status = unsafe { api.retain.expect("validated retain")(api.context) };
        if status.kind != OK {
            std::mem::forget(owner);
            return Err(raise(value.py(), Status::unproven(), 0));
        }
        Ok(Self {
            api,
            owner: Some(owner),
            quiescent: AtomicBool::new(false),
            rearm,
            finish,
            finished: AtomicBool::new(false),
        })
    }

    fn start(&self) -> Status {
        // SAFETY: native-only retained v1 start. This may block in a source stage.
        unsafe { self.api.start.expect("validated start")(self.api.context) }
    }

    fn cancel(&self) -> Status {
        // SAFETY: v1 cancellation is atomic, nonblocking and makes no Python callback.
        unsafe { self.api.cancel.expect("validated cancel")(self.api.context) }
    }

    fn observe(&self, wait: bool) -> (Status, Progress) {
        let mut progress = Progress::default();
        let observe = if wait { self.api.wait } else { self.api.poll }.expect("validated observer");
        // SAFETY: retained context plus writable ABI-sized stack output.
        let status = unsafe { observe(self.api.context, &mut progress) };
        if progress.done != 0 && matches!(status.kind, OK | CANCELED | INVALID | CAPACITY) {
            self.quiescent.store(true, Ordering::Release);
        }
        (status, progress)
    }

    fn rearm(&self) -> Status {
        let Some(rearm) = self.rearm else {
            return Status::invalid("stage has no reusable continuation capability");
        };
        if !self.quiescent.load(Ordering::Acquire) || self.finished.load(Ordering::Acquire) {
            return Status::invalid("rearm requires an unfinished checked stage");
        }
        self.quiescent.store(false, Ordering::Release);
        // SAFETY: optional tail validated at import; provider checks prior success and
        // preserves sticky cancellation while reserving its next native epoch.
        unsafe { rearm(self.api.context) }
    }

    fn finish(&self) -> Status {
        if self.finished.load(Ordering::Acquire) {
            return Status::ok();
        }
        if !self.quiescent.load(Ordering::Acquire) {
            return Status::unproven();
        }
        let status = match self.finish {
            // SAFETY: successful drain precedes release of this reusable reservation.
            Some(finish) => unsafe { finish(self.api.context) },
            None => Status::ok(),
        };
        if status.kind == OK {
            self.finished.store(true, Ordering::Release);
        }
        status
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        let owner = self.owner.take();
        if self.api.pid != u64::from(std::process::id())
            || !self.quiescent.load(Ordering::Acquire)
            || self.finish.is_some() && !self.finished.load(Ordering::Acquire)
        {
            // No checked drain: retain the actual provider context, capsule and its code.
            std::mem::forget(owner);
            return;
        }
        // SAFETY: consumes our one successful retain only after checked native completion.
        let status = unsafe { self.api.release.expect("validated release")(self.api.context) };
        if status.kind != OK {
            std::mem::forget(owner);
        }
    }
}

#[derive(Clone, Default)]
struct Snapshot {
    active: Option<usize>,
    completed_stages: usize,
    completed_steps: u64,
    canceled: bool,
    done: bool,
    completion_proven: bool,
    failure: Option<(usize, Status)>,
    completed_passes: u32,
}

struct Shared {
    stages: Vec<Input>,
    state: Mutex<Snapshot>,
    changed: Condvar,
    started: AtomicBool,
    canceled: AtomicBool,
    cancel_failed: AtomicBool,
    pid: u32,
    _module: Py<PyAny>,
    repeat: Option<(usize, usize, u32)>,
}

fn unsafe_status(status: Status) -> bool {
    matches!(status.kind, CUDA | INTERNAL | UNPROVEN)
}

fn raise(py: Python<'_>, status: Status, stage: usize) -> PyErr {
    let text = String::from_utf8_lossy(&status.detail[..(status.detail_len as usize).min(244)]);
    let error = NativeExecutionError::new_err(format!("native sequence stage {stage}: {text}"));
    if unsafe_status(status) || status.kind == CAPACITY {
        let _ = error.value(py).setattr(
            "error_code",
            if status.kind == CAPACITY {
                2
            } else {
                status.code
            },
        );
    }
    error
}

impl Shared {
    fn check_process(&self) -> PyResult<()> {
        if self.pid != std::process::id() {
            return Err(PyRuntimeError::new_err(
                "native sequence cannot be used after fork",
            ));
        }
        Ok(())
    }

    fn cancel(&self) {
        self.canceled.store(true, Ordering::Release);
        for stage in &self.stages {
            if !matches!(stage.cancel().kind, OK | CANCELED) {
                self.cancel_failed.store(true, Ordering::Release);
            }
        }
    }

    fn begin(self: &Arc<Self>) -> PyResult<()> {
        self.check_process()?;
        if self.started.swap(true, Ordering::AcqRel) {
            return Err(PyRuntimeError::new_err(
                "native sequence submission is one-shot",
            ));
        }
        let shared = self.clone();
        if let Err(error) = std::thread::Builder::new()
            .name("cozy-native-sequence".into())
            .spawn(move || shared.run())
        {
            self.cancel();
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            state.failure = Some((
                0,
                Status::new(INTERNAL, 999, "native sequence worker did not start"),
            ));
            state.done = true;
            self.changed.notify_all();
            // Keep the accepted native reservations if no cleanup worker can be started.
            std::mem::forget(self.clone());
            return Err(PyRuntimeError::new_err(format!(
                "native sequence worker: {error}"
            )));
        }
        Ok(())
    }

    fn run(self: Arc<Self>) {
        let outcome = catch_unwind(AssertUnwindSafe(|| self.advance()));
        let mut failure = match outcome {
            Ok(result) => result.err(),
            Err(_) => Some((
                0,
                Status::new(INTERNAL, 999, "native sequence worker panicked"),
            )),
        };
        if failure.is_some() || self.canceled.load(Ordering::Acquire) {
            self.cancel();
            // Drain every accepted stage, including those that were never started.
            for (index, stage) in self.stages.iter().enumerate() {
                let (status, progress) = stage.observe(true);
                if progress.done == 0 || !stage.quiescent.load(Ordering::Acquire) {
                    let status = if unsafe_status(status) {
                        status
                    } else {
                        Status::unproven()
                    };
                    failure = Some((index, status));
                } else if status.kind != OK
                    && status.kind != CANCELED
                    && (failure.is_none() || unsafe_status(status))
                {
                    failure = Some((index, status));
                }
            }
        }
        if self.cancel_failed.load(Ordering::Acquire) {
            failure = Some((
                0,
                Status::new(UNPROVEN, 999, "native stage cancellation failed"),
            ));
        }
        for (index, stage) in self.stages.iter().enumerate() {
            if stage.quiescent.load(Ordering::Acquire) && stage.finish().kind != OK {
                failure = Some((index, Status::unproven()));
            }
        }
        let complete = self.stages.iter().all(|stage| {
            stage.quiescent.load(Ordering::Acquire) && stage.finished.load(Ordering::Acquire)
        }) && !failure.is_some_and(|(_, status)| unsafe_status(status));
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.active = None;
        state.canceled |= self.canceled.load(Ordering::Acquire);
        state.failure = failure;
        state.completion_proven = complete;
        state.done = true;
        drop(state);
        self.changed.notify_all();
        if !complete {
            // Retain the whole actual chain, including future source/graph owners.
            std::mem::forget(self.clone());
        }
    }

    fn advance(&self) -> Result<(), (usize, Status)> {
        let (begin, end, passes) = self.repeat.unwrap_or((0, self.stages.len(), 1));
        for index in 0..begin {
            self.advance_stage(index, false)?;
        }
        for pass in 0..passes {
            let before = self
                .state
                .lock()
                .map_err(|_| (begin, Status::unproven()))?
                .completed_stages;
            for index in begin..end {
                self.advance_stage(index, pass != 0)?;
            }
            // A cancellation arriving after the final stage completed must not erase
            // that whole pass from progress (the GPU sampler may already advance its step).
            let mut state = self.state.lock().map_err(|_| (begin, Status::unproven()))?;
            if state.completed_stages - before == end - begin {
                state.completed_passes += 1;
            }
            drop(state);
            if self.canceled.load(Ordering::Acquire) {
                return Ok(());
            }
        }
        for index in end..self.stages.len() {
            self.advance_stage(index, false)?;
        }
        Ok(())
    }

    fn advance_stage(&self, index: usize, rearm: bool) -> Result<(), (usize, Status)> {
        let stage = &self.stages[index];
        if self.canceled.load(Ordering::Acquire) {
            return Ok(());
        }
        if rearm {
            let status = stage.rearm();
            if status.kind == CANCELED {
                self.canceled.store(true, Ordering::Release);
                return Ok(());
            }
            if status.kind != OK {
                return Err((index, status));
            }
        }
        self.state
            .lock()
            .map_err(|_| (index, Status::unproven()))?
            .active = Some(index);
        // No sequence lock spans provider calls: cancellation remains independent even
        // when a source-stage start runs its existing blocking refill worker inline.
        let status = stage.start();
        if !matches!(status.kind, OK | CANCELED) {
            return Err((index, status));
        }
        let (status, progress) = stage.observe(true);
        if !stage.quiescent.load(Ordering::Acquire) || progress.phase != DONE {
            return Err((
                index,
                if unsafe_status(status) {
                    status
                } else {
                    Status::unproven()
                },
            ));
        }
        if !matches!(status.kind, OK | CANCELED) {
            return Err((index, status));
        }
        let mut state = self.state.lock().map_err(|_| (index, Status::unproven()))?;
        state.completed_steps += u64::from(progress.completed_steps);
        state.active = None;
        if status.kind == CANCELED || progress.canceled != 0 {
            state.canceled = true;
            self.canceled.store(true, Ordering::Release);
            return Ok(());
        }
        state.completed_stages += 1;
        Ok(())
    }

    fn snapshot(&self, observe: bool) -> Result<Snapshot, Status> {
        let before = self.state.lock().map_err(|_| Status::unproven())?.clone();
        if observe && !before.done {
            if let Some(active) = before.active {
                let (status, progress) = self.stages[active].observe(false);
                if !matches!(status.kind, OK | CANCELED) {
                    // Only the worker may publish the composed terminal error, after it
                    // drains every accepted stage. A single failed stage is not a whole-
                    // sequence completion certificate, even if its own cleanup finished.
                    return self
                        .state
                        .lock()
                        .map(|state| state.clone())
                        .map_err(|_| Status::unproven());
                }
                let mut after = self.state.lock().map_err(|_| Status::unproven())?.clone();
                if after.active == Some(active)
                    && after.completed_stages == before.completed_stages
                    && !after.done
                {
                    after.completed_steps += u64::from(progress.completed_steps);
                    after.canceled |= progress.canceled != 0;
                }
                return Ok(after);
            }
        }
        Ok(before)
    }

    fn wait(&self) -> Result<Snapshot, Status> {
        let mut state = self.state.lock().map_err(|_| Status::unproven())?;
        while !state.done {
            state = self.changed.wait(state).map_err(|_| Status::unproven())?;
        }
        Ok(state.clone())
    }
}

#[pyclass(module = "cozy_runtime._kernels._execution", frozen)]
pub(super) struct NativeSequence {
    shared: Arc<Shared>,
}

#[pymethods]
impl NativeSequence {
    #[new]
    #[pyo3(signature = (stages, device, repeat = None))]
    fn new(
        py: Python<'_>,
        stages: Vec<Py<PyAny>>,
        device: i32,
        repeat: Option<(usize, usize, u32)>,
    ) -> PyResult<Self> {
        if stages.is_empty() || stages.len() > 65536 {
            return Err(PyRuntimeError::new_err(
                "native sequence needs 1..65536 prepared stages",
            ));
        }
        if let Some((begin, end, count)) = repeat {
            if begin >= end
                || end > stages.len()
                || count == 0
                || (end - begin) as u64 * u64::from(count) + (stages.len() - (end - begin)) as u64
                    > u64::from(u32::MAX)
            {
                return Err(PyRuntimeError::new_err(
                    "invalid bounded native repeat range",
                ));
            }
        }
        let module = py
            .import("cozy_runtime._kernels._execution")?
            .into_any()
            .unbind();
        let mut inputs = Vec::<Input>::new();
        for value in &stages {
            let imported = Input::import(value.bind(py), device);
            match imported {
                Ok(input)
                    if !inputs
                        .iter()
                        .any(|prior| prior.api.context == input.api.context)
                        && !repeat.is_some_and(|(begin, end, count)| {
                            count > 1
                                && inputs.len() >= begin
                                && inputs.len() < end
                                && input.rearm.is_none()
                        }) =>
                {
                    inputs.push(input)
                }
                other => {
                    let error = match other {
                        Err(error) => error,
                        Ok(input) => {
                            inputs.push(input);
                            PyRuntimeError::new_err(
                                "native sequence requires distinct capabilities and reusable repeated stages",
                            )
                        }
                    };
                    py.detach(|| {
                        for input in &inputs {
                            input.cancel();
                        }
                        for input in &inputs {
                            input.observe(true);
                            input.finish();
                        }
                    });
                    return Err(error);
                }
            }
        }
        Ok(Self {
            shared: Arc::new(Shared {
                stages: inputs,
                state: Mutex::new(Snapshot::default()),
                changed: Condvar::new(),
                started: AtomicBool::new(false),
                canceled: AtomicBool::new(false),
                cancel_failed: AtomicBool::new(false),
                pid: std::process::id(),
                _module: module,
                repeat,
            }),
        })
    }

    fn start(&self) -> PyResult<()> {
        self.shared.begin()
    }

    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        self.shared.check_process()?;
        self.shared.cancel();
        if self.shared.cancel_failed.load(Ordering::Acquire) {
            return Err(raise(py, Status::unproven(), 0));
        }
        Ok(())
    }

    fn poll(&self, py: Python<'_>) -> PyResult<(u64, bool, bool)> {
        self.shared.check_process()?;
        let result = py
            .detach(|| self.shared.snapshot(true))
            .map_err(|status| raise(py, status, 0))?;
        if let Some((stage, status)) = result.failure {
            return Err(raise(py, status, stage));
        }
        Ok((result.completed_steps, result.canceled, result.done))
    }

    fn wait(&self, py: Python<'_>) -> PyResult<(u64, bool)> {
        self.shared.check_process()?;
        if !self.shared.started.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err("native sequence has not started"));
        }
        let result = py
            .detach(|| self.shared.wait())
            .map_err(|status| raise(py, status, 0))?;
        if let Some((stage, status)) = result.failure {
            return Err(raise(py, status, stage));
        }
        Ok((result.completed_steps, result.canceled))
    }

    fn facts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.shared.check_process()?;
        let state = self
            .shared
            .snapshot(false)
            .map_err(|status| raise(py, status, 0))?;
        let facts = PyDict::new(py);
        facts.set_item("stages", self.shared.stages.len())?;
        facts.set_item("completed_stages", state.completed_stages)?;
        facts.set_item("completed_passes", state.completed_passes)?;
        facts.set_item("repeat_range", self.shared.repeat)?;
        facts.set_item("active_stage", state.active)?;
        facts.set_item("completed_steps", state.completed_steps)?;
        facts.set_item("canceled", state.canceled)?;
        facts.set_item("done", state.done)?;
        facts.set_item("completion_proven", state.completion_proven)?;
        facts.set_item("failed_stage", state.failure.map(|(stage, _)| stage))?;
        facts.set_item("native_stage_advancement", true)?;
        Ok(facts)
    }

    fn close_owner(&self, py: Python<'_>) -> PyResult<()> {
        self.shared.check_process()?;
        self.shared.cancel();
        if !self.shared.started.load(Ordering::Acquire) {
            self.shared.begin()?;
        }
        let state = py
            .detach(|| self.shared.wait())
            .map_err(|status| raise(py, status, 0))?;
        if !state.completion_proven || self.shared.cancel_failed.load(Ordering::Acquire) {
            return Err(raise(py, Status::unproven(), state.active.unwrap_or(0)));
        }
        Ok(())
    }
}

impl Drop for NativeSequence {
    fn drop(&mut self) {
        if self.shared.pid != std::process::id() {
            std::mem::forget(self.shared.clone());
            return;
        }
        self.shared.cancel();
        if !self.shared.started.load(Ordering::Acquire) {
            let _ = self.shared.begin();
        }
    }
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<Api>() == 88);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<ReusableApi>() == 112);
