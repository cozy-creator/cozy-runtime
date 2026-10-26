//! One resident graph definition, one GPU-driven loop, and its complete owner lifetime.
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
use std::{
    ffi::{CStr, c_char, c_void},
    ops::Deref,
    sync::{Arc, Mutex},
};

mod continuation;
mod sequence;

pyo3::create_exception!(_execution, NativeExecutionError, PyRuntimeError);

unsafe extern "C" {
    fn cozy_repeat_create(
        source: usize,
        index: usize,
        device: i32,
        steps: u32,
        prefix: usize,
        suffix: usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn cozy_repeat_launch(run: *mut c_void, producer: usize) -> i32;
    fn cozy_repeat_arm(run: *mut c_void, producer: usize) -> i32;
    fn cozy_repeat_launch_armed(run: *mut c_void) -> i32;
    fn cozy_repeat_launch_after(
        run: *mut c_void,
        streams: *const usize,
        count: u32,
        error: u64,
    ) -> i32;
    fn cozy_repeat_input_status(
        run: *mut c_void,
        entered: *mut u32,
        invalid: *mut u32,
        checked: *mut i32,
    );
    fn cozy_repeat_ready(run: *mut c_void, done: *mut i32) -> i32;
    fn cozy_repeat_poll(
        run: *mut c_void,
        completed: *mut u32,
        canceled: *mut u32,
        done: *mut i32,
    ) -> i32;
    fn cozy_repeat_cancel(run: *mut c_void) -> i32;
    fn cozy_repeat_wait(run: *mut c_void, completed: *mut u32, canceled: *mut u32) -> i32;
    fn cozy_repeat_nodes(run: *mut c_void) -> u32;
    fn cozy_repeat_validated_nodes(run: *mut c_void) -> u32;
    fn cozy_repeat_prefix_nodes(run: *mut c_void) -> u32;
    fn cozy_repeat_suffix_nodes(run: *mut c_void) -> u32;
    fn cozy_repeat_inspection(run: *mut c_void) -> *const c_char;
    fn cozy_repeat_destroy(run: *mut c_void) -> i32;
}

fn ineligible(py: Python<'_>, detail: impl Into<String>) -> PyErr {
    let error = NativeExecutionError::new_err(detail.into());
    let _ = error.value(py).setattr("eligibility_refusal", true);
    error
}

fn cuda_error(py: Python<'_>, operation: &str, code: i32) -> PyErr {
    if matches!(code, -102..=-100) {
        return ineligible(
            py,
            if code == -102 {
                "prepared submission during CUDA capture is not supported"
            } else if code == -101 {
                "prepared graph body is empty"
            } else {
                "prepared graph contains unsupported node types or exceeds bounded inspection limits"
            },
        );
    }
    if code < 0 {
        return PyRuntimeError::new_err(match code {
            -1 => "stale execution token",
            -2 => "prepared repeat is closed",
            -4 => "prepared repeat is reserved by a native load",
            _ => "prepared repeat is not running",
        });
    }
    let error = NativeExecutionError::new_err(format!("{operation}: CUDA error {code}"));
    let _ = error.value(py).setattr("error_code", code);
    error
}

struct State {
    pointer: usize,
    launched: bool,
    failed: bool,
    epoch: u64,
    launches: u64,
    pending: Option<u64>,
    destroy_failed: bool,
    streamed: bool, // Exclusive reusable continuation owns this definition between segments.
}

struct RepeatOwner {
    state: Mutex<State>,
    owners: Mutex<Vec<Py<PyAny>>>,
    pid: u32,
    nodes: u32,
    validated_nodes: u32,
    prefix_nodes: u32,
    suffix_nodes: u32,
    has_prefix: bool,
    has_suffix: bool,
    device: i32,
    steps: u32,
}

#[pyclass(
    name = "ResidentRepeat",
    module = "cozy_runtime._kernels._execution",
    frozen
)]
struct ResidentRepeat {
    inner: Arc<RepeatOwner>,
}

impl Deref for ResidentRepeat {
    type Target = RepeatOwner;
    fn deref(&self) -> &RepeatOwner {
        &self.inner
    }
}

impl RepeatOwner {
    fn check_process(&self) -> PyResult<()> {
        if self.pid != std::process::id() {
            return Err(PyRuntimeError::new_err(
                "prepared GPU work cannot be used after fork",
            ));
        }
        Ok(())
    }

    // The entire mutex lifetime is inside detach: no lock is held while reacquiring the GIL.
    fn native<T: Send>(
        &self,
        py: Python<'_>,
        operation: &str,
        epoch: Option<u64>,
        call: impl FnOnce(&mut State) -> Result<T, i32> + Send,
    ) -> PyResult<T> {
        self.check_process()?;
        py.detach(|| {
            let mut state = self.state.lock().map_err(|_| 999)?;
            if epoch.is_some_and(|expected| expected != state.epoch) {
                return Err(-1);
            }
            if state.pointer == 0 {
                return Err(-2);
            }
            if state.failed {
                return Err(999);
            }
            if state.pending.is_some() || state.streamed {
                return Err(-4);
            }
            let result = call(&mut state);
            if result.as_ref().is_err_and(|code| *code >= 0) {
                state.failed = true;
            }
            result
        })
        .map_err(|code| cuda_error(py, operation, code))
    }
}

impl RepeatOwner {
    fn dispose(&self, py: Python<'_>, epoch: Option<u64>) -> PyResult<()> {
        self.check_process()?;
        py.detach(|| {
            let mut state = self.state.lock().map_err(|_| 999)?;
            if epoch.is_some_and(|expected| expected != state.epoch) {
                return Err(-1);
            }
            if state.destroy_failed {
                return Err(999);
            }
            if state.pointer == 0 {
                return Ok(());
            }
            // SAFETY: destruction joins both streams before releasing graph/control storage.
            let code = unsafe { cozy_repeat_destroy(state.pointer as *mut c_void) };
            if code != 0 {
                state.failed = true;
                state.destroy_failed = true;
                return Err(code);
            }
            state.pointer = 0;
            Ok(())
        })
        .map_err(|code| cuda_error(py, "close resident repeat", code))?;
        let owners = {
            let mut held = self
                .owners
                .lock()
                .map_err(|_| PyRuntimeError::new_err("prepared owners poisoned"))?;
            std::mem::take(&mut *held)
        };
        // Python destructors may reenter close. No native mutex may span their execution.
        drop(owners);
        Ok(())
    }
}

#[pymethods]
impl ResidentRepeat {
    #[classattr]
    fn supports_phases() -> bool {
        true
    }

    #[classattr]
    fn supports_reusable_input() -> bool {
        true
    }

    #[classattr]
    fn supports_event_input() -> bool {
        true
    }

    #[new]
    #[pyo3(signature = (source_graph, index, device, steps, owners, *, prefix_graph=0, suffix_graph=0))]
    fn new(
        py: Python<'_>,
        source_graph: usize,
        index: usize,
        device: i32,
        steps: u32,
        owners: Vec<Py<PyAny>>,
        prefix_graph: usize,
        suffix_graph: usize,
    ) -> PyResult<Self> {
        if source_graph == 0 || index == 0 || steps == 0 {
            return Err(PyRuntimeError::new_err(
                "nonempty owned graph, GPU index and positive repeat count required",
            ));
        }
        let (pointer, code) = py.detach(|| {
            let mut pointer = std::ptr::null_mut();
            // SAFETY: private bridge retains every supplied Torch-owned graph definition
            // and the int64 device tensor; zero denotes an absent optional phase.
            let code = unsafe {
                cozy_repeat_create(
                    source_graph,
                    index,
                    device,
                    steps,
                    prefix_graph,
                    suffix_graph,
                    &mut pointer,
                )
            };
            (pointer as usize, code)
        });
        if code != 0 {
            let inspection = if code == -100 && pointer != 0 {
                // SAFETY: create returned this live partial Repeat. Its fixed, NUL-terminated
                // diagnostic contains no pointers and is copied before any owner cleanup.
                Some(unsafe {
                    CStr::from_ptr(cozy_repeat_inspection(pointer as *mut c_void))
                        .to_string_lossy()
                        .into_owned()
                })
            } else {
                None
            };
            // SAFETY: the helper returns its partial owner even on failure.
            if pointer != 0 && unsafe { cozy_repeat_destroy(pointer as *mut c_void) } != 0 {
                // Unknown device completion: retain every Python/Torch storage owner until process exit.
                std::mem::forget(owners);
            }
            return Err(match inspection.filter(|detail| !detail.is_empty()) {
                Some(detail) => ineligible(py, format!("prepared graph ineligible: {detail}")),
                None => cuda_error(py, "prepare resident repeat", code),
            });
        }
        // SAFETY: the successfully constructed handle remains exclusively owned here.
        let nodes = unsafe { cozy_repeat_nodes(pointer as *mut c_void) };
        // SAFETY: successful preparation inspected this same live native body recursively.
        let validated_nodes = unsafe { cozy_repeat_validated_nodes(pointer as *mut c_void) };
        // SAFETY: the same successfully constructed owner inspected these optional phases.
        let prefix_nodes = unsafe { cozy_repeat_prefix_nodes(pointer as *mut c_void) };
        // SAFETY: the same successfully constructed owner inspected these optional phases.
        let suffix_nodes = unsafe { cozy_repeat_suffix_nodes(pointer as *mut c_void) };
        Ok(Self {
            inner: Arc::new(RepeatOwner {
                state: Mutex::new(State {
                    pointer,
                    launched: false,
                    failed: false,
                    epoch: 0,
                    launches: 0,
                    pending: None,
                    destroy_failed: false,
                    streamed: false,
                }),
                owners: Mutex::new(owners),
                pid: std::process::id(),
                nodes,
                validated_nodes,
                prefix_nodes,
                suffix_nodes,
                has_prefix: prefix_graph != 0,
                has_suffix: suffix_graph != 0,
                device,
                steps,
            }),
        })
    }

    #[getter]
    fn body_nodes(&self) -> u32 {
        self.nodes
    }

    fn facts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.check_process()?;
        let (launches, closed, input) = {
            let state = self
                .state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("native owner poisoned"))?;
            let mut input = (0, 0, 0);
            if state.pointer != 0 {
                // SAFETY: this is a host-only snapshot under the handle's lifetime lock.
                unsafe {
                    cozy_repeat_input_status(
                        state.pointer as *mut c_void,
                        &mut input.0,
                        &mut input.1,
                        &mut input.2,
                    );
                }
            }
            (state.launches, state.pointer == 0, input)
        };
        // These facts originate in the actual constructed native handle, never a protocol match.
        let retained_owners = self
            .owners
            .lock()
            .map_err(|_| PyRuntimeError::new_err("prepared owners poisoned"))?
            .len();
        let result = PyDict::new(py);
        result.set_item("execution_mode", "cuda_conditional_while")?;
        result.set_item("body_nodes", self.nodes)?;
        result.set_item("validated_nodes", self.validated_nodes)?;
        result.set_item("prefix_nodes", self.prefix_nodes)?;
        result.set_item("suffix_nodes", self.suffix_nodes)?;
        result.set_item("has_prefix", self.has_prefix)?;
        result.set_item("has_suffix", self.has_suffix)?;
        result.set_item("device", self.device)?;
        result.set_item("steps_per_launch", self.steps)?;
        result.set_item("host_graph_launches", launches)?;
        result.set_item("retained_owner_count", retained_owners)?;
        result.set_item("control_kernels_preloaded", true)?;
        result.set_item("continuation_abi", 1)?;
        result.set_item("event_input_supported", true)?;
        result.set_item("body_entry_checked", input.2 != 0)?;
        result.set_item("body_entered", (input.2 != 0).then_some(input.0 != 0))?;
        result.set_item("input_invalid", (input.2 != 0).then_some(input.1 != 0))?;
        result.set_item("closed", closed)?;
        Ok(result)
    }

    fn launch(&self, py: Python<'_>, producer_stream: usize) -> PyResult<u64> {
        self.native(py, "launch resident repeat", None, |state| {
            if state.pending.is_some() {
                return Err(-4);
            }
            // SAFETY: exclusive handle, source owners retained; producer stream belongs to this device.
            match unsafe { cozy_repeat_launch(state.pointer as *mut c_void, producer_stream) } {
                0 => {
                    state.launched = true;
                    state.epoch += 1;
                    state.launches += 1;
                    Ok(state.epoch)
                }
                code => Err(code),
            }
        })
    }

    #[pyo3(signature = (producer_stream, *, reusable = false))]
    fn continuation(
        &self,
        py: Python<'_>,
        producer_stream: usize,
        reusable: bool,
    ) -> PyResult<(Py<PyAny>, u64)> {
        continuation::export(py, self.inner.clone(), producer_stream, reusable)
    }

    fn poll(&self, py: Python<'_>, epoch: u64) -> PyResult<(u32, bool, bool)> {
        self.native(py, "observe resident repeat", Some(epoch), |state| {
            if !state.launched {
                return Err(-3);
            }
            let (mut completed, mut canceled, mut done) = (0, 0, 0);
            // SAFETY: valid owner and stack outputs; snapshot reads GPU atomics on its own stream.
            match unsafe {
                cozy_repeat_poll(
                    state.pointer as *mut c_void,
                    &mut completed,
                    &mut canceled,
                    &mut done,
                )
            } {
                0 => Ok((completed, canceled != 0, done != 0)),
                code => Err(code),
            }
        })
    }

    fn cancel(&self, py: Python<'_>, epoch: u64) -> PyResult<()> {
        self.native(py, "cancel resident repeat", Some(epoch), |state| {
            if !state.launched {
                return Err(-3);
            }
            // SAFETY: cancellation is an atomic GPU write on a separate control stream.
            match unsafe { cozy_repeat_cancel(state.pointer as *mut c_void) } {
                0 => Ok(()),
                code => Err(code),
            }
        })
    }

    fn wait(&self, py: Python<'_>, epoch: u64) -> PyResult<(u32, bool)> {
        loop {
            let ready = self.native(py, "observe resident completion", Some(epoch), |state| {
                if !state.launched {
                    return Err(-3);
                }
                let mut done = 0;
                // SAFETY: event query only; no launch and no mutex held while parked below.
                match unsafe { cozy_repeat_ready(state.pointer as *mut c_void, &mut done) } {
                    0 => Ok(done != 0),
                    code => Err(code),
                }
            })?;
            if ready {
                break;
            }
            py.detach(|| std::thread::sleep(std::time::Duration::from_millis(1)));
        }
        self.native(py, "wait for resident repeat", Some(epoch), |state| {
            if !state.launched {
                return Err(-3);
            }
            let (mut completed, mut canceled) = (0, 0);
            // SAFETY: waits for GPU completion; never submits another iteration.
            match unsafe {
                cozy_repeat_wait(state.pointer as *mut c_void, &mut completed, &mut canceled)
            } {
                0 => Ok((completed, canceled != 0)),
                code => Err(code),
            }
        })
    }

    fn close(&self, py: Python<'_>, epoch: u64) -> PyResult<()> {
        self.dispose(py, Some(epoch))
    }

    fn close_owner(&self, py: Python<'_>) -> PyResult<()> {
        self.dispose(py, None)
    }
}

impl Drop for RepeatOwner {
    fn drop(&mut self) {
        if self.pid != std::process::id() {
            // Never acquire inherited CUDA/mutex state, or destroy inherited Torch owners after fork.
            std::mem::forget(std::mem::take(
                self.owners
                    .get_mut()
                    .unwrap_or_else(|error| error.into_inner()),
            ));
            return;
        }
        let Ok(state) = self.state.get_mut() else {
            std::mem::forget(std::mem::take(
                self.owners
                    .get_mut()
                    .unwrap_or_else(|error| error.into_inner()),
            ));
            return;
        };
        // SAFETY: the last Python owner implies no method can concurrently touch the handle.
        if state.destroy_failed
            || (state.pointer != 0
                && unsafe { cozy_repeat_destroy(state.pointer as *mut c_void) } != 0)
        {
            std::mem::forget(std::mem::take(
                self.owners
                    .get_mut()
                    .unwrap_or_else(|error| error.into_inner()),
            ));
        }
    }
}

#[pymodule]
fn _execution(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "NativeExecutionError",
        module.py().get_type::<NativeExecutionError>(),
    )?;
    module.add_class::<ResidentRepeat>()?;
    module.add_class::<sequence::NativeSequence>()?;
    Ok(())
}
