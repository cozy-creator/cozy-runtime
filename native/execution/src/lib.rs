//! One resident graph definition, one GPU-driven loop, and its complete owner lifetime.
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::{ffi::c_void, sync::Mutex};

pyo3::create_exception!(_execution, NativeExecutionError, PyRuntimeError);

unsafe extern "C" {
    fn cozy_repeat_create(source: usize, index: usize, device: i32, steps: u32, out: *mut *mut c_void) -> i32;
    fn cozy_repeat_launch(run: *mut c_void, producer: usize) -> i32;
    fn cozy_repeat_poll(run: *mut c_void, completed: *mut u32, canceled: *mut u32, done: *mut i32) -> i32;
    fn cozy_repeat_cancel(run: *mut c_void) -> i32;
    fn cozy_repeat_wait(run: *mut c_void, completed: *mut u32, canceled: *mut u32) -> i32;
    fn cozy_repeat_nodes(run: *mut c_void) -> u32;
    fn cozy_repeat_destroy(run: *mut c_void) -> i32;
}

fn cuda_error(py: Python<'_>, operation: &str, code: i32) -> PyErr {
    let error = NativeExecutionError::new_err(format!("{operation}: CUDA error {code}"));
    let _ = error.value(py).setattr("error_code", code);
    error
}

struct State { pointer: usize, launched: bool, failed: bool }

#[pyclass(name = "ResidentRepeat", module = "cozy_runtime._kernels._execution", frozen)]
struct ResidentRepeat {
    state: Mutex<State>,
    owners: Vec<Py<PyAny>>,
    pid: u32,
    nodes: u32,
}

impl ResidentRepeat {
    fn check_process(&self) -> PyResult<()> {
        if self.pid != std::process::id() {
            return Err(PyRuntimeError::new_err("prepared GPU work cannot be used after fork"));
        }
        Ok(())
    }

    // The entire mutex lifetime is inside detach: no lock is held while reacquiring the GIL.
    fn native<T: Send>(&self, py: Python<'_>, operation: &str, call: impl FnOnce(&mut State) -> Result<T, i32> + Send) -> PyResult<T> {
        self.check_process()?;
        py.detach(|| {
            let mut state = self.state.lock().map_err(|_| 999)?;
            if state.pointer == 0 || state.failed { return Err(999); }
            let result = call(&mut state);
            if result.is_err() { state.failed = true; }
            result
        }).map_err(|code| cuda_error(py, operation, code))
    }
}

#[pymethods]
impl ResidentRepeat {
    #[new]
    fn new(py: Python<'_>, source_graph: usize, index: usize, device: i32, steps: u32, owners: Vec<Py<PyAny>>) -> PyResult<Self> {
        if source_graph == 0 || index == 0 || steps == 0 {
            return Err(PyRuntimeError::new_err("nonempty owned graph, GPU index and positive repeat count required"));
        }
        let (pointer, code) = py.detach(|| {
            let mut pointer = std::ptr::null_mut();
            // SAFETY: private bridge passes a live Torch-owned graph definition and int64 device tensor.
            let code = unsafe { cozy_repeat_create(source_graph, index, device, steps, &mut pointer) };
            (pointer as usize, code)
        });
        if code != 0 {
            // SAFETY: the helper returns its partial owner even on failure.
            if pointer != 0 && unsafe { cozy_repeat_destroy(pointer as *mut c_void) } != 0 {
                // Unknown device completion: retain every Python/Torch storage owner until process exit.
                std::mem::forget(owners);
            }
            return Err(cuda_error(py, "prepare resident repeat", code));
        }
        // SAFETY: the successfully constructed handle remains exclusively owned here.
        let nodes = unsafe { cozy_repeat_nodes(pointer as *mut c_void) };
        Ok(Self { state: Mutex::new(State { pointer, launched: false, failed: false }), owners, pid: std::process::id(), nodes })
    }

    #[getter]
    fn body_nodes(&self) -> u32 { self.nodes }

    fn launch(&self, py: Python<'_>, producer_stream: usize) -> PyResult<()> {
        self.native(py, "launch resident repeat", |state| {
            if state.launched { return Err(1); }
            state.launched = true;
            // SAFETY: exclusive handle, source owners retained; producer stream belongs to this device.
            match unsafe { cozy_repeat_launch(state.pointer as *mut c_void, producer_stream) } {
                0 => Ok(()), code => Err(code),
            }
        })
    }

    fn poll(&self, py: Python<'_>) -> PyResult<(u32, bool, bool)> {
        self.native(py, "observe resident repeat", |state| {
            if !state.launched { return Err(1); }
            let (mut completed, mut canceled, mut done) = (0, 0, 0);
            // SAFETY: valid owner and stack outputs; snapshot reads GPU atomics on its own stream.
            match unsafe { cozy_repeat_poll(state.pointer as *mut c_void, &mut completed, &mut canceled, &mut done) } {
                0 => Ok((completed, canceled != 0, done != 0)), code => Err(code),
            }
        })
    }

    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        self.native(py, "cancel resident repeat", |state| {
            // SAFETY: cancellation is an atomic GPU write on a separate control stream.
            match unsafe { cozy_repeat_cancel(state.pointer as *mut c_void) } { 0 => Ok(()), code => Err(code) }
        })
    }

    fn wait(&self, py: Python<'_>) -> PyResult<(u32, bool)> {
        self.native(py, "wait for resident repeat", |state| {
            if !state.launched { return Err(1); }
            let (mut completed, mut canceled) = (0, 0);
            // SAFETY: waits for GPU completion; never submits another iteration.
            match unsafe { cozy_repeat_wait(state.pointer as *mut c_void, &mut completed, &mut canceled) } {
                0 => Ok((completed, canceled != 0)), code => Err(code),
            }
        })
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.check_process()?;
        py.detach(|| {
            let mut state = self.state.lock().map_err(|_| 999)?;
            if state.pointer == 0 { return Ok(()); }
            // SAFETY: destruction joins both streams before releasing graph/control storage.
            let code = unsafe { cozy_repeat_destroy(state.pointer as *mut c_void) };
            if code != 0 { state.failed = true; return Err(code); }
            state.pointer = 0;
            Ok(())
        }).map_err(|code| cuda_error(py, "close resident repeat", code))
    }
}

impl Drop for ResidentRepeat {
    fn drop(&mut self) {
        if self.pid != std::process::id() {
            // Never acquire inherited CUDA/mutex state, or destroy inherited Torch owners after fork.
            std::mem::forget(std::mem::take(&mut self.owners));
            return;
        }
        let Ok(state) = self.state.get_mut() else {
            std::mem::forget(std::mem::take(&mut self.owners));
            return;
        };
        // SAFETY: the last Python owner implies no method can concurrently touch the handle.
        if state.pointer != 0 && unsafe { cozy_repeat_destroy(state.pointer as *mut c_void) } != 0 {
            std::mem::forget(std::mem::take(&mut self.owners));
        }
    }
}

#[pymodule]
fn _execution(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("NativeExecutionError", module.py().get_type::<NativeExecutionError>())?;
    module.add_class::<ResidentRepeat>()?;
    Ok(())
}
