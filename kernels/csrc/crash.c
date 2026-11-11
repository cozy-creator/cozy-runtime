// SPDX-License-Identifier: Apache-2.0
// The executor's record of a fatal signal (cr-275). On SIGSEGV, SIGBUS, SIGABRT, SIGFPE or SIGILL it writes to stderr,
// which the machine keeps with the run: the crash line, every other thread's native stack (each thread answers a
// real-time signal into a static slot; identical stacks print once), every Python stack (faulthandler, the handler before
// this one), then the crashed thread's native stack, what the executor was answering and the crash line again, last, so
// the run's detail ends on them. Then the process dies by the same signal: its exit status and core are the kernel's.
// Handlers make async-signal-safe calls only; backtrace() is warmed at install, so the dlopen of its first call never
// runs in one. A record that hangs ends at the watchdog, by the same signal.
#define _GNU_SOURCE 1
#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x030C0000
#endif
#include <Python.h>

#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

enum { FRAMES = 64, SHOWN = 20, THREADS = 512, STACK = 256 << 10, WAIT_MS = 1000, WATCHDOG_S = 20, NOTE = 240 };

static const int FATAL[] = {SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL};
static struct sigaction previous[NSIG];
static int dump_signal;
static char exe[256];
static void *libc_base, *self_base;

// One thread's stack, from the frame it was interrupted in; `sig` is nonzero for a thread that crashed too.
typedef struct {
    atomic_int ready;
    atomic_int tid;
    int sig, n;
    void *pc[FRAMES];
} Stack;
static Stack seen[THREADS];
static int asked[THREADS];
static unsigned char printed[THREADS];
static void *frames[FRAMES + 8];
static atomic_int taken, answered, crasher, crash_signal;
static char notes[2][NOTE];
static atomic_int note_at;

static void put(const char *s, size_t n) {
    while (n > 0) {
        ssize_t w = write(STDERR_FILENO, s, n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return;
        s += w;
        n -= (size_t)w;
    }
}

static void say(const char *s) { put(s, strlen(s)); }

static size_t digits(char *out, uintptr_t v, unsigned base) {
    char b[24];
    size_t i = sizeof b, n = 0;
    do b[--i] = "0123456789abcdef"[v % base];
    while ((v /= base) != 0);
    if (base == 16) out[n++] = '0', out[n++] = 'x';
    while (i < sizeof b) out[n++] = b[i++];
    return n;
}

static void number(uintptr_t v, unsigned base) {
    char b[24];
    put(b, digits(b, v, base));
}

static const char *signame(int sig) {
    switch (sig) {
    case SIGSEGV: return "SIGSEGV";
    case SIGBUS: return "SIGBUS";
    case SIGABRT: return "SIGABRT";
    case SIGFPE: return "SIGFPE";
    case SIGILL: return "SIGILL";
    }
    return "a fatal signal";
}

static const char *cause(int sig, int code) {
    switch (sig) {
    case SIGSEGV: return code == SEGV_MAPERR ? "address not mapped" : code == SEGV_ACCERR ? "access not permitted" : "bad access";
    case SIGBUS: return code == BUS_ADRALN ? "misaligned" : code == BUS_ADRERR ? "past the end of a mapped file" : "bus error";
    case SIGFPE: return code == FPE_INTDIV ? "integer divide by zero" : "arithmetic fault";
    case SIGILL: return "illegal instruction";
    }
    return "";
}

// `<module>+0x<offset>`, and ` (<symbol>+0x<offset>)` when the module exports a symbol around it.
static void name(void *pc) {
    Dl_info dl;
    if (!dladdr(pc, &dl) || !dl.dli_fbase) {
        number((uintptr_t)pc, 16);
        return;
    }
    const char *path = dl.dli_fname && *dl.dli_fname ? dl.dli_fname : exe;
    const char *slash = strrchr(path, '/');
    say(slash ? slash + 1 : path);
    say("+");
    number((uintptr_t)pc - (uintptr_t)dl.dli_fbase, 16);
    if (dl.dli_sname && dl.dli_saddr) {
        say(" (");
        say(dl.dli_sname);
        say("+");
        number((uintptr_t)pc - (uintptr_t)dl.dli_saddr, 16);
        say(")");
    }
}

static void thread_name(int tid) {
    char path[64] = "/proc/self/task/", comm[32];
    size_t n = strlen(path);
    n += digits(path + n, (uintptr_t)tid, 10);
    memcpy(path + n, "/comm", 6);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    ssize_t got = read(fd, comm, sizeof comm);
    close(fd);
    if (got > 0 && comm[got - 1] == '\n') got--;
    if (got <= 0) return;
    say(" ");
    put(comm, (size_t)got);
}

static void *pc_of(void *raw) {
    ucontext_t *uc = raw;
#if defined(__x86_64__)
    return (void *)uc->uc_mcontext.gregs[REG_RIP];
#elif defined(__aarch64__)
    return (void *)uc->uc_mcontext.pc;
#else
    (void)uc;
    return NULL;
#endif
}

// The index of the interrupted frame: the handler's own frames and the kernel's trampoline come before it.
static int from(void **pcs, int n, void *pc) {
    for (int i = 0; i < n; i++)
        if (pcs[i] == pc) return i;
    Dl_info dl;
    int i = 0;
    while (i < n && dladdr(pcs[i], &dl) && dl.dli_fbase == self_base) i++;
    return i + 1 < n ? i + 1 : i;
}

// This thread's stack into the next slot. `sig`: it crashed too.
static void record(int sig, void *pc) {
    int slot = atomic_fetch_add(&taken, 1);
    if (slot < THREADS) {
        Stack *s = &seen[slot];
        atomic_store(&s->tid, (int)syscall(SYS_gettid));
        void *pcs[FRAMES + 8];
        int n = backtrace(pcs, FRAMES + 8), start = from(pcs, n, pc);
        s->n = n - start > FRAMES ? FRAMES : n - start;
        memcpy(s->pc, pcs + start, sizeof(void *) * (size_t)s->n);
        s->sig = sig;
        atomic_store(&s->ready, 1);
    }
    atomic_fetch_add(&answered, 1);
}

static void on_dump(int sig, siginfo_t *info, void *raw) {
    (void)sig;
    (void)info;
    int saved = errno;
    record(0, pc_of(raw));
    errno = saved;
}

// Every other thread asked for its stack; how many were asked.
static int ask(int self) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_dump;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    if (sigaction(dump_signal, &sa, NULL) != 0) return 0;
    int fd = open("/proc/self/task", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return 0;
    int count = 0, pid = getpid();
    char buf[4096] __attribute__((aligned(8)));
    long got;
    while ((got = syscall(SYS_getdents64, fd, buf, sizeof buf)) > 0) {
        for (long at = 0; at < got;) {
            struct {
                uint64_t ino;
                int64_t off;
                unsigned short reclen;
                unsigned char type;
                char name[];
            } *d = (void *)(buf + at);
            at += d->reclen;
            int tid = 0;
            for (const char *c = d->name; *c >= '0' && *c <= '9'; c++) tid = tid * 10 + (*c - '0');
            if (tid > 0 && tid != self && count < THREADS && syscall(SYS_tgkill, pid, tid, dump_signal) == 0)
                asked[count++] = tid;
        }
    }
    close(fd);
    return count;
}

static int same(const Stack *a, const Stack *b) {
    return a->sig == b->sig && a->n == b->n && memcmp(a->pc, b->pc, sizeof(void *) * (size_t)a->n) == 0;
}

// The answers, identical stacks once, then the threads that gave none (they block the signal: driver threads do).
static void others(int count) {
    struct timespec tick = {0, 1000000};
    for (int i = 0; i < WAIT_MS && atomic_load(&answered) < count; i++) nanosleep(&tick, NULL);
    int n = atomic_load(&taken);
    if (n > THREADS) n = THREADS;
    say("the other threads, most recent call first:\n");
    for (int i = 0; i < n; i++) {
        Stack *s = &seen[i];
        if (printed[i] || !atomic_load(&s->ready)) continue;
        int group = 0;
        for (int j = i; j < n; j++) {
            if (printed[j] || !atomic_load(&seen[j].ready) || !same(s, &seen[j])) continue;
            printed[j] = 1;
            if (group < 8) {
                say(group ? ", " : "thread ");
                number((uintptr_t)atomic_load(&seen[j].tid), 10);
                thread_name(atomic_load(&seen[j].tid));
            }
            group++;
        }
        if (group > 8) {
            say(" and ");
            number((uintptr_t)(group - 8), 10);
            say(" more");
        }
        if (s->sig) {
            say(", which crashed too (");
            say(signame(s->sig));
            say(")");
        }
        say(":\n");
        for (int k = 0; k < s->n && k < SHOWN; k++) {
            say("  #");
            number((uintptr_t)k, 10);
            say(" ");
            name(s->pc[k]);
            say("\n");
        }
    }
    int silent = 0;
    for (int a = 0; a < count; a++) {
        int found = 0;
        for (int i = 0; i < n && !found; i++) found = atomic_load(&seen[i].ready) && atomic_load(&seen[i].tid) == asked[a];
        if (found) continue;
        say(silent++ ? ", " : "no answer from thread ");
        number((uintptr_t)asked[a], 10);
        thread_name(asked[a]);
    }
    if (silent) say("\n");
}

// The previous handler: faulthandler, whose Python stacks follow. It restores its own previous handler and raises the
// signal, which stays pending while this handler runs.
static void chain(int sig, siginfo_t *info, void *raw) {
    const struct sigaction *old = &previous[sig];
    if (old->sa_flags & SA_SIGINFO) {
        if (old->sa_sigaction) old->sa_sigaction(sig, info, raw);
    } else if (old->sa_handler != SIG_DFL && old->sa_handler != SIG_IGN) {
        old->sa_handler(sig);
    }
}

static void line(int sig, const siginfo_t *info, void *frame, void *pc) {
    say("executor crashed: ");
    say(signame(sig));
    say(" in ");
    name(frame);
    if (info->si_code <= 0 && info->si_pid == getpid()) {
        say("\n  raised by this process");
    } else if (info->si_code <= 0) {
        say("\n  sent by process ");
        number((uintptr_t)info->si_pid, 10);
    } else {
        say("\n  address ");
        number((uintptr_t)info->si_addr, 16);
        say(" (");
        say(cause(sig, info->si_code));
        say(")");
    }
    say(", pc ");
    number((uintptr_t)pc, 16);
    say("\n");
}

static void die(int sig) {
    struct sigaction dfl;
    memset(&dfl, 0, sizeof dfl);
    dfl.sa_handler = SIG_DFL;
    sigemptyset(&dfl.sa_mask);
    sigaction(sig, &dfl, NULL);
    // Pending until this handler returns: then the process ends by `sig`, as it would have without a handler.
    raise(sig);
}

static void on_watchdog(int unused) {
    (void)unused;
    int sig = atomic_load(&crash_signal);
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, sig);
    die(sig);
    sigprocmask(SIG_UNBLOCK, &set, NULL);
}

static void on_fatal(int sig, siginfo_t *info, void *raw) {
    int self = (int)syscall(SYS_gettid), first = 0;
    void *pc = pc_of(raw);
    if (!atomic_compare_exchange_strong(&crasher, &first, self)) {
        // Another thread writes the record; this one joins it and waits for the process to end. One that faulted while
        // answering has a slot already: its stack can't be taken. A fault in the record itself never comes here: every
        // fatal signal is blocked while it is written, so the kernel ends the process by it.
        int n = atomic_load(&taken);
        for (int i = 0; i < n && i < THREADS; i++) {
            if (atomic_load(&seen[i].tid) == self && !atomic_load(&seen[i].ready)) {
                seen[i].sig = sig;
                atomic_store(&seen[i].ready, 1);
                atomic_fetch_add(&answered, 1);
                for (;;) pause();
            }
        }
        record(sig, pc);
        for (;;) pause();
    }
    struct sigaction watch;
    memset(&watch, 0, sizeof watch);
    watch.sa_handler = on_watchdog;
    sigemptyset(&watch.sa_mask);
    atomic_store(&crash_signal, sig);
    sigaction(SIGALRM, &watch, NULL);
    alarm(WATCHDOG_S);
    line(sig, info, pc, pc);  // first, so a fault while unwinding still leaves it
    int n = backtrace(frames, FRAMES + 8), start = from(frames, n, pc);
    void **own = frames + start;
    n = n - start > FRAMES ? FRAMES : n - start;
    others(ask(self));
    chain(sig, info, raw);
    say("thread ");
    number((uintptr_t)self, 10);
    thread_name(self);
    say(" crashed, most recent call last:\n");
    for (int k = n - 1; k >= 0; k--) {
        say("  #");
        number((uintptr_t)k, 10);
        say(" ");
        name(own[k]);
        say("\n");
    }
    const char *note = notes[atomic_load(&note_at)];
    if (*note) {
        say("  answering: ");
        say(note);
        say("\n");
    }
    // An abort names the first frame outside the C library: its caller, not abort() itself.
    void *frame = pc;
    Dl_info dl;
    for (int k = 0; sig == SIGABRT && k < n; k++) {
        if (!dladdr(own[k], &dl) || dl.dli_fbase != libc_base) {
            frame = own[k];
            break;
        }
    }
    line(sig, info, frame, pc);
    die(sig);
}

// An alternate signal stack for the calling thread, so a stack overflow still leaves a record.
static void arm_thread(void) {
    stack_t now;
    if (sigaltstack(NULL, &now) == 0 && !(now.ss_flags & SS_DISABLE) && now.ss_size >= STACK) return;
    void *stack = mmap(NULL, STACK, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_STACK, -1, 0);
    if (stack == MAP_FAILED) return;
    stack_t armed = {.ss_sp = stack, .ss_size = STACK, .ss_flags = 0};
    if (sigaltstack(&armed, NULL) != 0) munmap(stack, STACK);
}

static PyObject *arm(PyObject *module, PyObject *unused) {
    (void)module;
    (void)unused;
    arm_thread();
    Py_RETURN_NONE;
}

static PyObject *install(PyObject *module, PyObject *unused) {
    (void)module;
    (void)unused;
    void *warm[2];
    backtrace(warm, 2);
    Dl_info dl;
    if (dladdr((void *)&abort, &dl)) libc_base = dl.dli_fbase;
    if (dladdr((void *)&on_fatal, &dl)) self_base = dl.dli_fbase;
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    exe[n > 0 ? n : 0] = 0;
    dump_signal = SIGRTMIN + 6 <= SIGRTMAX ? SIGRTMIN + 6 : SIGRTMAX;
    arm_thread();
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_fatal;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    for (size_t i = 0; i < sizeof FATAL / sizeof *FATAL; i++) sigaddset(&sa.sa_mask, FATAL[i]);
    sigaddset(&sa.sa_mask, dump_signal);
    for (size_t i = 0; i < sizeof FATAL / sizeof *FATAL; i++) {
        struct sigaction now;
        if (sigaction(FATAL[i], NULL, &now) == 0 && (now.sa_flags & SA_SIGINFO) && now.sa_sigaction == on_fatal) continue;
        if (sigaction(FATAL[i], &sa, &previous[FATAL[i]]) != 0) return PyErr_SetFromErrno(PyExc_OSError);
    }
    Py_RETURN_NONE;
}

static PyObject *note(PyObject *module, PyObject *text) {
    (void)module;
    Py_ssize_t size;
    const char *utf8 = PyUnicode_AsUTF8AndSize(text, &size);
    if (!utf8) return NULL;
    int next = 1 - atomic_load(&note_at);
    size_t n = (size_t)size < NOTE - 1 ? (size_t)size : NOTE - 1;
    memcpy(notes[next], utf8, n);
    notes[next][n] = 0;
    atomic_store(&note_at, next);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"install", install, METH_NOARGS,
     "Install the record ahead of the current handlers (faulthandler's), which it chains to; arm this thread."},
    {"arm", arm, METH_NOARGS, "Give the calling thread an alternate signal stack."},
    {"note", note, METH_O, "What the process answers now, for the record."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, "_crash", "The executor's record of a fatal signal (cr-275).", -1, methods, NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__crash(void) { return PyModule_Create(&definition); }
