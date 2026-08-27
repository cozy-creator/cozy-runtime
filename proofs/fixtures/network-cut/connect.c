#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

static int (*real_connect)(int, const struct sockaddr *, socklen_t);

int connect(int fd, const struct sockaddr *addr, socklen_t len) {
    if (!real_connect) real_connect = dlsym(RTLD_NEXT, "connect");
    if (addr && (addr->sa_family == AF_INET || addr->sa_family == AF_INET6)) {
        const char *log = getenv("COZY_NETWORK_ATTEMPT_LOG");
        if (log) {
            int marker = open(log, O_WRONLY | O_CREAT | O_APPEND, 0600);
            if (marker >= 0) {
                ssize_t written = write(marker, "1", 1);
                (void)written;
                (void)close(marker);
            }
        }
        errno = ECONNREFUSED;
        return -1;
    }
    return real_connect(fd, addr, len);
}
