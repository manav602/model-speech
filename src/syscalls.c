// Newlib syscall stubs that route printf through ARM semihosting,
// so QEMU prints to host stdout. _sbrk for malloc against the linker heap.
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>

#define SYS_WRITE0  0x04
#define SYS_WRITE   0x05

static inline int semihost_call(int op, void* arg) {
    register int r0 asm("r0") = op;
    register void* r1 asm("r1") = arg;
    asm volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

int _write(int fd, const char* buf, int len) {
    (void)fd;
    uint32_t args[3] = {1 /*stdout*/, (uint32_t)(uintptr_t)buf, (uint32_t)len};
    semihost_call(SYS_WRITE, args);
    return len;
}

extern char _heap_start, _heap_end;
static char* heap_ptr = &_heap_start;

caddr_t _sbrk(int incr) {
    char* prev = heap_ptr;
    if (heap_ptr + incr > &_heap_end) {
        errno = ENOMEM;
        return (caddr_t)-1;
    }
    heap_ptr += incr;
    return (caddr_t)prev;
}

int _close(int fd) { (void)fd; return -1; }
int _fstat(int fd, struct stat* st) { (void)fd; st->st_mode = S_IFCHR; return 0; }
int _isatty(int fd) { (void)fd; return 1; }
off_t _lseek(int fd, off_t off, int whence) { (void)fd; (void)off; (void)whence; return 0; }
int _read(int fd, char* buf, int len) { (void)fd; (void)buf; (void)len; return 0; }
int _kill(int pid, int sig) { (void)pid; (void)sig; return -1; }
int _getpid(void) { return 1; }
void _exit(int rc) { (void)rc; while (1); }
