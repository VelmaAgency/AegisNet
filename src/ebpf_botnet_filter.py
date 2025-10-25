# ebpf_botnet_filter.py - Optimized eBPF Botnet Filters for AegisNet v2.1.1
from bcc import BPF
import logging
from typing import List
import ctypes as ct

logger = logging.getLogger(__name__)

# Optimized eBPF code with XDP and ring buffer
EBPF_CODE = """
#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>
#include <linux/bpf.h>

BPF_HASH(blocked_ips, u32, u32);
BPF_RINGBUF(blocked_events, 256 * 1024);

struct event_t {
    u32 ip;
};

int xdp_filter(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;
    struct ethhdr *eth = data;
    if (data + sizeof(*eth) > data_end)
        return XDP_PASS;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;

    u32 src_ip = ip->saddr;
    u32 *val = blocked_ips.lookup(&src_ip);
    if (val) {
        struct event_t event = {.ip = src_ip};
        blocked_events.ringbuf_output(&event, sizeof(event), 0);
        return XDP_DROP;
    }
    return XDP_PASS;
}
"""

class eBPFBotnetFilter:
    """Optimized eBPF filters for botnet traffic."""
    def __init__(self):
        try:
            self.bpf = BPF(text=EBPF_CODE)
            self.blocked_ips = self.bpf["blocked_ips"]
            self.ringbuf = self.bpf["blocked_events"]
            self.ringbuf.open_ring_buffer(self._handle_event)
            logger.info("eBPF filter initialized")
        except Exception as e:
            logger.error(f"eBPF init error: {e}")

    def _handle_event(self, ctx, data, size):
        """Handle ring buffer events."""
        event = ct.cast(data, ct.POINTER(ct.c_uint32)).contents
        logger.info("Blocked botnet IP", extra={"ip": event.value})

    def add_blocked_ips(self, ips: List[str]):
        """Batch-add IPs to blocked map."""
        try:
            for ip in ips:
                ip_int = ct.c_uint32(int.from_bytes(socket.inet_aton(ip), "big"))
                self.blocked_ips[ip_int] = ct.c_uint32(1)
            logger.info("Blocked IPs added", extra={"ips": ips})
        except Exception as e:
            logger.error(f"Block IPs error: {e}")

    def filter_botnet_traffic(self):
        """Run optimized eBPF filter."""
        try:
            self.bpf.attach_xdp("eth0", fn_name="xdp_filter")
            self.ringbuf.poll()
        except KeyboardInterrupt:
            logger.info("eBPF filter stopped")
        except Exception as e:
            logger.error(f"eBPF run error: {e}")

# Example usage
if __name__ == "__main__":
    filter = eBPFBotnetFilter()
    filter.add_blocked_ips(["192.168.1.100", "192.168.1.101"])
    filter.filter_botnet_traffic()