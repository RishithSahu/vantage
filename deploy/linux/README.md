# Hardened assistant unit

`vgpt-assistant.service.in` is the assistant's systemd unit with every sandbox property that `tools_isolation_rig.py` tested on a Raspberry Pi 4 (cgroup v2, seccomp, private network / pids / tmp / devices, read-only file system, no capabilities, no real-time priority, memory, task and CPU limits, the cores it may use).

What was measured with it (results in `results/isolation/`, discussion in `docs/SAFETY_ISOLATION.md`):

* 22 of 22 escape attempts refused (network, raw and CAN sockets, writes to the vehicle files, seeing or signalling the safety process, real-time priority, moving onto the reserved core, raising its own limits, mount, reboot, kernel module, clock, namespaces, a child process opening a socket);
* a 100 Hz real-time loop on its own core never missed its 1 ms deadline in 10 scenarios (idle, answering at full speed, every allowed core at 100 %, a CPU hog allowed onto the safety core, memory exhaustion, fork bomb, disk flood, repeated crashes, the attack list); worst wake-up delay 390 us.

To install: replace `@USER@`, `@APP_DIR@`, `@STATE_DIR@`, `@EXEC_START@`, set `AllowedCPUs` to the cores that are not the vehicle's, copy to `/etc/systemd/system/`, `systemctl enable --now vgpt-assistant`. To repeat the measurement: `python tools_isolation_rig.py --seconds 45` (needs passwordless sudo, creates only transient units and a scratch folder under /var/lib).

Limits of the claim: this is operating-system isolation on one host, not a hypervisor or hardware boundary, and it is not an ISO 26262 safety case.
