# THIS IS README DOCS FOR CHECK VERSION COMPATIBLE OF THIS LIB

**because this is my first project for libs for simulate numpy, for make AI run using C++**

## check compatible version

### LINUX SYSTEM

syntax:

```bash
# Basic check using /proc/cpuinfo (most reliable)
grep -i 'sse\|avx' /proc/cpuinfo

# Alternative using lscpu (more structured output)
lscpu | grep -i 'sse\|avx'
```

### MACOS SYSTEM

```bash
# Using sysctl (native macOS command)
sysctl -a | grep cpu.features

# OR for a more focused result
sysctl -a | grep -i 'sse\|avx'
```

### WINDOWS SYSTEM

syntax:

```powershell
# Using PowerShell (built-in)
Get-CimInstance -ClassName Win32_Processor | Select-Object Name, Caption, Description

# Using Command Prompt (built-in)
wmic cpu get name, caption
```
