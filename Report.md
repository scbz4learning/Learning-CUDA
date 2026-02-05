scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor$ cd codes/stage1/cuda/Learning-CUDA/
.git       LICENSE    Makefile   README.md  learning/  src/       tester/
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor$ cd codes/stage1/cuda/Learning-CUDA/
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ git pull
There is no tracking information for the current branch.
Please specify which branch you want to merge with.
See git-pull(1) for details.

    git pull <remote> <branch>

If you wish to set tracking information for this branch you can do so with:

    git branch --set-upstream-to=origin/<branch> 2025-winter

scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ git branch --set-upstream-to=origin/2025-winter 2025-winter
branch '2025-winter' set up to track 'origin/2025-winter'.
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ git pull
Already up to date.
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ gigit push
bash: gigit: command not found
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ git push
Everything up-to-date
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ git pull
remote: Enumerating objects: 4, done.
remote: Counting objects: 100% (4/4), done.
remote: Compressing objects: 100% (2/2), done.
remote: Total 4 (delta 2), reused 4 (delta 2), pack-reused 0 (from 0)
Unpacking objects: 100% (4/4), 754 bytes | 754.00 KiB/s, done.
From github.com:scbz4learning/Learning-CUDA
   0e78eb4..1906256  2025-winter -> origin/2025-winter
Updating 0e78eb4..1906256
Fast-forward
 src/kernels.cu | 23 +++++++++++++++++++++++
 1 file changed, 23 insertions(+)
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ export SKIP_ATTENTION=1
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
make: nvcc: No such file or directory
make: *** [Makefile:103: src/kernels.o] Error 127
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ ls
LICENSE  Makefile  README.md  learning  src  tester
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ srun bash
srun: Required node not available (down, drained or reserved)
srun: job 64 queued and waiting for resources
^Csrun: Job allocation 64 has been revoked
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ srun --gres=gpu:nvidia:1 --cpus-per-task=16 --mem=16G bash
srun: error: Unable to allocate resources: Requested node configuration is not available
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ srun bash
srun: Required node not available (down, drained or reserved)
srun: job 66 queued and waiting for resources
^Csrun: Job allocation 66 has been revoked
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ nvcc --version
bash: nvcc: command not found
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ srun make
srun: Required node not available (down, drained or reserved)
srun: job 67 queued and waiting for resources
^Csrun: Job allocation 67 has been revoked
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ ^C
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ nvidia-smi
Sun Feb  1 12:59:08 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.20             Driver Version: 570.133.20     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB          On  |   00000000:03:00.0 Off |                    0 |
| N/A   37C    P0             62W /  400W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A100-SXM4-80GB          On  |   00000000:04:00.0 Off |                    0 |
| N/A   35C    P0             68W /  400W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA A100-SXM4-80GB          On  |   00000000:09:00.0 Off |                    0 |
| N/A   36C    P0             70W /  400W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A100-SXM4-80GB          On  |   00000000:0A:00.0 Off |                    0 |
| N/A   38C    P0             64W /  400W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA A100-SXM4-80GB          On  |   00000000:10:00.0 Off |                    0 |
| N/A   36C    P0             65W /  400W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA A100-SXM4-80GB          On  |   00000000:11:00.0 Off |                    0 |
| N/A   35C    P0             66W /  400W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA A100-SXM4-80GB          On  |   00000000:16:00.0 Off |                    0 |
| N/A   35C    P0             69W /  400W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA A100-SXM4-80GB          On  |   00000000:17:00.0 Off |                    0 |
| N/A   38C    P0             68W /  400W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ vim ~/.bashrc 
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ source /data/shared/
InfiniTrain-dev/ Llama-3.2-1B/    miniconda3/      
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ source /data/shared/
InfiniTrain-dev/ Llama-3.2-1B/    miniconda3/      
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ source /data/shared/
InfiniTrain-dev/ Llama-3.2-1B/    miniconda3/      
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ source /data/shared/InfiniTrain-dev/
data/ logs/ 
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ source /data/shared/InfiniTrain-dev/data/llmc/
gpt2/   llama3/ 
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ source /data/shared/InfiniTrain-dev/data/llmc/
gpt2/   llama3/ 
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ vim ~/.bashrc 
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ ls /usr/local/
bin/       cuda-12/   etc/       include/   licensing/ mpi/       share/     ucx/       
cuda/      cuda-12.8/ games/     lib/       man/       sbin/      src/       
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ ls /usr/local/cuda-12
NsightSystems-cli-2024.6.2  bin  compat  compute-sanitizer  doc  extras  gds  include  lib64  nvml  nvvm  share  src  targets
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ ls /usr/local/cuda-12/bin/
__nvcc_device_query  crt               cuda-gdb-python3.10-tui  cuda-gdb-python3.8-tui  cudafe++   nsys          nvdisasm  nvprune
bin2c                cuda-gdb          cuda-gdb-python3.11-tui  cuda-gdb-python3.9-tui  cuobjdump  nvcc          nvlink    ptxas
compute-sanitizer    cuda-gdb-minimal  cuda-gdb-python3.12-tui  cuda-gdbserver          fatbinary  nvcc.profile  nvprof
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ vim ~/.bash
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ vim ~/.bashrc
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ . ~/.bash
bash: /home/scbz/.bash: No such file or directory
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ . ~/.bashrc
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ nvcc --version
bash: nvcc: command not found
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ cat ~/.bashrc
# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
        # We have color support; assume it's compliant with Ecma-48
        # (ISO/IEC-6429). (Lack of such support is extremely rare, and such
        # a case would tend to support setf rather than setaf.)
        color_prompt=yes
    else
        color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

export NCCL_IB_HCA=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=23
export NCCL_LAUNCH_MODE=GROUP

export PATH="/usr/share/cuda-12/bin:$PATH"
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ ls /usr/share/cuda-12/bin
ls: cannot access '/usr/share/cuda-12/bin': No such file or directory
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ ls /usr/share/
X11/                       debconf/                   glog/                      locales/                   python-wheels/
aclocal/                   debianutils/               gnupg/                     lto-disabled-list/         python3/
aclocal-1.16/              dict/                      groff/                     man/                       readline/
alsa/                      doc/                       grpc/                      man-db/                    rsync/
applications/              doc-base/                  gtk-doc/                   menu/                      sensible-utils/
apport/                    dot.bashrc                 i18n/                      misc/                      sgml/
autoconf/                  dot.profile                icons/                     motd                       staff-group-for-usr-local
automake-1.16/             dot.profile.md5sums        icu/                       nano/                      systemd/
base-files/                dpkg/                      info/                      networks                   tabset/
base-passwd/               emacs/                     info.dir                   openssh/                   tcltk/
bash-completion/           file/                      initramfs-tools/           pam/                       terminfo/
binfmts/                   fontconfig/                javascript/                pam-configs/               ucf/
bug/                       fonts/                     keyrings/                  perl/                      util-linux/
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ ls /usr/local/cuda
NsightSystems-cli-2024.6.2  bin  compat  compute-sanitizer  doc  extras  gds  include  lib64  nvml  nvvm  share  src  targets
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ ls /usr/local/cuda-12
NsightSystems-cli-2024.6.2  bin  compat  compute-sanitizer  doc  extras  gds  include  lib64  nvml  nvvm  share  src  targets
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ vim ~/.bashrc
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ . ~/.bashrc
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
src/kernels.cu(6): warning #1835-D: attribute "__global__" does not apply here
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                 ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

src/kernels.cu(6): error: incomplete type "void" is not allowed
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                               ^

src/kernels.cu(6): error: identifier "T" is undefined
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                                            ^

src/kernels.cu(6): error: identifier "A" is undefined
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                                               ^

src/kernels.cu(6): error: type name is not allowed
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                                                  ^

src/kernels.cu(6): error: expected a ")"
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                                                         ^

src/kernels.cu(6): error: expected a ";"
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                                                                  ^

src/kernels.cu(9): warning #2506-D: a user-provided literal suffix must begin with "_"
   double sum = 0.0lf;
                ^

src/kernels.cu(42): warning #12-D: parsing restarts here after previous syntax error
   ); } } while (0);
    ^

src/kernels.cu(42): error: expected a declaration
   ); } } while (0);
      ^

src/kernels.cu(44): error: this declaration has no storage class or type specifier
    trace_kernel<<<grid_size, block_size>>>(h, N, p);
    ^

src/kernels.cu(44): error: variable "trace_kernel" has already been defined (previous definition at line 6)
    trace_kernel<<<grid_size, block_size>>>(h, N, p);
    ^

src/kernels.cu(44): error: expected a ";"
    trace_kernel<<<grid_size, block_size>>>(h, N, p);
                ^

src/kernels.cu(47): error: expected a declaration
    do { cudaError_t err = cudaMemcpy(h_input, h, sizeof(T), cudaMemcpyDeviceToHost); if (err != cudaSuccess) { std::cerr << "Runtime error at " << "src/kernels.cu" << ":" << 47 << " - " << cudaGetErrorString(err) << "\n"; exit(
    ^

src/kernels.cu(47): warning #12-D: parsing restarts here after previous syntax error
   ); } } while (0);
                   ^

src/kernels.cu(49): error: expected a declaration
    return *p;
    ^

src/kernels.cu(50): error: expected a declaration
  }
  ^

src/kernels.cu(80): warning #12-D: parsing restarts here after previous syntax error
  template int trace<int>(const std::vector<int>&, size_t, size_t);
                                                                  ^

src/kernels.cu(81): error: trace is not a template
  template float trace<float>(const std::vector<float>&, size_t, size_t);
                 ^

src/kernels.cu(81): error: invalid explicit instantiation declaration
  template float trace<float>(const std::vector<float>&, size_t, size_t);
           ^

src/kernels.cu(82): error: flashAttention is not a template
  template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
                ^

src/kernels.cu(82): error: invalid explicit instantiation declaration
  template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
           ^

src/kernels.cu(85): error: flashAttention is not a template
  template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
                ^

src/kernels.cu(85): error: invalid explicit instantiation declaration
  template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
           ^

19 errors detected in the compilation of "src/kernels.cu".
make: *** [Makefile:103: src/kernels.o] Error 2
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ export SKIP_ATTENTION=1
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
src/kernels.cu(6): warning #1835-D: attribute "__global__" does not apply here
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                 ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

src/kernels.cu(6): error: incomplete type "void" is not allowed
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                               ^

src/kernels.cu(6): error: identifier "T" is undefined
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                                            ^

src/kernels.cu(6): error: identifier "A" is undefined
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                                               ^

src/kernels.cu(6): error: type name is not allowed
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                                                  ^

src/kernels.cu(6): error: expected a ")"
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                                                         ^

src/kernels.cu(6): error: expected a ";"
  __attribute__((global)) void trace_kernel(T* A, size_t N, T *p) {
                                                                  ^

src/kernels.cu(9): warning #2506-D: a user-provided literal suffix must begin with "_"
   double sum = 0.0lf;
                ^

src/kernels.cu(42): warning #12-D: parsing restarts here after previous syntax error
   ); } } while (0);
    ^

src/kernels.cu(42): error: expected a declaration
   ); } } while (0);
      ^

src/kernels.cu(44): error: this declaration has no storage class or type specifier
    trace_kernel<<<grid_size, block_size>>>(h, N, p);
    ^

src/kernels.cu(44): error: variable "trace_kernel" has already been defined (previous definition at line 6)
    trace_kernel<<<grid_size, block_size>>>(h, N, p);
    ^

src/kernels.cu(44): error: expected a ";"
    trace_kernel<<<grid_size, block_size>>>(h, N, p);
                ^

src/kernels.cu(47): error: expected a declaration
    do { cudaError_t err = cudaMemcpy(h_input, h, sizeof(T), cudaMemcpyDeviceToHost); if (err != cudaSuccess) { std::cerr << "Runtime error at " << "src/kernels.cu" << ":" << 47 << " - " << cudaGetErrorString(err) << "\n"; exit(
    ^

src/kernels.cu(47): warning #12-D: parsing restarts here after previous syntax error
   ); } } while (0);
                   ^

src/kernels.cu(49): error: expected a declaration
    return *p;
    ^

src/kernels.cu(50): error: expected a declaration
  }
  ^

src/kernels.cu(80): warning #12-D: parsing restarts here after previous syntax error
  template int trace<int>(const std::vector<int>&, size_t, size_t);
                                                                  ^

src/kernels.cu(81): error: trace is not a template
  template float trace<float>(const std::vector<float>&, size_t, size_t);
                 ^

src/kernels.cu(81): error: invalid explicit instantiation declaration
  template float trace<float>(const std::vector<float>&, size_t, size_t);
           ^

src/kernels.cu(82): error: flashAttention is not a template
  template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
                ^

src/kernels.cu(82): error: invalid explicit instantiation declaration
  template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
           ^

src/kernels.cu(85): error: flashAttention is not a template
  template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
                ^

src/kernels.cu(85): error: invalid explicit instantiation declaration
  template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
           ^

19 errors detected in the compilation of "src/kernels.cu".
make: *** [Makefile:103: src/kernels.o] Error 2
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
src/kernels.cu(10): warning #2506-D: a user-provided literal suffix must begin with "_"
   double sum = 0.0lf;
                ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

src/kernels.cu(10): error: user-defined literal operator not found
   double sum = 0.0lf;
                ^

src/kernels.cu(43): error: no suitable conversion function from "const std::vector<int, std::allocator<int>>" to "const void *" exists
    do { cudaError_t err = cudaMemcpy(h, h_input, N * N * sizeof(T), cudaMemcpyHostToDevice); if (err != cudaSuccess) { std::cerr << "Runtime error at " << "src/kernels.cu" << ":" << 43 << " - " << cudaGetErrorString(err) << "\n"; exit(
                                         ^
          detected during instantiation of "T trace(const std::vector<T, std::allocator<T>> &, size_t, size_t) [with T=int]" at line 81

src/kernels.cu(48): error: no suitable conversion function from "const std::vector<int, std::allocator<int>>" to "void *" exists
    do { cudaError_t err = cudaMemcpy(h_input, h, sizeof(T), cudaMemcpyDeviceToHost); if (err != cudaSuccess) { std::cerr << "Runtime error at " << "src/kernels.cu" << ":" << 48 << " - " << cudaGetErrorString(err) << "\n"; exit(
                                      ^
          detected during instantiation of "T trace(const std::vector<T, std::allocator<T>> &, size_t, size_t) [with T=int]" at line 81

src/kernels.cu(43): error: no suitable conversion function from "const std::vector<float, std::allocator<float>>" to "const void *" exists
    do { cudaError_t err = cudaMemcpy(h, h_input, N * N * sizeof(T), cudaMemcpyHostToDevice); if (err != cudaSuccess) { std::cerr << "Runtime error at " << "src/kernels.cu" << ":" << 43 << " - " << cudaGetErrorString(err) << "\n"; exit(
                                         ^
          detected during instantiation of "T trace(const std::vector<T, std::allocator<T>> &, size_t, size_t) [with T=float]" at line 82

src/kernels.cu(48): error: no suitable conversion function from "const std::vector<float, std::allocator<float>>" to "void *" exists
    do { cudaError_t err = cudaMemcpy(h_input, h, sizeof(T), cudaMemcpyDeviceToHost); if (err != cudaSuccess) { std::cerr << "Runtime error at " << "src/kernels.cu" << ":" << 48 << " - " << cudaGetErrorString(err) << "\n"; exit(
                                      ^
          detected during instantiation of "T trace(const std::vector<T, std::allocator<T>> &, size_t, size_t) [with T=float]" at line 82

5 errors detected in the compilation of "src/kernels.cu".
make: *** [Makefile:103: src/kernels.o] Error 2
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
src/kernels.cu(43): error: no suitable conversion function from "const std::vector<int, std::allocator<int>>" to "const void *" exists
    do { cudaError_t err = cudaMemcpy(h, h_input, N * N * sizeof(T), cudaMemcpyHostToDevice); if (err != cudaSuccess) { std::cerr << "Runtime error at " << "src/kernels.cu" << ":" << 43 << " - " << cudaGetErrorString(err) << "\n"; exit(
                                         ^
          detected during instantiation of "T trace(const std::vector<T, std::allocator<T>> &, size_t, size_t) [with T=int]" at line 81

src/kernels.cu(48): error: no suitable conversion function from "const std::vector<int, std::allocator<int>>" to "void *" exists
    do { cudaError_t err = cudaMemcpy(h_input, h, sizeof(T), cudaMemcpyDeviceToHost); if (err != cudaSuccess) { std::cerr << "Runtime error at " << "src/kernels.cu" << ":" << 48 << " - " << cudaGetErrorString(err) << "\n"; exit(
                                      ^
          detected during instantiation of "T trace(const std::vector<T, std::allocator<T>> &, size_t, size_t) [with T=int]" at line 81

src/kernels.cu(43): error: no suitable conversion function from "const std::vector<float, std::allocator<float>>" to "const void *" exists
    do { cudaError_t err = cudaMemcpy(h, h_input, N * N * sizeof(T), cudaMemcpyHostToDevice); if (err != cudaSuccess) { std::cerr << "Runtime error at " << "src/kernels.cu" << ":" << 43 << " - " << cudaGetErrorString(err) << "\n"; exit(
                                         ^
          detected during instantiation of "T trace(const std::vector<T, std::allocator<T>> &, size_t, size_t) [with T=float]" at line 82

src/kernels.cu(48): error: no suitable conversion function from "const std::vector<float, std::allocator<float>>" to "void *" exists
    do { cudaError_t err = cudaMemcpy(h_input, h, sizeof(T), cudaMemcpyDeviceToHost); if (err != cudaSuccess) { std::cerr << "Runtime error at " << "src/kernels.cu" << ":" << 48 << " - " << cudaGetErrorString(err) << "\n"; exit(
                                      ^
          detected during instantiation of "T trace(const std::vector<T, std::allocator<T>> &, size_t, size_t) [with T=float]" at line 82

4 errors detected in the compilation of "src/kernels.cu".
make: *** [Makefile:103: src/kernels.o] Error 2
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
src/kernels.cu(48): error: argument of type "const int *" is incompatible with parameter of type "void *"
    do { cudaError_t err = cudaMemcpy(h_input.data(), h, sizeof(T), cudaMemcpyDeviceToHost); if (err != cudaSuccess) { std::cerr << "Runtime error at " << "src/kernels.cu" << ":" << 48 << " - " << cudaGetErrorString(err) << "\n"; exit(
                                      ^
          detected during instantiation of "T trace(const std::vector<T, std::allocator<T>> &, size_t, size_t) [with T=int]" at line 81

src/kernels.cu(48): error: argument of type "const float *" is incompatible with parameter of type "void *"
    do { cudaError_t err = cudaMemcpy(h_input.data(), h, sizeof(T), cudaMemcpyDeviceToHost); if (err != cudaSuccess) { std::cerr << "Runtime error at " << "src/kernels.cu" << ":" << 48 << " - " << cudaGetErrorString(err) << "\n"; exit(
                                      ^
          detected during instantiation of "T trace(const std::vector<T, std::allocator<T>> &, size_t, size_t) [with T=float]" at line 82

2 errors detected in the compilation of "src/kernels.cu".
make: *** [Makefile:103: src/kernels.o] Error 2
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ template <typename T>^C
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ 


scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ VERBOSE=1 make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
=== Linking executable (student code + test logic) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -o test_kernels src/kernels.o tester/tester_nv.o 
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
=== Running tests (output from src/kernels.o) ===
=== Verbose mode: Enabled (using '--verbose') ===
./test_kernels  
Testing on device: NVIDIA A100-SXM4-80GB

=== trace Tests ===
make: *** [Makefile:81: run] Segmentation fault
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ cuda-memcheck ./test_kernels
bash: cuda-memcheck: command not found
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ gdb ./test_kernels
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ cuda-gdb ./test_kernels
NVIDIA (R) cuda-gdb 12.8
Portions Copyright (C) 2007-2024 NVIDIA Corporation
Based on GNU gdb 13.2
Copyright (C) 2023 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This CUDA-GDB was configured as "x86_64-pc-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://forums.developer.nvidia.com/c/developer-tools/cuda-developer-tools/cuda-gdb>.
Find the CUDA-GDB manual and other documentation resources online at:
--Type <RET> for more, q to quit, c to continue without paging--
    <https://docs.nvidia.com/cuda/cuda-gdb/index.html>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./test_kernels...
(No debugging symbols found in ./test_kernels)
(cuda-gdb) b trace                                                                                                                          
Function "trace" not defined.
Make breakpoint pending on future shared library load? (y or [n]) y
Breakpoint 1 (trace) pending.
(cuda-gdb) r
Starting program: /home/scbz/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA/test_kernels 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7ffff33ff000 (LWP 2534650)]
[New Thread 0x7ffff1fff000 (LWP 2534651)]
[Detaching after fork from child process 2534652]
[New Thread 0x7fffe899c000 (LWP 2534765)]
Testing on device: NVIDIA A100-SXM4-80GB

=== trace Tests ===

Thread 1 "test_kernels" received signal SIGSEGV, Segmentation fault.
0x00007ffff3638cdc in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
(cuda-gdb) quit                                                                        
A debugging session is active.

        Inferior 1 [process 2534645] will be killed.

Quit anyway? (y or n) y
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ DEGUG=1 make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
=== Linking executable (student code + test logic) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -o test_kernels src/kernels.o tester/tester_nv.o 
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
=== Running tests (output from src/kernels.o) ===
=== Verbose mode: Enabled (using '--verbose') ===
./test_kernels  
Testing on device: NVIDIA A100-SXM4-80GB

=== trace Tests ===
make: *** [Makefile:81: run] Segmentation fault
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make clean
=== Cleaning temporary files ===
rm -f test_kernels src/kernels.o
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ DEGUG=1 make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
=== Linking executable (student code + test logic) ===
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -o test_kernels src/kernels.o tester/tester_nv.o 
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
=== Running tests (output from src/kernels.o) ===
=== Verbose mode: Enabled (using '--verbose') ===
./test_kernels  
Testing on device: NVIDIA A100-SXM4-80GB

=== trace Tests ===
make: *** [Makefile:81: run] Segmentation fault
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ cuda-gdb ./test_kernels
NVIDIA (R) cuda-gdb 12.8
Portions Copyright (C) 2007-2024 NVIDIA Corporation
Based on GNU gdb 13.2
Copyright (C) 2023 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This CUDA-GDB was configured as "x86_64-pc-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://forums.developer.nvidia.com/c/developer-tools/cuda-developer-tools/cuda-gdb>.
Find the CUDA-GDB manual and other documentation resources online at:
    <https://docs.nvidia.com/cuda/cuda-gdb/index.html>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./test_kernels...
(No debugging symbols found in ./test_kernels)
(cuda-gdb) b trace
Function "trace" not defined.
Make breakpoint pending on future shared library load? (y or [n]) y
Breakpoint 1 (trace) pending.
(cuda-gdb) r      
Starting program: /home/scbz/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA/test_kernels 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7ffff33ff000 (LWP 2543409)]
[New Thread 0x7ffff1fff000 (LWP 2543413)]
[Detaching after fork from child process 2543414]
[New Thread 0x7fffe899c000 (LWP 2543595)]
Testing on device: NVIDIA A100-SXM4-80GB

=== trace Tests ===

Thread 1 "test_kernels" received signal SIGSEGV, Segmentation fault.
0x00007ffff3638cdc in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
(cuda-gdb) quit
A debugging session is active.

        Inferior 1 [process 2543382] will be killed.

Quit anyway? (y or n) y
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make clean
=== Cleaning temporary files ===
rm -f test_kernels src/kernels.o
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -g -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
=== Linking executable (student code + test logic) ===
nvcc -std=c++17 -O0 -g -DPLATFORM_NVIDIA -o test_kernels src/kernels.o tester/tester_nv.o 
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
=== Running tests (output from src/kernels.o) ===
=== Verbose mode: Enabled (using '--verbose') ===
./test_kernels  
Testing on device: NVIDIA A100-SXM4-80GB

=== trace Tests ===
make: *** [Makefile:81: run] Segmentation fault
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ cuda-gdb ./test_kernels
NVIDIA (R) cuda-gdb 12.8
Portions Copyright (C) 2007-2024 NVIDIA Corporation
Based on GNU gdb 13.2
Copyright (C) 2023 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This CUDA-GDB was configured as "x86_64-pc-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://forums.developer.nvidia.com/c/developer-tools/cuda-developer-tools/cuda-gdb>.
Find the CUDA-GDB manual and other documentation resources online at:
    <https://docs.nvidia.com/cuda/cuda-gdb/index.html>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./test_kernels...
(cuda-gdb) b trace
Breakpoint 1 at 0xa448: trace. (2 locations)
(cuda-gdb) r
Starting program: /home/scbz/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA/test_kernels 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7ffff33ff000 (LWP 2583146)]
[New Thread 0x7ffff1fff000 (LWP 2583147)]
[Detaching after fork from child process 2583148]
[New Thread 0x7fffe899c000 (LWP 2583176)]
Testing on device: NVIDIA A100-SXM4-80GB

=== trace Tests ===

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0)
    at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) s
34        const size_t N = rows<cols ? rows : cols;
(cuda-gdb) s
35        const size_t SIZE = rows*cols;
(cuda-gdb) s
43        RUNTIME_CHECK(cudaMalloc((void**) &d_input, SIZE * sizeof(T)));
(cuda-gdb) s
44        RUNTIME_CHECK(cudaMalloc((void**) &p, sizeof(double)));
(cuda-gdb) s
45        RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(T), cudaMemcpyHostToDevice));
(cuda-gdb) s
std::vector<int, std::allocator<int> >::data (this=0x7fffffffc320)
    at /usr/include/c++/13/bits/stl_vector.h:1266
1266          { return _M_data_ptr(this->_M_impl._M_start); }
(cuda-gdb) s
std::vector<int, std::allocator<int> >::_M_data_ptr<int> (this=0x7fffffffc320, 
    __ptr=0x0) at /usr/include/c++/13/bits/stl_vector.h:1991
1991            { return __ptr; }
(cuda-gdb) n
std::vector<int, std::allocator<int> >::data (this=0x7fffffffc320)
    at /usr/include/c++/13/bits/stl_vector.h:1266
1266          { return _M_data_ptr(this->_M_impl._M_start); }
(cuda-gdb) n
trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:47
47        trace_kernel<<<1,1>>>(d_input, N, SIZE, p);
(cuda-gdb) n
50        RUNTIME_CHECK(cudaMemcpy(ans, p, sizeof(double), cudaMemcpyDeviceToHost));
(cuda-gdb) s
52        return static_cast<T>(*ans);
(cuda-gdb) s
53      }
(cuda-gdb) s
0x000055555556256f in void run_trace_test<int>(int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, unsigned long, unsigned long, unsigned long, unsigned long) ()
(cuda-gdb) s
Single stepping until exit from function _Z14run_trace_testIiEviRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEmmmm,
which has no line number information.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0)
    at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) s
34        const size_t N = rows<cols ? rows : cols;
(cuda-gdb) s
35        const size_t SIZE = rows*cols;
(cuda-gdb) s
43        RUNTIME_CHECK(cudaMalloc((void**) &d_input, SIZE * sizeof(T)));
(cuda-gdb) s
44        RUNTIME_CHECK(cudaMalloc((void**) &p, sizeof(double)));
(cuda-gdb) s
45        RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(T), cudaMemcpyHostToDevice));
(cuda-gdb) s
std::vector<int, std::allocator<int> >::data (this=0x7fffffffc320)
    at /usr/include/c++/13/bits/stl_vector.h:1266
1266          { return _M_data_ptr(this->_M_impl._M_start); }
(cuda-gdb) s
std::vector<int, std::allocator<int> >::_M_data_ptr<int> (this=0x7fffffffc320, 
    __ptr=0x0) at /usr/include/c++/13/bits/stl_vector.h:1991
1991            { return __ptr; }
(cuda-gdb) s
std::vector<int, std::allocator<int> >::data (this=0x7fffffffc320)
    at /usr/include/c++/13/bits/stl_vector.h:1266
1266          { return _M_data_ptr(this->_M_impl._M_start); }
(cuda-gdb) s
trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:47
47        trace_kernel<<<1,1>>>(d_input, N, SIZE, p);
(cuda-gdb) s
dim3::dim3 (this=0x7fffffffc25c, vx=1, vy=1, vz=1)
    at /usr/local/cuda-12.8/targets/x86_64-linux/include/vector_types.h:431
431         __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
(cuda-gdb) s
dim3::dim3 (this=0x7fffffffc250, vx=1, vy=1, vz=1)
    at /usr/local/cuda-12.8/targets/x86_64-linux/include/vector_types.h:431
431         __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
(cuda-gdb) s
trace_kernel<int> (A=0x0, N=0, SIZE=0, p=0x7fffb7800200) at src/kernels.cu:8
8       __global__ void trace_kernel(T* A, size_t N, size_t SIZE, double *p) {
(cuda-gdb) s
__wrapper__device_stub_trace_kernel<int> (__cuda_0=@0x7fffffffc1d8: 0x0, 
    __cuda_1=@0x7fffffffc1d0: 0, __cuda_2=@0x7fffffffc1c8: 0, 
    __cuda_3=@0x7fffffffc1c0: 0x7fffb7800200)
    at /tmp/tmpxft_0027651b_00000000-6_kernels.cudafe1.stub.c:15
15      /tmp/tmpxft_0027651b_00000000-6_kernels.cudafe1.stub.c: No such file or directory.
(cuda-gdb) s
__device_stub__Z12trace_kernelIiEvPT_mmPd (__par0=0x0, __par1=0, __par2=0, 
    __par3=0x7fffb7800200)
    at /tmp/tmpxft_0027651b_00000000-6_kernels.cudafe1.stub.c:14
14      in /tmp/tmpxft_0027651b_00000000-6_kernels.cudafe1.stub.c
(cuda-gdb) s
cudaLaunchKernel<char> (
    func=0x55555555eca5 <trace_kernel<int>(int*, unsigned long, unsigned long, double*)> "\363\017\036\372UH\211\345H\203\354 H\211}\370H\211u\360H\211U\350H\211M\340H\215M\340H\215U\350H\215u\360H\215E\370H\211\307\350\324\363\377\377\220\311\303\363\017\036\372UH\211\345H\203\354 H\211}\370H\211u\360H\211U\350H\211M\340H\215M\340H\215U\350H\215u\360H\215E\370H\211\307\350M\365\377\377\220\311Ð\363\017\036\372UH\211\345H\203\354\020H\211}\370H\213E\370H\213\020H\213E\370H\211\326H\211\307\350*", gridDim=..., 
    blockDim=..., args=0x7fffffffc150, sharedMem=0, stream=0x0)
    at /usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_runtime.h:217
217         return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream);
(cuda-gdb) s
218     }
(cuda-gdb) s
__wrapper__device_stub_trace_kernel<int> (__cuda_0=@0x7fffffffc1d8: 0x0, 
    __cuda_1=@0x7fffffffc1d0: 0, __cuda_2=@0x7fffffffc1c8: 0, 
    __cuda_3=@0x7fffffffc1c0: 0x7fffb7800200)
    at /tmp/tmpxft_0027651b_00000000-6_kernels.cudafe1.stub.c:15
15      /tmp/tmpxft_0027651b_00000000-6_kernels.cudafe1.stub.c: No such file or directory.
(cuda-gdb) s
trace_kernel<int> (A=0x0, N=0, SIZE=0, p=0x7fffb7800200) at src/kernels.cu:15
15      }
(cuda-gdb) s
trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:50
50        RUNTIME_CHECK(cudaMemcpy(ans, p, sizeof(double), cudaMemcpyDeviceToHost));
(cuda-gdb) s

Thread 1 "test_kernels" received signal SIGSEGV, Segmentation fault.
0x00007ffff3638cdc in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
(cuda-gdb) bt
#0  0x00007ffff3638cdc in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#1  0x00007ffff36d6c7a in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007ffff37f3fcc in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007ffff37fb337 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007ffff37fb51b in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#5  0x00007ffff4163bac in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#6  0x00007ffff35dd2fc in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#7  0x00007ffff411ae8a in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#8  0x00007ffff35cd9c5 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#9  0x00007ffff377c4ca in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#10 0x00005555555a63e9 in libcudart_static_e760a7280bc1af45ef561886abc194823c5edadb ()
#11 0x0000555555574b64 in libcudart_static_901c335edfe2e89188fa6b8fe3e2a2194ad05bff ()
#12 0x00005555555d250e in cudaMemcpy ()
#13 0x000055555555e76f in trace<int> (h_input=..., rows=0, cols=0)
    at src/kernels.cu:50
#14 0x00005555555625d8 in void run_trace_test<int>(int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, unsigned long, unsigned long, unsigned long, unsigned long) ()
#15 0x000055555555ffb4 in main ()
                                   (cuda-gdb) p d_input                                                                                                                                     
No symbol "d_input" in current context.
(cuda-gdb) p p 
No symbol "p" in current context.
(cuda-gdb) quit
A debugging session is active.

        Inferior 1 [process 2583142] will be killed.

Quit anyway? (y or n) y
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ cuda-gdb ./test_kernels
NVIDIA (R) cuda-gdb 12.8
Portions Copyright (C) 2007-2024 NVIDIA Corporation
Based on GNU gdb 13.2
Copyright (C) 2023 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This CUDA-GDB was configured as "x86_64-pc-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://forums.developer.nvidia.com/c/developer-tools/cuda-developer-tools/cuda-gdb>.
Find the CUDA-GDB manual and other documentation resources online at:
    <https://docs.nvidia.com/cuda/cuda-gdb/index.html>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./test_kernels...
(cuda-gdb) b Quit
(cuda-gdb) quit
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make clean
=== Cleaning temporary files ===
rm -f test_kernels src/kernels.o
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -g -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
=== Linking executable (student code + test logic) ===
nvcc -std=c++17 -O0 -g -DPLATFORM_NVIDIA -o test_kernels src/kernels.o tester/tester_nv.o 
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
=== Running tests (output from src/kernels.o) ===
=== Verbose mode: Enabled (using '--verbose') ===
./test_kernels  
Testing on device: NVIDIA A100-SXM4-80GB

=== trace Tests ===
make: *** [Makefile:81: run] Segmentation fault
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ cuda-gdb ./test_kernels
NVIDIA (R) cuda-gdb 12.8
Portions Copyright (C) 2007-2024 NVIDIA Corporation
Based on GNU gdb 13.2
Copyright (C) 2023 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This CUDA-GDB was configured as "x86_64-pc-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://forums.developer.nvidia.com/c/developer-tools/cuda-developer-tools/cuda-gdb>.
Find the CUDA-GDB manual and other documentation resources online at:
    <https://docs.nvidia.com/cuda/cuda-gdb/index.html>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./test_kernels...
(cuda-gdb) b trace
Breakpoint 1 at 0xa448: trace. (2 locations)
(cuda-gdb) r
Starting program: /home/scbz/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA/test_kernels 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7ffff33ff000 (LWP 2617655)]
[New Thread 0x7ffff1fff000 (LWP 2617656)]
[Detaching after fork from child process 2617657]
[New Thread 0x7fffe899c000 (LWP 2617759)]
Testing on device: NVIDIA A100-SXM4-80GB

=== trace Tests ===

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) s
34        const size_t N = rows<cols ? rows : cols;
(cuda-gdb) s
35        const size_t SIZE = rows*cols;
(cuda-gdb) s
36        dim3 grid_size(256);
(cuda-gdb) s
37        dim3 block_size(
(cuda-gdb) s
dim3::dim3 (this=0x7fffffffc25c, vx=0, vy=1, vz=1) at /usr/local/cuda-12.8/targets/x86_64-linux/include/vector_types.h:431
431         __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
(cuda-gdb) s
trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:43
43        RUNTIME_CHECK(cudaMalloc((void**) &d_input, SIZE * sizeof(T)));
(cuda-gdb) s
44        RUNTIME_CHECK(cudaMalloc((void**) &p, sizeof(double)));
(cuda-gdb) s
45        RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(T), cudaMemcpyHostToDevice));
(cuda-gdb) s
std::vector<int, std::allocator<int> >::data (this=0x7fffffffc320) at /usr/include/c++/13/bits/stl_vector.h:1266
1266          { return _M_data_ptr(this->_M_impl._M_start); }
(cuda-gdb) s
std::vector<int, std::allocator<int> >::_M_data_ptr<int> (this=0x7fffffffc320, __ptr=0x0) at /usr/include/c++/13/bits/stl_vector.h:1991
1991            { return __ptr; }
(cuda-gdb) s
std::vector<int, std::allocator<int> >::data (this=0x7fffffffc320) at /usr/include/c++/13/bits/stl_vector.h:1266
1266          { return _M_data_ptr(this->_M_impl._M_start); }
(cuda-gdb) s
trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:47
47        trace_kernel<<<grid_size,block_size>>>(d_input, N, SIZE, p);
(cuda-gdb) s
trace_kernel<int> (A=0x0, N=0, SIZE=0, p=0x7fffb7800000) at src/kernels.cu:8
8       __global__ void trace_kernel(T* A, size_t N, size_t SIZE, double *p) {
(cuda-gdb) s
__wrapper__device_stub_trace_kernel<int> (__cuda_0=@0x7fffffffc1d8: 0x0, __cuda_1=@0x7fffffffc1d0: 0, __cuda_2=@0x7fffffffc1c8: 0, 
    __cuda_3=@0x7fffffffc1c0: 0x7fffb7800000) at /tmp/tmpxft_0027ee7e_00000000-6_kernels.cudafe1.stub.c:15
15      /tmp/tmpxft_0027ee7e_00000000-6_kernels.cudafe1.stub.c: No such file or directory.
(cuda-gdb) s
__device_stub__Z12trace_kernelIiEvPT_mmPd (__par0=0x0, __par1=0, __par2=0, __par3=0x7fffb7800000)
    at /tmp/tmpxft_0027ee7e_00000000-6_kernels.cudafe1.stub.c:14
14      in /tmp/tmpxft_0027ee7e_00000000-6_kernels.cudafe1.stub.c
(cuda-gdb) s
cudaLaunchKernel<char> (
    func=0x55555555eccd <trace_kernel<int>(int*, unsigned long, unsigned long, double*)> "\363\017\036\372UH\211\345H\203\354 H\211}\370H\211u\360H\211U\350H\211M\340H\215M\340H\215U\350H\215u\360H\215E\370H\211\307\350\254\363\377\377\220\311\303\363\017\036\372UH\211\345H\203\354 H\211}\370H\211u\360H\211U\350H\211M\340H\215M\340H\215U\350H\215u\360H\215E\370H\211\307\350%\365\377\377\220\311Ð\363\017\036\372UH\211\345H\203\354\020H\211}\370H\213E\370H\213\020H\213E\370H\211\326H\211\307\350*", gridDim=..., blockDim=..., args=0x7fffffffc150, sharedMem=0, stream=0x0)
    at /usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_runtime.h:217
217         return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream);
(cuda-gdb) s
warning: Cuda Runtime API error detected: cudaLaunchKernel returned cudaErrorInvalidConfiguration(0x9): invalid configuration argument

218     }
(cuda-gdb) s
__wrapper__device_stub_trace_kernel<int> (__cuda_0=@0x7fffffffc1d8: 0x0, __cuda_1=@0x7fffffffc1d0: 0, __cuda_2=@0x7fffffffc1c8: 0, 
    __cuda_3=@0x7fffffffc1c0: 0x7fffb7800000) at /tmp/tmpxft_0027ee7e_00000000-6_kernels.cudafe1.stub.c:15
15      /tmp/tmpxft_0027ee7e_00000000-6_kernels.cudafe1.stub.c: No such file or directory.
(cuda-gdb) s
trace_kernel<int> (A=0x0, N=0, SIZE=0, p=0x7fffb7800000) at src/kernels.cu:15
15      }
(cuda-gdb) n
trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:50
50        RUNTIME_CHECK(cudaMemcpy(ans, p, sizeof(double), cudaMemcpyDeviceToHost));
(cuda-gdb) s
52        return static_cast<T>(*ans);
(cuda-gdb) s
53      }
(cuda-gdb) s
0x0000555555562597 in void run_trace_test<int>(int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, unsigned long, unsigned long, unsigned long, unsigned long) ()
(cuda-gdb) r
The program being debugged has been started already.
Start it from the beginning? (y or n) n
Program not restarted.
(cuda-gdb) help 
List of classes of commands:

aliases -- User-defined aliases of other commands.
breakpoints -- Making program stop at certain points.
cuda  -- CUDA commands.
data -- Examining data.
files -- Specifying and examining files.
internals -- Maintenance commands.
obscure -- Obscure features.
running -- Running the program.
stack -- Examining the stack.
status -- Status inquiries.
support -- Support facilities.
text-user-interface -- TUI is the GDB text based interface.
tracepoints -- Tracing of program execution without stopping the program.
user-defined -- User-defined commands.

Type "help" followed by a class name for a list of commands in that class.
Type "help all" for the list of all commands.
Type "help" followed by command name for full documentation.
Type "apropos word" to search for commands related to "word".
Type "apropos -v word" for full documentation of commands related to "word".
Command name abbreviations are allowed if unambiguous.
(cuda-gdb) help running
Running the program.

List of commands:

advance -- Continue the program up to the given location (same form as args for break command).
attach -- Attach to a process or file outside of GDB.
continue, fg, c -- Continue program being debugged, after signal or breakpoint.
detach -- Detach a process or file previously attached.
detach checkpoint -- Detach from a checkpoint (experimental).
detach inferiors -- Detach from inferior ID (or list of IDS).
disconnect -- Disconnect from a target.
finish, fin -- Execute until selected stack frame returns.
handle -- Specify how to handle signals.
inferior -- Use this command to switch between inferiors.
interrupt -- Interrupt the execution of the debugged program.
jump, j -- Continue program being debugged at specified line or address.
kill -- Kill execution of program being debugged.
kill inferiors -- Kill inferior ID (or list of IDs).
next, n -- Step program, proceeding through subroutine calls.
nexti, ni -- Step one instruction, but proceed through subroutine calls.
queue-signal -- Queue a signal to be delivered to the current thread when it is resumed.
reverse-continue, rc -- Continue program being debugged but run it in reverse.
reverse-finish -- Execute backward until just before selected stack frame is called.
--Type <RET> for more, q to quit, c to continue without paging--q
Quit
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) s
34        const size_t N = rows<cols ? rows : cols;
(cuda-gdb) n
35        const size_t SIZE = rows*cols;
(cuda-gdb) n
36        dim3 grid_size(256);
(cuda-gdb) n
37        dim3 block_size(
(cuda-gdb) n
43        RUNTIME_CHECK(cudaMalloc((void**) &d_input, SIZE * sizeof(T)));
(cuda-gdb) n
44        RUNTIME_CHECK(cudaMalloc((void**) &p, sizeof(double)));
(cuda-gdb) n
45        RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(T), cudaMemcpyHostToDevice));
(cuda-gdb) n
47        trace_kernel<<<grid_size,block_size>>>(d_input, N, SIZE, p);
(cuda-gdb) s
trace_kernel<int> (A=0x0, N=0, SIZE=0, p=0x7fffb7800200) at src/kernels.cu:8
8       __global__ void trace_kernel(T* A, size_t N, size_t SIZE, double *p) {
(cuda-gdb) n
warning: Cuda Runtime API error detected: cudaLaunchKernel returned cudaErrorInvalidConfiguration(0x9): invalid configuration argument

15      }
(cuda-gdb) p A
$1 = (int *) 0x0
(cuda-gdb) p N
$2 = 0
(cuda-gdb) quit   
A debugging session is active.

        Inferior 1 [process 2617646] will be killed.

Quit anyway? (y or n) y
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make clean
=== Cleaning temporary files ===
rm -f test_kernels src/kernels.o
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ make
=== Compiling student code (src/kernels.cu ) ===
nvcc -std=c++17 -O0 -g -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
^[[A=== Linking executable (student code + test logic) ===
nvcc -std=c++17 -O0 -g -DPLATFORM_NVIDIA -o test_kernels src/kernels.o tester/tester_nv.o 
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
=== Running tests (output from src/kernels.o) ===
=== Verbose mode: Enabled (using '--verbose') ===
./test_kernels  
Testing on device: NVIDIA A100-SXM4-80GB

=== trace Tests ===
Test # 1: int    | Verification: Passed
Test # 1: float  | Verification: Passed
make: *** [Makefile:81: run] Segmentation fault
scbz@dsw-607126-66f5f844d-2v4s7:~/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA$ cuda-gdb ./test_kernels
NVIDIA (R) cuda-gdb 12.8
Portions Copyright (C) 2007-2024 NVIDIA Corporation
Based on GNU gdb 13.2
Copyright (C) 2023 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This CUDA-GDB was configured as "x86_64-pc-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://forums.developer.nvidia.com/c/developer-tools/cuda-developer-tools/cuda-gdb>.
Find the CUDA-GDB manual and other documentation resources online at:
    <https://docs.nvidia.com/cuda/cuda-gdb/index.html>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./test_kernels...
(cuda-gdb) b trace
Breakpoint 1 at 0xa448: trace. (2 locations)
(cuda-gdb) r
Starting program: /home/scbz/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA/test_kernels 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7ffff33ff000 (LWP 2629242)]
[New Thread 0x7ffff1fff000 (LWP 2629243)]
[Detaching after fork from child process 2629244]
[New Thread 0x7fffe899c000 (LWP 2629299)]
Testing on device: NVIDIA A100-SXM4-80GB

=== trace Tests ===

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.
Test # 1: int    | Verification: Passed

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.
Test # 1: float  | Verification: Passed

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=1, cols=1) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) s
34        if (rows == 0 || cols == 0)
(cuda-gdb) n
36        const size_t N = rows<cols ? rows : cols;
(cuda-gdb) n
37        const size_t SIZE = rows*cols;
(cuda-gdb) n
38        dim3 grid_size(256);
(cuda-gdb) n
39        dim3 block_size(
(cuda-gdb) n
45        RUNTIME_CHECK(cudaMalloc((void**) &d_input, SIZE * sizeof(T)));
(cuda-gdb) n
46        RUNTIME_CHECK(cudaMalloc((void**) &p, sizeof(double)));
(cuda-gdb) n
47        RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(T), cudaMemcpyHostToDevice));
(cuda-gdb) n
49        trace_kernel<<<grid_size,block_size>>>(d_input, N, SIZE, p);
(cuda-gdb) n
52        RUNTIME_CHECK(cudaMemcpy(ans, p, sizeof(double), cudaMemcpyDeviceToHost));
(cuda-gdb) n

Thread 1 "test_kernels" received signal SIGSEGV, Segmentation fault.
0x00007ffff3638c78 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
(cuda-gdb) r
The program being debugged has been started already.
Start it from the beginning? (y or n) y
Starting program: /home/scbz/Learning-Infinitensor/codes/stage1/cuda/Learning-CUDA/test_kernels 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7ffff33ff000 (LWP 2631606)]
[New Thread 0x7ffff1fff000 (LWP 2631607)]
[Detaching after fork from child process 2631608]
c[New Thread 0x7fffe899c000 (LWP 2631778)]
Testing on device: NVIDIA A100-SXM4-80GB

=== trace Tests ===

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.
Test # 1: int    | Verification: Passed

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.

Thread 1 "test_kernels" hit Breakpoint 1.2, trace<float> (h_input=..., rows=0, cols=0) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) c
Continuing.
Test # 1: float  | Verification: Passed

Thread 1 "test_kernels" hit Breakpoint 1.1, trace<int> (h_input=..., rows=1, cols=1) at src/kernels.cu:32
32      T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
(cuda-gdb) s
34        if (rows == 0 || cols == 0)
(cuda-gdb) n
36        const size_t N = rows<cols ? rows : cols;
(cuda-gdb) n
37        const size_t SIZE = rows*cols;
(cuda-gdb) n
38        dim3 grid_size(256);
(cuda-gdb) n
39        dim3 block_size(
(cuda-gdb) n
45        RUNTIME_CHECK(cudaMalloc((void**) &d_input, SIZE * sizeof(T)));
(cuda-gdb) n
46        RUNTIME_CHECK(cudaMalloc((void**) &p, sizeof(double)));
(cuda-gdb) n
47        RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(T), cudaMemcpyHostToDevice));
(cuda-gdb) n
49        trace_kernel<<<grid_size,block_size>>>(d_input, N, SIZE, p);
(cuda-gdb) s
trace_kernel<int> (A=0x7fffb7800000, N=1, SIZE=1, p=0x7fffb7800200) at src/kernels.cu:8
8       __global__ void trace_kernel(T* A, size_t N, size_t SIZE, double *p) {
(cuda-gdb) n
15      }
(cuda-gdb) n
trace<int> (h_input=..., rows=1, cols=1) at src/kernels.cu:52
52        RUNTIME_CHECK(cudaMemcpy(ans, p, sizeof(double), cudaMemcpyDeviceToHost));
(cuda-gdb) s

Thread 1 "test_kernels" received signal SIGSEGV, Segmentation fault.
0x00007ffff3638c78 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1

