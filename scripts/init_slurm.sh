#!/bin/bash
# =============================================================================
# SLURM 集群初始化脚本（多节点通用版）
#
# 自动检测当前节点角色（slurmctld / slurmd），动态生成 slurm.conf，
# 并启动对应的守护进程。
#
# 角色判断逻辑：
#   - 如果 hostname 包含 "slurmctld" → 控制节点
#   - 如果 hostname 包含 "slurmd"     → 计算节点
#   - 否则                            → 单机模式（本机同时运行 slurmctld + slurmd）
#
# 多节点发现机制：
#   平台创建的 pod 命名规则为 hpc-job-{JOB_ID}-cluster-{role}-{N}
#   通过 Kubernetes DNS 互相发现：
#     slurmctld: hpc-job-{JOB_ID}-cluster-slurmctld-0.hpc-job-{JOB_ID}-cluster-slurmctld
#     slurmd-N:  hpc-job-{JOB_ID}-cluster-slurmd-N.hpc-job-{JOB_ID}-cluster-slurmd
# =============================================================================

set -euo pipefail

HOSTNAME=$(hostname)
echo "[INFO] Hostname: ${HOSTNAME}"

# ---------------------------------------------------------------------------
# 1. 启动 munge（所有节点都需要）
# ---------------------------------------------------------------------------
mkdir -p /run/munge /var/log/munge /var/lib/munge
chown munge:munge /run/munge /var/log/munge /var/lib/munge 2>/dev/null || true
munged -f
sleep 1

if ! munge -n | unmunge >/dev/null 2>&1; then
    echo "[ERROR] munge 认证启动失败"
    exit 1
fi
echo "[OK] munge started"

# ---------------------------------------------------------------------------
# 2. 创建必要目录
# ---------------------------------------------------------------------------
mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /run/slurm
chown slurm:slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /run/slurm 2>/dev/null || true

# ---------------------------------------------------------------------------
# 3. 判断节点角色
# ---------------------------------------------------------------------------
if echo "${HOSTNAME}" | grep -q "slurmctld"; then
    ROLE="controller"
elif echo "${HOSTNAME}" | grep -q "slurmd"; then
    ROLE="compute"
else
    ROLE="standalone"
fi
echo "[INFO] Detected role: ${ROLE}"

# ---------------------------------------------------------------------------
# 4. 动态生成 slurm.conf
# ---------------------------------------------------------------------------
CPU_COUNT=$(nproc)
MEM_MB=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 ))

# 推断 Job ID 和集群 DNS 前缀
# hostname 格式: hpc-job-{JOB_ID}-cluster-{role}-{N}
JOB_ID=$(echo "${HOSTNAME}" | sed -n 's/hpc-job-\([0-9]*\)-cluster-.*/\1/p')
NAMESPACE="ai4education"

if [ -n "${JOB_ID}" ] && [ "${ROLE}" != "standalone" ]; then
    # 多节点模式：通过 K8s DNS 发现
    CTLD_HOST="hpc-job-${JOB_ID}-cluster-slurmctld-0.hpc-job-${JOB_ID}-cluster-slurmctld.${NAMESPACE}.svc.cluster.local"
    CTLD_PORT=6817

    # 探测计算节点数量：尝试 DNS 解析 slurmd-0, slurmd-1, ...
    COMPUTE_NODES=""
    COMPUTE_NODE_DEFS=""
    NODE_IDX=0
    while true; do
        NODE_FQDN="hpc-job-${JOB_ID}-cluster-slurmd-${NODE_IDX}.hpc-job-${JOB_ID}-cluster-slurmd.${NAMESPACE}.svc.cluster.local"
        if getent hosts "${NODE_FQDN}" >/dev/null 2>&1; then
            NODE_IP=$(getent hosts "${NODE_FQDN}" | awk '{print $1}')
            COMPUTE_NODES="${COMPUTE_NODES} ${NODE_FQDN}"
            COMPUTE_NODE_DEFS="${COMPUTE_NODE_DEFS}NodeName=${NODE_FQDN} CPUs=${CPU_COUNT} RealMemory=${MEM_MB} State=UNKNOWN
"
            NODE_IDX=$((NODE_IDX + 1))
        else
            break
        fi
    done

    # 如果 DNS 还没就绪，回退：用 SLURM 环境变量推断节点数
    if [ "${NODE_IDX}" -eq 0 ] && [ -n "${SLURM_NNODES:-}" ]; then
        echo "[WARN] DNS 发现失败，使用 SLURM_NNODES=${SLURM_NNODES} 构造节点列表"
        for i in $(seq 0 $((SLURM_NNODES - 1))); do
            NODE_FQDN="hpc-job-${JOB_ID}-cluster-slurmd-${i}.hpc-job-${JOB_ID}-cluster-slurmd.${NAMESPACE}.svc.cluster.local"
            COMPUTE_NODE_DEFS="${COMPUTE_NODE_DEFS}NodeName=${NODE_FQDN} CPUs=${CPU_COUNT} RealMemory=${MEM_MB} State=UNKNOWN
"
        done
        NODE_IDX=${SLURM_NNODES}
    fi

    echo "[INFO] 发现 ${NODE_IDX} 个计算节点"

    cat > /etc/slurm/slurm.conf <<CONF
ClusterName=olmo-hpc
SlurmctldHost=${CTLD_HOST}
AuthType=auth/munge
SlurmUser=slurm
SlurmctldPort=${CTLD_PORT}
SlurmdPort=6818
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
SlurmctldPidFile=/run/slurm/slurmctld.pid
SlurmdPidFile=/run/slurm/slurmd.pid
SlurmctldDebug=info
SlurmdDebug=info
SchedulerType=sched/builtin
SelectType=select/cons_tres
ProctrackType=proctrack/pgid
${COMPUTE_NODE_DEFS}PartitionName=debug Nodes=ALL Default=YES MaxTime=INFINITE State=UP
TaskPlugin=task/none
JobCompType=jobcomp/none
ReturnToService=2
MaxArraySize=1000
MaxJobCount=10000
CONF
    echo "[OK] Generated slurm.conf (multi-node: ${NODE_IDX} compute nodes)"

else
    # 单机模式
    cat > /etc/slurm/slurm.conf <<CONF
ClusterName=olmo-hpc
SlurmctldHost=localhost
AuthType=auth/munge
SlurmUser=slurm
SlurmctldPort=6817
SlurmdPort=6818
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
SlurmctldPidFile=/run/slurm/slurmctld.pid
SlurmdPidFile=/run/slurm/slurmd.pid
SlurmctldDebug=info
SlurmdDebug=info
SchedulerType=sched/builtin
SelectType=select/cons_tres
ProctrackType=proctrack/pgid
NodeName=localhost CPUs=${CPU_COUNT} RealMemory=${MEM_MB} State=UNKNOWN
PartitionName=debug Nodes=localhost Default=YES MaxTime=INFINITE State=UP
TaskPlugin=task/none
JobCompType=jobcomp/none
ReturnToService=2
MaxArraySize=1000
MaxJobCount=10000
CONF
    echo "[OK] Generated slurm.conf (standalone: ${CPU_COUNT} CPUs, ${MEM_MB}MB RAM)"
fi

# ---------------------------------------------------------------------------
# 5. 启动守护进程
# ---------------------------------------------------------------------------
# 清理旧进程
pkill -9 slurmctld 2>/dev/null || true
pkill -9 slurmd 2>/dev/null || true
sleep 1

case "${ROLE}" in
    controller)
        echo "[INFO] Starting as CONTROLLER (slurmctld only)"
        slurmctld -c
        sleep 2
        # 等待计算节点注册
        echo "[INFO] Waiting for compute nodes to register..."
        for i in $(seq 1 30); do
            IDLE_COUNT=$(sinfo -h -N -t idle | wc -l)
            echo "  [${i}/30] ${IDLE_COUNT} nodes idle"
            if [ "${IDLE_COUNT}" -ge "${NODE_IDX}" ] 2>/dev/null; then
                echo "[OK] All compute nodes registered!"
                break
            fi
            sleep 5
        done
        sinfo
        ;;
    compute)
        echo "[INFO] Starting as COMPUTE NODE (slurmd only)"
        slurmd
        sleep 2
        echo "[OK] slurmd started, registering with controller..."
        ;;
    standalone)
        echo "[INFO] Starting as STANDALONE (slurmctld + slurmd)"
        slurmctld -c
        sleep 2
        slurmd
        sleep 2
        sinfo
        ;;
esac

echo "[OK] SLURM initialization complete! Role=${ROLE}"
