#!/bin/bash
# Wraps an sbatch submission for the autolauncher CI flow.
#
# The autolauncher launch step runs the command in the commit title on the
# login node and captures only stdout into job.info, which the monitor job
# later greps for the job id. If sbatch fails, its error goes to stderr and
# job.info ends up empty, so the failure only surfaces much later (and
# cryptically) in the monitor job.
#
# This wrapper re-echoes sbatch's output on stdout and exits non-zero if the
# job id could not be extracted, so submission problems fail the launch job
# immediately with the sbatch error visible in the CI log.
set -u

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <sbatch_script>" >&2
    exit 1
fi

output=$(sbatch "$1" 2>&1)
status=$?
echo "$output"

job_id=$(echo "$output" | awk '/Submitted batch job/ {print $NF}')

if [[ $status -ne 0 || -z "$job_id" ]]; then
    echo "ERROR: sbatch submission failed for '$1' (exit code $status)." >&2
    exit 1
fi

echo "Job id: $job_id"
