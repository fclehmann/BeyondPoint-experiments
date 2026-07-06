import subprocess


def get_sacct_job_status(job_id):
    """
    Get the status of a SLURM job using sacct.
    Returns the status as a string or None if the job is not found.
    """
    try:
        # Execute the sacct command to get the status of the job
        result = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=State", "--noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            # Strip and process the output
            status = result.stdout.strip().splitlines()
            if status:
                return status[0].strip()  # Return the first status found
            else:
                return None  # Job not found in sacct
        else:
            # Handle errors
            raise RuntimeError(f"sacct error: {result.stderr.strip()}")
    except Exception as e:
        raise RuntimeError(f"Failed to query job status: {str(e)}")