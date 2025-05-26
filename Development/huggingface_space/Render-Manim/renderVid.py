import subprocess
import os

def runCommand(command, working_dir=None, shell=False):
    """
    Run a shell command and return the result

    Args:
        command (list or str): Command to run as a list of arguments or string (if shell=True)
        working_dir (str, optional): Directory to run the command in
        shell (bool, optional): Whether to run command in the shell. Default is False.

    Returns:
        tuple: (success, output, error)
    """
    print("runCmd got called")
    print(f"Command: {command}")
    print(f"Working directory: {working_dir}")
    
    try:
        # Validate the command type based on shell parameter
        if shell and isinstance(command, list):
            command = ' '.join(command)
        elif not shell and not isinstance(command, list):
            if isinstance(command, str):
                # Convert to list if it's a string but shell=False
                command = command.split()
            else:
                return False, "", f"Command must be a list or string, got {type(command)}"
        
        # Validate working directory exists if specified
        if working_dir and not os.path.exists(working_dir):
            return False, "", f"Working directory does not exist: {working_dir}"
            
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,  # Don't raise an exception on non-zero return codes
            cwd=working_dir,  # Set the working directory if specified
            shell=shell  # Use shell if specified
        )

        success = result.returncode == 0
        if not success:
            print(f"Command failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            
        return success, result.stdout, result.stderr

    except Exception as e:
        print(f"Exception running command: {str(e)}")
        return False, "", str(e)
