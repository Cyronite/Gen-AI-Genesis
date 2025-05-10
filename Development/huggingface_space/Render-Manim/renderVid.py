import subprocess
import os

def runCommand(command, working_dir=None):
    """
    Run a shell command and return the result

    Args:
        command (list): Command to run as a list of arguments
        working_dir (str, optional): Directory to run the command in

    Returns:
        tuple: (success, output, error)
    """
    print("runCmd got called")
    print(f"Command: {command}")
    print(f"Working directory: {working_dir}")
    
    try:
        # Validate the command is a list
        if not isinstance(command, list):
            return False, "", f"Command must be a list, got {type(command)}"
        
        # Validate working directory exists if specified
        if working_dir and not os.path.exists(working_dir):
            return False, "", f"Working directory does not exist: {working_dir}"
            
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,  # Don't raise an exception on non-zero return codes
            cwd=working_dir  # Set the working directory if specified
        )

        success = result.returncode == 0
        if not success:
            print(f"Command failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            
        return success, result.stdout, result.stderr

    except Exception as e:
        print(f"Exception running command: {str(e)}")
        return False, "", str(e)
