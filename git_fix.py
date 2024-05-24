import subprocess
import sys

def run_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(result.stderr.decode())
        sys.exit(1)
    return result.stdout.decode()

def main(batch_size):
    # Get the list of commits
    commit_list = run_command("git rev-list --reverse HEAD").strip().split('\n')
    
    # Split the commits into batches
    batches = [commit_list[i:i + batch_size] for i in range(0, len(commit_list), batch_size)]
    
    for i, batch in enumerate(batches):
        # Create a new branch for the batch
        batch_branch = f"batch_{i}"
        run_command(f"git checkout -b {batch_branch} {batch[0]}")
        
        for commit in batch[1:]:
            run_command(f"git cherry-pick {commit}")
        
        # Push the batch branch to remote
        run_command(f"git push origin {batch_branch}")
        
        # Switch back to main branch
        run_command("git checkout main")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python git_fix.py <batch_size>")
        sys.exit(1)
    
    batch_size = int(sys.argv[1])
    main(batch_size)
