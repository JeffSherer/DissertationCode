import os
import subprocess

def run_command(command):
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(output.decode())
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e.output.decode()}")
        return False

# Define batch size
batch_size = 20  # Adjust batch size as needed

# Get a list of all commits
commit_list = subprocess.check_output("git rev-list HEAD", shell=True).decode().splitlines()

# Process commits in batches
for i in range(1, len(commit_list), batch_size):
    batch_commits = commit_list[i:i + batch_size]
    batch_branch = f"batch_{i // batch_size + 1}"
    print(f"Processing {batch_branch} with commits {batch_commits}")

    # Checkout a new branch for the batch
    if not run_command(f"git checkout -b {batch_branch}"):
        break

    # Start an interactive rebase for the batch
    if not run_command(f"git rebase -i {batch_commits[-1]}^"):
        break

    # Push the batch branch to the remote repository
    if not run_command(f"git push origin {batch_branch}"):
        break

    # Checkout the main branch before starting the next batch
    if not run_command("git checkout main"):
        break

print("Completed splitting and pushing commits.")
