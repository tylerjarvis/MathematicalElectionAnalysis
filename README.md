This repository contains files developed for studying redistricting and election results.


# Contributing


### Installations

You need the following software to contribute.
- Git from http://git-scm.com/downloads. For Windows, this will also install GitBash, a terminal-like application for running git commands.

With git installed, open a command prompt (terminal on Mac or Linux, git bash on Windows) and execute the following:
```bash
git config --global user.name "your name here"
git config --global user.email "your email here"
```
If you don't want the configuration to be global (i.e. if you have multiple git-linked accounts), use `--local` instead of `--global` within your clone of your fork (see next section).

### Setup

1. If you don't have one already, make a GitHub account at https://github.com.
Choose an intelligible username (so team members know who you are) and memorable password.
2. Tell me your usernema so that I can give you access to this repository.
3. Fork this repository by clicking the **Fork** button at the top right corner of this page (stop and read https://help.github.com/articles/about-forks/ if you don't know what a fork is).
You may name your fork whatever you want.
4. Clone the fork of your repository to your machine and add the original repository as a remote. If `username` is your GitHub username and `forkname` is the name of your new fork, do the following:
```bash
# Navigate to wherever you want to put your folder, then clone the fork.
cd ~/Desktop
git clone https://username@github.com/username/forkname.git

# Add the original repository as a remote.
cd forkname
git remote add upstream https://github.com/tylerjarvis/MathematicalElectionAnalysis.git
```
Within your repository folder, `origin` refers to your fork repository and `upstream` refers to this source repository.

![xkcd:git](https://imgs.xkcd.com/comics/git.png)

### Workflow

Git is all about careful coordination and communication.
You work on the labs on your computer copy and make sure your online copy matches them, then you use your online copy to submit requests for changes to the online repository.
In turn, you update your computer copy to make sure it has all new changes from the source repository.

##### Sync your Fork with the Source

Open command prompt (or git bash) and cd into your repository folder.
Run `git branch` to check your current branch.
If a star appears next to `master`, you are on the default branch, called master.
**NEVER MAKE EDITS WHILE ON MASTER;** keep it as a clean copy of the source repository.
Update `master` with the following commands.
```bash
git checkout master                    # Switch to the master branch.
git pull upstream master               # Get updates from the source repo.
git push origin master                 # Push updates to your fork.
```
##### Make Edits

1. Create a new branch for editing.
```bash
git checkout master                    # Switch to the master branch.
git checkout -b newbranch               # Make a new branch and switch to it. Pick a good branch name.
```
**Only make new branches from the `master` branch** (when you make a new brach with `git branch`, it "branches off" of the current branch).
To switch between branches, use `git checkout <branchname>`.

2. Make edits to the labs, saving your progress at reasonable segments.
```bash
git add filethatyouchanged
git commit -m "<a DESCRIPTIVE commit message>"
```
3. Push your working branch to your fork once you're done making edits.
```bash
git push origin newbranch               # Make sure the branch name matches your current branch
```
4. Create a pull request.
Go to the page for this repository.
Click the green **New Pull Request** button.

##### Clean Up

After your pull request is merged, you need to get those changes (and any other changes from other contributors) into your `master` branch and delete your working branch.
If you continue to work on the same branch without deleting it, you are risking major merge conflicts.

1. Update the `master` branch.
```bash
git checkout master            # Switch to master.
git pull upstream master       # Pull changes from the source repo.
git push origin master	        # Push updates to your fork.
```
2. Delete your working branch. **Always do this after (and only after) your pull request is merged.**
```bash
git checkout newbranch          # Switch to the branch where your now-merged edits came from.
git merge master               # Reconcile the commits in newbranch and master.
git checkout master            # Switch back to master.
git branch -d newbranch         # Delete the working branch.
git push origin :newbranch      # Tell your fork to delete the example branch.
```

See https://help.github.com/articles/creating-a-pull-request-from-a-fork/ for more info on pull requests and the "Git Help" file (currently in Google Drive) for more details and examples on git commands for lab development.
GitHub's [git cheat sheet](https://services.github.com/on-demand/downloads/github-git-cheat-sheet.pdf) may also be helpful.

![xkcd:git commit](https://imgs.xkcd.com/comics/git_commit.png)
