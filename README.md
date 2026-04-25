<div align="center">

<br/>

![SimpleMind Logo](Simplemind_logo.png)

<br/>

# AS-OCT Auto-Seg

### CADe Glaucoma Analysis Platform — GUI

*A clinical-grade graphical interface for automated anterior segment OCT segmentation, powered by the SimpleMind computer vision pipeline.*

<br/>


</div>

---

## About

**AS-OCT Auto-Seg** is a clinician-facing GUI built on top of the **CADe glaucoma analysis platform**. It wraps the [SimpleMind](https://gitlab.com/sm-ai-team/simplemind/-/tree/mbrown/new_sm/) pipeline — a modular computer vision framework where image processing tools are composed and connected via a **JSON plan** — into an accessible, point-and-click interface designed for clinical workflows.

**Key capabilities:**
- GUI-driven access to the SimpleMind pipeline for AS-OCT image segmentation
- JSON-plan-based tool orchestration (no coding required for clinicians)
- Automated glaucoma-relevant structure analysis from anterior segment OCT scans
- Available on WSL (Linux) and Windows (beta)

---

## Installation

> **Note:** WSL (Windows Subsystem for Linux) is required for the full installation. A native Windows executable (beta) is also available.

---

### Step 1 — Set Up WSL

If you don't have WSL installed, follow Microsoft's official guide:

[How to install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

**Quick install (Windows 11 / Windows 10 Build 19041+):**

```powershell
# Run in PowerShell (as Administrator)
wsl --install
```

Restart your machine when prompted, then open the **WSL terminal** to continue.

---

### Step 2 — Clone This Repository

Inside your WSL terminal, clone the repository:

```bash
git clone https://github.com/carooni/AS-OCT-Auto-Seg.git
cd AS-OCT-Auto-Seg
```

> **Don't have Git?** Install it first:
> ```bash
> sudo apt update && sudo apt install git -y
> ```

---

### Step 3 — Download Required Model Files

Download the required files from Google Drive and place them in the **root directory** of the cloned repo (`AS-OCT-Auto-Seg/`):

[Google Drive — Required Files](https://drive.google.com/drive/folders/1JSvcFIGqTCOLl8dlrSK55C0WOuau5nGr)

After downloading, your directory should look something like:

```
AS-OCT-Auto-Seg/
├── AS-OCT Auto-Seg          ← Linux executable
├── AS-OCT Auto-Seg.exe      ← Windows executable (beta)
├── env.yaml
├── <downloaded model files>
└── ...
```

---

### Step 4 — Install Micromamba & Create Environment

Run the following commands in your WSL terminal to install [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) and set up the Python environment:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba create -f env.yaml
micromamba activate smcore
```

> **Note:** After installing Micromamba, you may need to restart your terminal or run `source ~/.bashrc` before the `micromamba` command is recognized.

---

### Step 5 — Install SMCore

Install the SMCore backend (requires [Go](https://go.dev/doc/install)):

```bash
go install gitlab.com/hoffman-lab/core@v1.1.1
echo 'export PATH="$PATH:$HOME/go/bin"' >> ~/.bashrc
source ~/.bashrc
```

> **Don't have Go installed?**
> ```bash
> sudo apt install golang-go -y
> ```

---

## Running the Application

### WSL / Linux

Open your WSL terminal, navigate to the repository directory, and run:

```bash
cd AS-OCT-Auto-Seg
micromamba activate smcore
./"AS-OCT Auto-Seg"
```

### Windows *(Beta)*

Simply **double-click** `AS-OCT Auto-Seg.exe` in the repository folder.

> Windows support is currently in beta. For the most stable experience, use the WSL method above.

---

## Tech Stack

| Component | Technology |
|---|---|
| GUI Framework | Python (SimpleMind-integrated) |
| Computer Vision Pipeline | [SimpleMind](https://gitlab.com/sm-ai-team/simplemind/-/tree/mbrown/new_sm/) |
| Pipeline Configuration | JSON Plans |
| Environment Management | [Micromamba](https://mamba.readthedocs.io/) |
| Backend Runtime | [SMCore](https://gitlab.com/hoffman-lab/core) (Go) |
| Target Platform | WSL (Linux) / Windows |
