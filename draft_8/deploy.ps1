<#
.SYNOPSIS
    V11.0 Automated Deployment Lifecycle (PowerShell Edition) - Fixed Syntax
.DESCRIPTION
    Orchestrates the upload, launch, tunneling, and data retrieval 
    for the IRER V11.0 HPC Suite on Azure.
#>

# --- CONFIGURATION ---
$VM_IP = "102.133.145.78"             
$VM_USER = "JAKE240501"               
$KEY_FILE = ".\IRER-V11-LAUNCH-R_ID1.pem"  
$REMOTE_DIR = "/home/$VM_USER/v11_hpc_suite"
$LOCAL_SAVE_DIR = ".\run_data_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$DURATION_SECONDS = 36000             # 10 Hours

# --- PHASE 1: CONNECTION CHECK ---
Write-Host "--- [PHASE 1] CONNECTIVITY CHECK ---" -ForegroundColor Cyan
if (-not (Test-Path $KEY_FILE)) {
    Write-Error "Key file not found! Make sure you are in the correct directory."
    exit
}

Write-Host "Testing connection to $VM_IP..."
# Using ${VM_IP} syntax to prevent parser errors
ssh -i $KEY_FILE -o StrictHostKeyChecking=no "$VM_USER@${VM_IP}" "echo '✅ Connection Verified'"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to connect. Please check your internet or Azure firewall."
    exit
}

# --- PHASE 2: UPLOAD ---
Write-Host "`n--- [PHASE 2] UPLOADING SUITE ---" -ForegroundColor Cyan
# Create remote directory structure
ssh -i $KEY_FILE "$VM_USER@${VM_IP}" "mkdir -p $REMOTE_DIR/templates"

# Upload Files 
$FilesToUpload = @("app.py", "settings.py", "core_engine.py", "worker_sncgl_sdg.py", "validation_pipeline.py", "solver_sdg.py", "aste_hunter.py", "requirements.txt")

foreach ($file in $FilesToUpload) {
    if (Test-Path $file) {
        # FIX: Used ${VM_IP} to separate the IP from the colon
        scp -i $KEY_FILE $file "$VM_USER@${VM_IP}:$REMOTE_DIR/"
    } else {
        Write-Warning "File $file not found locally. Skipping."
    }
}

# Upload Template
if (Test-Path "templates\index.html") {
    scp -i $KEY_FILE "templates\index.html" "$VM_USER@${VM_IP}:$REMOTE_DIR/templates/"
}

# --- PHASE 3: REMOTE LAUNCH (Robust Version) ---
Write-Host "`n--- [PHASE 3] REMOTE INSTALL & LAUNCH ---" -ForegroundColor Cyan
$RemoteScript = @"
    cd $REMOTE_DIR
    echo 'Starting Remote Setup in: $REMOTE_DIR'

    echo 'Installing System Dependencies...'
    sudo apt-get update -qq > /dev/null
    sudo apt-get install -y python3-pip python3-venv -qq > /dev/null
    
    echo 'Setting up Virtual Environment (venv)...'
    python3 -m venv venv
    
    # Define absolute paths for reliability
    VENV_PYTHON="\$REMOTE_DIR/venv/bin/python3"
    VENV_PIP="\$REMOTE_DIR/venv/bin/pip3"

    echo 'Installing Python Libraries with venv pip...'
    \$VENV_PIP install -r requirements.txt > /dev/null 2>&1
    
    echo 'Creating Data Directories...'
    mkdir -p input_configs simulation_data provenance_reports logs

    echo 'Launching Control Hub on Port 8080...'
    # Use pkill to ensure a clean start
    pkill -f 'app.py' || true 
    
    # CRITICAL: Launch app using the venv's absolute Python path
    nohup \$VENV_PYTHON app.py > app.log 2>&1 &
    
    echo 'Launch command sent. Waiting 5 seconds for initialization...'
    sleep 5
    
    echo '--- REMOTE LAUNCH FINISHED ---'

"@

# CRITICAL FIX: Replace Windows CRLF (`r`n) with Linux LF (`n`)
$LinuxScript = $RemoteScript.Replace("`r`n","`n")

# Use $LinuxScript for execution
ssh -i $KEY_FILE "$VM_USER@${VM_IP}" $LinuxScript

# --- PHASE 4: TUNNELING ---
Write-Host "`n--- [PHASE 4] ESTABLISHING TUNNEL ---" -ForegroundColor Cyan
Write-Host "Mapping Remote:8080 -> Local:8080"
Write-Host "Starting background tunnel..."

# Start SSH Tunnel in background job
$TunnelJob = Start-Job -ScriptBlock {
    param($key, $user, $ip)
    ssh -i $key -N -L 8080:localhost:8080 "$user@$ip"
} -ArgumentList $KEY_FILE, $VM_USER, $VM_IP

Start-Sleep -Seconds 5

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "🚀 SYSTEM LIVE!"
Write-Host "--------------------------------------------------------"
Write-Host "1. Open your browser: http://localhost:8080"
Write-Host "2. Click 'Start New Hunt'"
Write-Host "--------------------------------------------------------"
Write-Host "⏳ Running for 10 hours. DO NOT CLOSE THIS WINDOW."
Write-Host "========================================================"

# Wait Loop
Start-Sleep -Seconds $DURATION_SECONDS

# --- PHASE 5: SHUTDOWN & RETRIEVE ---
Write-Host "`n--- [PHASE 5] TIMEOUT REACHED - RETRIEVING DATA ---" -ForegroundColor Cyan

# Stop Remote
ssh -i $KEY_FILE "$VM_USER@${VM_IP}" "pkill -f app.py"

# Download
New-Item -ItemType Directory -Force -Path $LOCAL_SAVE_DIR | Out-Null
Write-Host "Downloading artifacts to $LOCAL_SAVE_DIR..."

try {
    # FIX: Used ${VM_IP} to separate the IP from the colon here as well
    scp -i $KEY_FILE -r "$VM_USER@${VM_IP}:$REMOTE_DIR/simulation_data" "$LOCAL_SAVE_DIR\"
    scp -i $KEY_FILE -r "$VM_USER@${VM_IP}:$REMOTE_DIR/provenance_reports" "$LOCAL_SAVE_DIR\"
    scp -i $KEY_FILE "$VM_USER@${VM_IP}:$REMOTE_DIR/simulation_ledger.csv" "$LOCAL_SAVE_DIR\"
} catch {
    Write-Error "Error downloading data: $_"
}

# Cleanup Tunnel
Stop-Job $TunnelJob
Remove-Job $TunnelJob

Write-Host "`n✅ RUN COMPLETE. Data saved." -ForegroundColor Green
Write-Host "⚠️ GO TO AZURE PORTAL AND STOP THE VM NOW!" -ForegroundColor Yellow