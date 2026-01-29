#!/bin/bash
# Submit training jobs to GPU cluster
# Usage: bash train.sh [OPTIONS]

set -e

# Default configuration
DEFAULT_PARTITION="gpu1"
DEFAULT_TIME="24:00:00"
DEFAULT_JOB_NAME="dl_models_train"
DEFAULT_CPUS=4
DEFAULT_DATA_PATH="Input/raw/data.mat"
DEFAULT_OUTPUT_DIR="Output/models"
DEFAULT_LOGS_DIR="Output/logs"
DEFAULT_RESULTS_DIR="Output/results"
DEFAULT_PROJECT_ROOT="/share/home/hwxu/Projects/Client1"

# Initialize variables
PARTITION="$DEFAULT_PARTITION"
TIME_LIMIT="$DEFAULT_TIME"
JOB_NAME="$DEFAULT_JOB_NAME"
CPUS="$DEFAULT_CPUS"
GPU_ID=""
DATA_PATH="$DEFAULT_DATA_PATH"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
SHOW_HELP=false

# Function to display help information
show_help() {
    cat << EOF
Usage: bash train.sh [OPTIONS]

Submit training jobs to GPU cluster using SLURM.

OPTIONS:
    -g, --gpu ID           Specify GPU ID (e.g., 0, 1, 2, 5). Only 1 GPU will be used.
    -p, --partition NAME   Specify SLURM partition (default: $DEFAULT_PARTITION)
    -t, --time TIME        Specify time limit (default: $DEFAULT_TIME)
    -n, --job-name NAME    Specify job name (default: $DEFAULT_JOB_NAME)
    -c, --cpus NUM         Specify number of CPUs (default: $DEFAULT_CPUS)
    -d, --data-path PATH   Specify data path (default: $DEFAULT_DATA_PATH)
    -o, --output-dir DIR    Specify output directory (default: $DEFAULT_OUTPUT_DIR)
    -h, --help             Show this help message

EXAMPLES:
    # Use default configuration
    bash train.sh

    # Specify GPU and partition
    bash train.sh -g 5 -p gpu1

    # Specify time limit
    bash train.sh -t 48:00:00

    # Full example
    bash train.sh -g 5 -p gpu1 -t 24:00:00 -n my_training_job

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -g|--gpu)
                if [ -z "$2" ] || [[ "$2" =~ ^- ]]; then
                    echo "Error: -g/--gpu requires a GPU ID"
                    exit 1
                fi
                GPU_ID="$2"
                shift 2
                ;;
            -p|--partition)
                if [ -z "$2" ] || [[ "$2" =~ ^- ]]; then
                    echo "Error: -p/--partition requires a partition name"
                    exit 1
                fi
                PARTITION="$2"
                shift 2
                ;;
            -t|--time)
                if [ -z "$2" ] || [[ "$2" =~ ^- ]]; then
                    echo "Error: -t/--time requires a time limit"
                    exit 1
                fi
                TIME_LIMIT="$2"
                shift 2
                ;;
            -n|--job-name)
                if [ -z "$2" ] || [[ "$2" =~ ^- ]]; then
                    echo "Error: -n/--job-name requires a job name"
                    exit 1
                fi
                JOB_NAME="$2"
                shift 2
                ;;
            -c|--cpus)
                if [ -z "$2" ] || [[ "$2" =~ ^- ]]; then
                    echo "Error: -c/--cpus requires a number"
                    exit 1
                fi
                CPUS="$2"
                shift 2
                ;;
            -d|--data-path)
                if [ -z "$2" ] || [[ "$2" =~ ^- ]]; then
                    echo "Error: -d/--data-path requires a path"
                    exit 1
                fi
                DATA_PATH="$2"
                shift 2
                ;;
            -o|--output-dir)
                if [ -z "$2" ] || [[ "$2" =~ ^- ]]; then
                    echo "Error: -o/--output-dir requires a directory"
                    exit 1
                fi
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -h|--help)
                SHOW_HELP=true
                shift
                ;;
            *)
                echo "Error: Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Try to find project root
# Script is in Libs/scripts/, so project root should be two levels up
if [ -f "$SCRIPT_DIR/train_all_models.py" ]; then
    # Script is in Libs/scripts/, project root is two levels up
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
elif [ -f "$DEFAULT_PROJECT_ROOT/Libs/scripts/train_all_models.py" ]; then
    PROJECT_ROOT="$DEFAULT_PROJECT_ROOT"
else
    # Try current directory
    PROJECT_ROOT="$(pwd)"
    if [ ! -f "$PROJECT_ROOT/Libs/scripts/train_all_models.py" ]; then
        # Try going up from script directory
        PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
        if [ ! -f "$PROJECT_ROOT/Libs/scripts/train_all_models.py" ]; then
            echo "Error: Cannot find project root. Please run from project root or Libs/scripts directory."
            echo "  Script directory: $SCRIPT_DIR"
            echo "  Current directory: $(pwd)"
            echo "  Expected project root: $DEFAULT_PROJECT_ROOT"
            exit 1
        fi
    fi
fi

# Verify project root is correct
if [ ! -f "$PROJECT_ROOT/Libs/scripts/train_all_models.py" ]; then
    echo "Error: Invalid project root detected: $PROJECT_ROOT"
    echo "  Cannot find Libs/scripts/train_all_models.py"
    exit 1
fi

# Parse arguments
parse_args "$@"

# Show help if requested
if [ "$SHOW_HELP" = true ]; then
    show_help
    exit 0
fi

# Validate GPU ID if provided
if [ -n "$GPU_ID" ]; then
    if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
        echo "Error: GPU ID must be a number (got: $GPU_ID)"
        exit 1
    fi
fi

# Validate CPUs
if ! [[ "$CPUS" =~ ^[0-9]+$ ]] || [ "$CPUS" -lt 1 ]; then
    echo "Error: CPUs must be a positive number (got: $CPUS)"
    exit 1
fi

# Validate time format (basic check)
if ! [[ "$TIME_LIMIT" =~ ^[0-9]+:[0-9]{2}:[0-9]{2}$ ]] && ! [[ "$TIME_LIMIT" =~ ^[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "Warning: Time limit format may be incorrect (expected HH:MM:SS or D-HH:MM:SS, got: $TIME_LIMIT)"
fi

# Create necessary directories
# Logs: Output/logs (SLURM output/error logs and Python experiment logs)
# Results: Output/results (result tables in JSON format: all_dl_models_results.json, *_metrics.json)
# Models: Output/models (saved model files)
mkdir -p "$PROJECT_ROOT/Output/logs" || exit 1
mkdir -p "$PROJECT_ROOT/Output/models" || exit 1
mkdir -p "$PROJECT_ROOT/Output/results" || exit 1

# Generate temporary job script
JOB_SCRIPT=$(mktemp /tmp/train_job_XXXXXX.sh)
trap "rm -f $JOB_SCRIPT" EXIT

# Build GPU resource specification - Always use exactly 1 GPU
# Note: SLURM --gres format is "gpu:N" for N GPUs, or "gpu:type:N" for specific GPU type
# We always request exactly 1 GPU regardless of GPU_ID specification
GPU_SPEC="gpu:1"

# Store GPU_ID for use in generated script (if specified)
GENERATED_GPU_ID="$GPU_ID"

# Generate job script content
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --gres=$GPU_SPEC
#SBATCH --time=$TIME_LIMIT
#SBATCH --chdir=$PROJECT_ROOT
#SBATCH --output=$PROJECT_ROOT/Output/logs/train_%j.out
#SBATCH --error=$PROJECT_ROOT/Output/logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com

# Train all deep learning baseline models
# This script is auto-generated by train.sh

set -e

# Set project root (use absolute path from submission time)
PROJECT_ROOT="$PROJECT_ROOT"

# Change to project root directory
cd "\$PROJECT_ROOT" || {
    echo "Error: Failed to change to project root: \$PROJECT_ROOT"
    echo "Current directory: \$(pwd)"
    exit 1
}

# Verify we're in the right directory
if [ ! -f "Libs/scripts/train_all_models.py" ]; then
    echo "Error: Cannot find project root. Current directory: \$(pwd)"
    echo "Expected file: Libs/scripts/train_all_models.py"
    echo "PROJECT_ROOT variable: \$PROJECT_ROOT"
    ls -la . || true
    exit 1
fi

# Create necessary directories
# Logs: Output/logs
# Results: Output/results  
# Models: Output/models
LOGS_DIR="Output/logs"
RESULTS_DIR="Output/results"
MODELS_DIR="$OUTPUT_DIR"

mkdir -p "\$PROJECT_ROOT/\$LOGS_DIR" || exit 1
mkdir -p "\$PROJECT_ROOT/\$RESULTS_DIR" || exit 1
mkdir -p "\$PROJECT_ROOT/\$MODELS_DIR" || exit 1

# Activate conda environment
source /share/home/hwxu/miniconda3/bin/activate hw

# Set environment variables
export PYTHONPATH="\$PROJECT_ROOT:\$PYTHONPATH"

# Limit to 1 GPU - Set CUDA_VISIBLE_DEVICES if GPU_ID is specified
# Otherwise, SLURM will assign 1 GPU automatically
if [ -n "$GENERATED_GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GENERATED_GPU_ID"
    echo "Using GPU ID: $GENERATED_GPU_ID (1 GPU only)"
else
    echo "Using auto-assigned GPU (1 GPU only)"
fi

# Fix GLIBCXX version issue by using conda's libstdc++
CONDA_PREFIX=\$(conda info --base)
export LD_LIBRARY_PATH="\$CONDA_PREFIX/envs/hw/lib:\$LD_LIBRARY_PATH"

# Print job information
echo "=========================================="
echo "GPU Training Job Information"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node: \$SLURM_NODELIST"
echo "GPU: \${CUDA_VISIBLE_DEVICES:-auto-assigned}"
echo "Partition: \$SLURM_JOB_PARTITION"
echo "Time Limit: \$SLURM_JOB_TIME_LIMIT"
echo "Working Directory: \$PROJECT_ROOT"
echo "Date: \$(date)"
echo "=========================================="
echo ""

# Check GPU availability
echo "Checking GPU availability..."
echo "GPU Resource: $GPU_SPEC (1 GPU only)"
nvidia-smi || echo "Warning: nvidia-smi not available"
echo ""

# Run training for all models
echo "=========================================="
echo "Starting Training for All Models"
echo "=========================================="
echo ""

# Run training - results will be saved to Output/results
# Logs will be written to Output/logs
python3 -m Libs.scripts.train_all_models \\
    --data_path $DATA_PATH \\
    --output_dir $OUTPUT_DIR

TRAIN_EXIT_CODE=\$?

echo ""
echo "=========================================="
echo "Training Completed"
echo "=========================================="
echo "Date: \$(date)"
echo ""

# Check if training was successful
if [ \$TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"
    echo ""
    echo "Results saved to:"
    echo "  - Models: \$PROJECT_ROOT/\$MODELS_DIR/"
    echo "  - Results (tables): \$PROJECT_ROOT/\$RESULTS_DIR/"
    echo "    * all_dl_models_results.json"
    echo "    * Individual model metrics: *_metrics.json"
    echo "  - Logs: \$PROJECT_ROOT/\$LOGS_DIR/"
    echo "    * train_\$SLURM_JOB_ID.out (SLURM output)"
    echo "    * train_\$SLURM_JOB_ID.err (SLURM errors)"
    echo "    * experiment.log (Python logs)"
else
    echo "✗ Training failed with exit code \$TRAIN_EXIT_CODE"
    exit \$TRAIN_EXIT_CODE
fi
EOF

# Make job script executable
chmod +x "$JOB_SCRIPT"

# Display submission information
echo "=========================================="
echo "Submitting Training Job to GPU Cluster"
echo "=========================================="
echo "Job Name: $JOB_NAME"
echo "Partition: $PARTITION"
echo "Time Limit: $TIME_LIMIT"
echo "CPUs: $CPUS"
if [ -n "$GPU_ID" ]; then
    echo "GPU ID: $GPU_ID"
else
    echo "GPU: Auto-assigned"
fi
echo "Data Path: $DATA_PATH"
echo "Models Dir: $OUTPUT_DIR"
echo "Logs Dir: $DEFAULT_LOGS_DIR"
echo "Results Dir: $DEFAULT_RESULTS_DIR"
echo "Project Root: $PROJECT_ROOT"
echo "=========================================="
echo ""

# Submit job
echo "Submitting job..."
JOB_ID=$(sbatch "$JOB_SCRIPT" 2>&1 | grep -oP '\d+' | head -1)

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit job. Check SLURM configuration."
    exit 1
fi

echo "✓ Job submitted successfully!"
echo ""
echo "Job ID: $JOB_ID"
echo "Job Script: $JOB_SCRIPT"
echo ""
echo "To check job status:"
echo "  squeue -j $JOB_ID"
echo ""
echo "To view job output:"
echo "  tail -f $PROJECT_ROOT/$DEFAULT_LOGS_DIR/train_${JOB_ID}.out"
echo ""
echo "To view job errors:"
echo "  tail -f $PROJECT_ROOT/$DEFAULT_LOGS_DIR/train_${JOB_ID}.err"
echo ""
echo "To view Python logs:"
echo "  tail -f $PROJECT_ROOT/$DEFAULT_LOGS_DIR/experiment.log"
echo ""
echo "To view results:"
echo "  ls -lh $PROJECT_ROOT/$DEFAULT_RESULTS_DIR/"
echo ""
echo "To cancel job:"
echo "  scancel $JOB_ID"
echo ""
