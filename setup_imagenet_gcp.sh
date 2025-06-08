#!/bin/bash

# ==============================================================================
# ImageNet Training Setup Script for GCP
# ==============================================================================

set -e  # Exit on any error

# Configuration Variables
PROJECT_ID=${PROJECT_ID:-"your-gcp-project-id"}
ZONE=${ZONE:-"us-central1-a"}
INSTANCE_NAME=${INSTANCE_NAME:-"imagenet-training-instance"}
MACHINE_TYPE=${MACHINE_TYPE:-"n1-standard-8"}
GPU_TYPE=${GPU_TYPE:-"nvidia-tesla-v100"}
GPU_COUNT=${GPU_COUNT:-1}
BOOT_DISK_SIZE=${BOOT_DISK_SIZE:-"100GB"}
BUCKET_NAME=${BUCKET_NAME:-"${PROJECT_ID}-imagenet-data"}
KAGGLE_COMPETITION=${KAGGLE_COMPETITION:-"imagenet-object-localization-challenge"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if kaggle is installed
    if ! command -v kaggle &> /dev/null; then
        log_error "Kaggle CLI is not installed. Installing..."
        pip install kaggle
    fi
    
    # Check if Kaggle credentials exist
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        log_error "Kaggle credentials not found. Please place your kaggle.json in ~/.kaggle/"
        log_info "You can download it from: https://www.kaggle.com/account"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Set up GCP project
setup_gcp_project() {
    log_info "Setting up GCP project: $PROJECT_ID"
    
    # Set the project
    gcloud config set project $PROJECT_ID
    
    # Enable necessary APIs
    log_info "Enabling required GCP APIs..."
    gcloud services enable compute.googleapis.com
    gcloud services enable storage-api.googleapis.com
    gcloud services enable ml.googleapis.com
    
    log_success "GCP project setup complete"
}

# Create GCS bucket
create_bucket() {
    log_info "Creating GCS bucket: $BUCKET_NAME"
    
    # Check if bucket exists
    if gsutil ls -b gs://$BUCKET_NAME 2>/dev/null; then
        log_warning "Bucket $BUCKET_NAME already exists"
    else
        gsutil mb gs://$BUCKET_NAME
        log_success "Bucket $BUCKET_NAME created"
    fi
}

# Download and prepare ImageNet dataset
download_imagenet() {
    log_info "Downloading ImageNet dataset from Kaggle competition..."
    
    # Create local directory for dataset
    mkdir -p ./imagenet_data
    cd ./imagenet_data
    
    # Check if user has accepted competition rules
    log_info "Checking Kaggle competition access..."
    if ! kaggle competitions list | grep -q "$KAGGLE_COMPETITION"; then
        log_warning "You may need to accept the competition rules first"
        log_info "Please visit: https://www.kaggle.com/c/$KAGGLE_COMPETITION"
        log_info "Click 'Join Competition' and accept the rules, then try again"
    fi
    
    # Download ImageNet dataset from competition
    log_info "Downloading ImageNet Object Localization Challenge dataset..."
    log_info "This is a large dataset (~150GB), download will take time..."
    
    # Download all competition files
    kaggle competitions download -c "$KAGGLE_COMPETITION"
    
    # Check what files were downloaded
    log_info "Downloaded files:"
    ls -lh *.zip 2>/dev/null || ls -lh
    
    # Extract the main dataset files
    log_info "Extracting dataset files..."
    
    # Extract all zip files
    for zipfile in *.zip; do
        if [ -f "$zipfile" ]; then
            log_info "Extracting $zipfile..."
            unzip -q "$zipfile"
        fi
    done
    
    # Check the extracted structure
    log_info "Final dataset structure:"
    find . -maxdepth 3 -type d | head -20
    
    # Look for the ILSVRC directory structure
    if [ -d "ILSVRC" ]; then
        log_success "Found ILSVRC directory structure"
    elif [ -d "Data" ]; then
        log_success "Found Data directory structure"
    else
        log_warning "Unexpected directory structure, listing contents:"
        ls -la
    fi
    
    cd ..
    log_success "ImageNet dataset downloaded and extracted"
}

# Convert ImageNet to TFDS format and upload to GCS
prepare_tfds_imagenet() {
    log_info "Preparing ImageNet for TFDS format..."
    
    # Create TFDS preparation script
    cat > prepare_imagenet_tfds.py << 'EOF'
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import shutil
from pathlib import Path

def find_imagenet_data():
    """Find the ImageNet data directory structure."""
    possible_paths = [
        "./imagenet_data/ILSVRC/Data/CLS-LOC",
        "./imagenet_data/ILSVRC",
        "./imagenet_data/Data/CLS-LOC", 
        "./imagenet_data/Data",
        "./imagenet_data"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it has train/val structure
            train_path = os.path.join(path, "train")
            val_path = os.path.join(path, "val")
            if os.path.exists(train_path) and os.path.exists(val_path):
                print(f"Found ImageNet data at: {path}")
                return path
    
    # List directory structure for debugging
    print("Available directories in ./imagenet_data:")
    for root, dirs, files in os.walk("./imagenet_data"):
        level = root.replace("./imagenet_data", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 3:  # Don't go too deep
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files)-5} more files")
    
    return None

def prepare_imagenet_tfds():
    """Prepare ImageNet dataset for TFDS."""
    
    # Find the ImageNet data
    imagenet_path = find_imagenet_data()
    
    if not imagenet_path:
        print("ERROR: Could not find ImageNet data with train/val structure")
        print("Please check the downloaded data structure")
        return False
    
    # Create TFDS data directory
    tfds_data_dir = "./tfds_data"
    os.makedirs(tfds_data_dir, exist_ok=True)
    
    try:
        print(f"Using ImageNet data from: {imagenet_path}")
        
        # Prepare ImageNet using TFDS with manual directory
        builder = tfds.builder("imagenet2012", data_dir=tfds_data_dir)
        
        # Set up manual directory for TFDS
        manual_dir = os.path.abspath(imagenet_path)
        download_config = tfds.download.DownloadConfig(
            manual_dir=manual_dir,
            verify_ssl=False
        )
        
        print("Starting TFDS preparation (this may take a while)...")
        builder.download_and_prepare(
            download_config=download_config
        )
        
        print("SUCCESS: ImageNet TFDS dataset prepared")
        return True
        
    except Exception as e:
        print(f"ERROR preparing TFDS dataset: {e}")
        print("\nTrying alternative approach...")
        
        # Alternative: just upload raw data and process on the instance
        try:
            raw_data_dir = "./raw_imagenet_data"
            shutil.copytree(imagenet_path, raw_data_dir)
            print("SUCCESS: Raw ImageNet data prepared for upload")
            return True
        except Exception as e2:
            print(f"ERROR with alternative approach: {e2}")
            return False

if __name__ == "__main__":
    success = prepare_imagenet_tfds()
    exit(0 if success else 1)
EOF

    # Run the TFDS preparation
    python prepare_imagenet_tfds.py
    
    if [ $? -eq 0 ]; then
        log_success "ImageNet dataset prepared"
        
        # Upload processed data to GCS
        log_info "Uploading data to GCS bucket..."
        
        # Upload TFDS data if it exists
        if [ -d "./tfds_data" ]; then
            log_info "Uploading TFDS data..."
            gsutil -m cp -r ./tfds_data gs://$BUCKET_NAME/
        fi
        
        # Upload raw data if it exists
        if [ -d "./raw_imagenet_data" ]; then
            log_info "Uploading raw ImageNet data..."
            gsutil -m cp -r ./raw_imagenet_data gs://$BUCKET_NAME/
        fi
        
        # Also upload the original extracted data as backup
        log_info "Uploading original ImageNet data as backup..."
        gsutil -m cp -r ./imagenet_data gs://$BUCKET_NAME/imagenet_original/
        
        log_success "All data uploaded to GCS"
    else
        log_error "Failed to prepare ImageNet dataset"
        exit 1
    fi
}

# Create GPU instance
create_gpu_instance() {
    log_info "Creating GPU instance: $INSTANCE_NAME"
    
    # Check if instance already exists
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE 2>/dev/null; then
        log_warning "Instance $INSTANCE_NAME already exists"
        return
    fi
    
    # Create the instance with GPU
    gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --boot-disk-size=$BOOT_DISK_SIZE \
        --boot-disk-type=pd-ssd \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata="install-nvidia-driver=True"
    
    log_success "GPU instance created: $INSTANCE_NAME"
    
    # Wait for instance to be ready
    log_info "Waiting for instance to be ready..."
    sleep 60
    
    # Copy training files to instance
    log_info "Copying training files to instance..."
    gcloud compute scp train.py $INSTANCE_NAME:~/ --zone=$ZONE
    gcloud compute scp setup_training_env.sh $INSTANCE_NAME:~/ --zone=$ZONE
    
    log_success "Files copied to instance"
}

# Create training environment setup script
create_training_setup_script() {
    log_info "Creating training environment setup script..."
    
    cat > setup_training_env.sh << 'EOF'
#!/bin/bash

# Setup script to run on the GPU instance
set -e

log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

log_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

# Update system
log_info "Updating system..."
sudo apt-get update
sudo apt-get install -y python3-pip git

# Install Python dependencies
log_info "Installing Python dependencies..."
pip install --upgrade pip
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax orbax-checkpoint
pip install tensorflow tensorflow-datasets
pip install matplotlib seaborn scikit-learn pandas tqdm

# Create directories
mkdir -p ~/models
mkdir -p ~/data

# Download ImageNet data from GCS
log_info "Downloading ImageNet data from GCS..."
mkdir -p ~/data

# Try to download TFDS data first
if gsutil ls gs://BUCKET_NAME/tfds_data/ 2>/dev/null; then
    log_info "Downloading processed TFDS data..."
    gsutil -m cp -r gs://BUCKET_NAME/tfds_data ~/data/
else
    log_info "TFDS data not found, downloading raw data..."
    # Download raw data if TFDS processing failed
    if gsutil ls gs://BUCKET_NAME/raw_imagenet_data/ 2>/dev/null; then
        gsutil -m cp -r gs://BUCKET_NAME/raw_imagenet_data ~/data/
    else
        log_info "Downloading original ImageNet data..."
        gsutil -m cp -r gs://BUCKET_NAME/imagenet_original ~/data/
    fi
fi

log_success "Training environment setup complete!"
log_info "You can now run: python train.py"
EOF

    # Replace BUCKET_NAME placeholder
    sed -i "s/BUCKET_NAME/$BUCKET_NAME/g" setup_training_env.sh
    
    log_success "Training environment setup script created"
}

# Main execution function
main() {
    log_info "Starting ImageNet training setup on GCP..."
    
    # Check if we're running locally or on GCP instance
    if [ "$1" == "--setup-instance" ]; then
        log_info "Setting up training environment on GCP instance..."
        chmod +x setup_training_env.sh
        ./setup_training_env.sh
        return
    fi
    
    # Local setup steps
    check_prerequisites
    setup_gcp_project
    create_bucket
    download_imagenet
    prepare_tfds_imagenet
    create_training_setup_script
    create_gpu_instance
    
    # Setup training environment on instance
    log_info "Setting up training environment on the GPU instance..."
    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="chmod +x setup_training_env.sh && ./setup_training_env.sh"
    
    log_success "Setup complete! You can now SSH to the instance and run training:"
    log_info "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
    log_info "python train.py"
    
    log_warning "Remember to stop the instance when done to avoid charges:"
    log_info "gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
}

# Help function
show_help() {
    echo "ImageNet Training Setup Script for GCP"
    echo ""
    echo "Usage:"
    echo "  $0                    # Run full setup (download, prepare, create instance)"
    echo "  $0 --setup-instance   # Setup training environment (run on GCP instance)"
    echo "  $0 --help             # Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  PROJECT_ID            # GCP Project ID (required)"
    echo "  ZONE                  # GCP Zone (default: us-central1-a)"
    echo "  INSTANCE_NAME         # Instance name (default: imagenet-training-instance)"
    echo "  MACHINE_TYPE          # Machine type (default: n1-standard-8)"
    echo "  GPU_TYPE              # GPU type (default: nvidia-tesla-v100)"
    echo "  BUCKET_NAME           # GCS bucket name (default: PROJECT_ID-imagenet-data)"
    echo ""
    echo "Prerequisites:"
    echo "  - gcloud CLI installed and authenticated"
    echo "  - Kaggle API credentials in ~/.kaggle/kaggle.json"
    echo "  - GCP project with billing enabled"
}

# Handle command line arguments
case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    --setup-instance)
        main "$1"
        ;;
    *)
        if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" == "your-gcp-project-id" ]; then
            log_error "Please set PROJECT_ID environment variable"
            log_info "Example: export PROJECT_ID=my-gcp-project"
            exit 1
        fi
        main
        ;;
esac 