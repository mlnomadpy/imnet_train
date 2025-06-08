#!/bin/bash

# ImageNet Dataset Setup with Automated GCP Instance (Bash)
# =========================================================
# This script creates a temporary GCP instance, downloads ImageNet from Kaggle,
# processes the data, uploads to Cloud Storage, and cleans up automatically.

set -e

# Default configuration
PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}
ZONE=${ZONE:-"us-central1-a"}
MACHINE_TYPE=${MACHINE_TYPE:-"n1-standard-4"}

echo "🚀 ImageNet Data Setup with Automated GCP Instance"
echo "=================================================="
echo "This will:"
echo "✅ Create a temporary CPU instance"
echo "✅ Download ImageNet from Kaggle"
echo "✅ Process and upload to Cloud Storage"
echo "✅ Delete the instance automatically"
echo "💰 Cost: ~\$2-5 total"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check gcloud
    if [ -z "$PROJECT_ID" ]; then
        echo "❌ No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
        return 1
    fi
    echo "✅ GCP Project: $PROJECT_ID"
    
    # Check Kaggle credentials
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo "❌ Kaggle credentials not found!"
        echo "1. Download kaggle.json from https://www.kaggle.com/account"
        echo "2. Save it to ~/.kaggle/kaggle.json"
        echo "3. Run: chmod 600 ~/.kaggle/kaggle.json"
        return 1
    fi
    echo "✅ Kaggle credentials found"
    
    return 0
}

# Function to create startup script
create_startup_script() {
    cat > startup-script.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 ImageNet Data Processing Instance Started"
echo "============================================"

# Update system  
apt-get update -y
apt-get install -y python3-pip unzip wget curl

# Install required packages
pip3 install kaggle google-cloud-storage

# Setup Kaggle
mkdir -p ~/.kaggle
echo "$KAGGLE_JSON" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Get instance metadata
INSTANCE_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | cut -d/ -f4)

echo "📋 Instance: $INSTANCE_NAME in zone $ZONE"

# Verify Kaggle access
echo "🏆 Testing Kaggle access..."
python3 -c "
import kaggle
print('✅ Kaggle API working')
try:
    files = kaggle.api.competition_list_files('imagenet-object-localization-challenge')
    print(f'✅ ImageNet competition accessible - {len(files)} files found')
    for f in files[:3]:
        print(f'  - {f.name} ({f.size} bytes)')
except Exception as e:
    print(f'❌ Error accessing ImageNet competition: {e}')
    exit(1)
"

# Create working directory
WORK_DIR="/tmp/imagenet_work"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "📥 Starting ImageNet download and processing..."
python3 << 'PYTHON_EOF'
import os
import shutil
import subprocess
import zipfile
import tarfile
import time
from pathlib import Path
from google.cloud import storage

def log(msg):
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")

def upload_directory_to_gcs(local_dir, bucket, gcs_prefix):
    """Upload a directory to GCS with progress tracking."""
    uploaded = 0
    total_size = 0
    
    # Count files first
    total_files = 0
    for root, dirs, files in os.walk(local_dir):
        total_files += len([f for f in files if f.endswith(('.JPEG', '.jpg', '.png'))])
    
    log(f"   📊 Found {total_files} image files to upload")
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith(('.JPEG', '.jpg', '.png')):
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_dir)
                gcs_path = f"{gcs_prefix}/{rel_path}".replace("\\", "/")
                
                try:
                    blob = bucket.blob(gcs_path)
                    blob.upload_from_filename(local_path)
                    uploaded += 1
                    
                    file_size = os.path.getsize(local_path)
                    total_size += file_size
                    
                    if uploaded % 1000 == 0:
                        log(f"   📤 Progress: {uploaded}/{total_files} files ({total_size/1024/1024/1024:.1f}GB)")
                except Exception as e:
                    log(f"   ⚠️  Failed to upload {local_path}: {e}")
    
    log(f"   ✅ Upload complete: {uploaded} files ({total_size/1024/1024/1024:.1f}GB)")
    return uploaded

def download_and_process_imagenet():
    """Download and process ImageNet data."""
    import kaggle
    
    log("🔍 Getting competition files...")
    files = kaggle.api.competition_list_files("imagenet-object-localization-challenge")
    log(f"Found {len(files)} files in competition")
    
    # Focus on the main data files
    main_files = [f for f in files if any(keyword in f.name.lower() 
                 for keyword in ['ilsvrc2012_img_train', 'ilsvrc2012_img_val', 'ilsvrc2012_img_test'])]
    
    if not main_files:
        # Fallback to any large files that might contain the data
        main_files = [f for f in files if f.size > 1000000000]  # Files > 1GB
        
    log(f"Main data files to process: {[f.name for f in main_files]}")
    
    if not main_files:
        log("❌ No main data files found!")
        return False
    
    # Initialize GCS client
    client = storage.Client()
    bucket_name = os.environ['BUCKET_NAME']
    bucket = client.bucket(bucket_name)
    
    success = False
    
    for file_ref in main_files:
        filename = file_ref.name
        log(f"📦 Processing: {filename}")
        
        try:
            # Download file
            log(f"   ⬇️  Downloading {filename}...")
            kaggle.api.competition_download_file("imagenet-object-localization-challenge", 
                                                filename, path=".")
            
            if not os.path.exists(filename):
                log(f"   ❌ {filename} not downloaded")
                continue
                
            file_size = os.path.getsize(filename) / (1024**3)  # GB
            log(f"   ✅ Downloaded {filename} ({file_size:.1f}GB)")
            
            # Upload raw file to GCS first
            log(f"   ⬆️  Uploading raw file to GCS...")
            blob = bucket.blob(f"raw/{filename}")
            blob.upload_from_filename(filename)
            log(f"   ✅ Raw file uploaded")
            
            # Extract and process
            log(f"   🔄 Extracting...")
            
            # Extract based on file type
            if filename.endswith('.tar'):
                with tarfile.open(filename, 'r') as tar:
                    tar.extractall()
            elif filename.endswith('.zip'):
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall()
            
            # Look for ImageNet structure
            log(f"   🔍 Searching for ImageNet structure...")
            
            # Common paths where ImageNet data might be
            search_paths = [
                "./ILSVRC/Data/CLS-LOC",
                "./ILSVRC2012", 
                "./ILSVRC",
                "./Data/CLS-LOC",
                "./Data",
                "."
            ]
            
            found_structure = False
            for search_path in search_paths:
                if os.path.exists(search_path):
                    train_path = os.path.join(search_path, "train")
                    val_path = os.path.join(search_path, "val") 
                    
                    log(f"   📁 Checking: {search_path}")
                    
                    if os.path.exists(train_path):
                        # Count training classes
                        train_classes = len([d for d in os.listdir(train_path) 
                                           if os.path.isdir(os.path.join(train_path, d))])
                        log(f"      Training classes: {train_classes}")
                        
                        if train_classes > 900:  # Should be 1000 for full ImageNet
                            log(f"   ✅ Valid ImageNet training structure found!")
                            
                            # Upload training data
                            log(f"   ⬆️  Uploading training data...")
                            train_uploaded = upload_directory_to_gcs(
                                train_path, bucket, "processed/imagenet_organized/train")
                            log(f"   ✅ Training data uploaded: {train_uploaded} files")
                            
                            found_structure = True
                            success = True
                    
                    if os.path.exists(val_path):
                        # Check if it's a directory of images or subdirectories
                        val_contents = os.listdir(val_path)
                        val_images = [f for f in val_contents if f.endswith(('.JPEG', '.jpg', '.png'))]
                        val_dirs = [d for d in val_contents if os.path.isdir(os.path.join(val_path, d))]
                        
                        log(f"      Validation images: {len(val_images)}")
                        log(f"      Validation subdirs: {len(val_dirs)}")
                        
                        if len(val_images) > 10000 or len(val_dirs) > 900:  # Should be ~50K images or 1000 dirs
                            log(f"   ✅ Valid ImageNet validation structure found!")
                            
                            # Upload validation data
                            log(f"   ⬆️  Uploading validation data...")
                            val_uploaded = upload_directory_to_gcs(
                                val_path, bucket, "processed/imagenet_organized/val")
                            log(f"   ✅ Validation data uploaded: {val_uploaded} files")
                            
                            found_structure = True
                            success = True
                    
                    if found_structure:
                        break
            
            if not found_structure:
                log(f"   ❌ No valid ImageNet structure found in {filename}")
            
            # Clean up extracted files
            log(f"   🧹 Cleaning up extracted files...")
            for item in os.listdir("."):
                if item != filename and os.path.isdir(item):
                    shutil.rmtree(item)
            
            # Remove downloaded file to save space
            os.remove(filename)
            log(f"   🧹 {filename} cleaned up")
            
        except Exception as e:
            log(f"   ❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return success

# Run the processing
if download_and_process_imagenet():
    print("🎉 ImageNet processing completed successfully!")
    
    # Create completion marker
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(os.environ['BUCKET_NAME'])
    blob = bucket.blob("processing_complete.txt")
    completion_info = f"""ImageNet processing completed at {time.strftime('%Y-%m-%d %H:%M:%S')}

Processed data location: gs://{os.environ['BUCKET_NAME']}/processed/imagenet_organized/

Directory structure:
- train/        # 1000 class directories with training images  
- val/          # Validation images

Usage for training:
gsutil -m cp -r gs://{os.environ['BUCKET_NAME']}/processed/imagenet_organized ~/data/
"""
    blob.upload_from_string(completion_info)
    
else:
    print("❌ ImageNet processing failed!")

PYTHON_EOF

echo "✅ Data processing complete!"
echo "🛑 Instance will shut down in 60 seconds..."
sleep 60

# Self-destruct
gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet

EOF
}

# Main execution
echo "Checking prerequisites..."
if ! check_prerequisites; then
    exit 1
fi

BUCKET_NAME="$PROJECT_ID-imagenet-data"
INSTANCE_NAME="imagenet-processor-$(date +%Y%m%d-%H%M%S)"

echo "📊 Configuration:"
echo "   Project: $PROJECT_ID"
echo "   Bucket: $BUCKET_NAME"  
echo "   Instance: $INSTANCE_NAME"
echo "   Zone: $ZONE"
echo "   Machine: $MACHINE_TYPE"
echo ""

# Read Kaggle credentials
KAGGLE_JSON=$(cat ~/.kaggle/kaggle.json)

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable compute.googleapis.com storage-api.googleapis.com

# Create bucket
echo "🗄️  Creating storage bucket..."
if gsutil mb "gs://$BUCKET_NAME" 2>/dev/null; then
    echo "✅ Created bucket: gs://$BUCKET_NAME"
else
    echo "✅ Using existing bucket: gs://$BUCKET_NAME"
fi

# Create startup script
echo "📝 Creating startup script..."
create_startup_script

# Create instance
echo "🖥️  Creating compute instance: $INSTANCE_NAME"

gcloud compute instances create "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-standard \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata-from-file startup-script=startup-script.sh \
    --metadata BUCKET_NAME="$BUCKET_NAME",KAGGLE_JSON="$KAGGLE_JSON" \
    --preemptible

echo "✅ Instance $INSTANCE_NAME created and starting..."

# Clean up script file
rm -f startup-script.sh

echo ""
echo "🚀 Data processing started!"
echo "📊 You can monitor progress:"
echo "   gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "🗄️  Check bucket contents:"
echo "   gsutil ls -la gs://$BUCKET_NAME/**"
echo ""
echo "⏱️  Estimated time: 2-4 hours"
echo "💰 Estimated cost: \$2-5"
echo "🛑 Instance will self-destruct when complete"
echo ""
echo "✅ Setup complete! The instance is now downloading and processing ImageNet data." 