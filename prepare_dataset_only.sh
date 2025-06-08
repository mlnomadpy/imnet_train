#!/bin/bash

# ==============================================================================
# ImageNet Dataset Preparation for Google Cloud Storage
# Cost-effective approach: Only prepare data, no GPU instance creation
# ==============================================================================

set -e

PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}
BUCKET_NAME=${BUCKET_NAME:-"${PROJECT_ID}-imagenet-data"}
KAGGLE_COMPETITION="imagenet-object-localization-challenge"

echo "💾 ImageNet Dataset Preparation for GCS"
echo "======================================="
echo "This script will:"
echo "  ✅ Download ImageNet from Kaggle"
echo "  ✅ Process and organize the data"
echo "  ✅ Upload to Google Cloud Storage"
echo "  ❌ NOT create any expensive GPU instances"
echo ""

# Check prerequisites
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No project ID found. Please set one:"
    echo "   gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "✅ Using project: $PROJECT_ID"

# Install Kaggle CLI if needed
if ! command -v kaggle &> /dev/null; then
    echo "📦 Installing Kaggle CLI..."
    pip3 install --user kaggle
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Click 'Create New API Token' to download kaggle.json"
    if [ -n "$CLOUD_SHELL" ]; then
        echo "3. In Cloud Shell, click the 'Upload file' button (⋮ menu)"
        echo "4. Upload your kaggle.json file"
    else
        echo "3. Save kaggle.json to ~/.kaggle/"
    fi
    echo "5. Run: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    read -p "Press Enter when you've completed the Kaggle setup..."
    
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo "❌ Kaggle credentials still not found. Exiting."
        exit 1
    fi
fi

# Enable required APIs
echo "🔧 Enabling GCP APIs..."
gcloud services enable storage-api.googleapis.com --quiet

# Create bucket
echo "🪣 Creating GCS bucket: $BUCKET_NAME"
if ! gsutil ls -b gs://$BUCKET_NAME 2>/dev/null; then
    gsutil mb gs://$BUCKET_NAME
    echo "✅ Bucket created"
else
    echo "✅ Bucket already exists"
fi

# Check competition access
echo "🏆 Checking ImageNet competition access..."
if ! kaggle competitions list | grep -q "$KAGGLE_COMPETITION"; then
    echo "⚠️  You need to join the ImageNet competition first:"
    echo "   https://www.kaggle.com/c/$KAGGLE_COMPETITION"
    echo "   Click 'Join Competition' and accept the rules"
    echo ""
    read -p "Press Enter after joining the competition..."
    
    # Verify access again
    if ! kaggle competitions list | grep -q "$KAGGLE_COMPETITION"; then
        echo "❌ Still can't access competition. Please check you've joined and try again."
        exit 1
    fi
fi

# Check available disk space
available_space=$(df /home -BG 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//' || echo "50")
echo "💽 Available disk space: ${available_space}GB"

# Choose download strategy based on available space
if [ "$available_space" -lt 200 ]; then
    echo "⚠️  Limited disk space detected. Using streaming approach."
    USE_STREAMING=true
else
    echo "✅ Sufficient disk space. Using local processing approach."
    USE_STREAMING=false
fi

echo ""
echo "📥 Starting ImageNet download..."
echo "   Competition: $KAGGLE_COMPETITION"
echo "   Strategy: $([ "$USE_STREAMING" = true ] && echo "Streaming to GCS" || echo "Local processing")"
echo "   Estimated time: 2-4 hours depending on connection"
echo ""

# Create working directory
WORK_DIR="$HOME/imagenet_work"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ "$USE_STREAMING" = true ]; then
    echo "🚀 Streaming download approach (for limited disk space)..."
    
    # Download files one by one and upload immediately
    kaggle competitions files -c "$KAGGLE_COMPETITION" | tail -n +2 | while read -r line; do
        filename=$(echo "$line" | awk '{print $1}')
        if [[ "$filename" == *.zip ]]; then
            echo "  📦 Processing: $filename"
            kaggle competitions download -c "$KAGGLE_COMPETITION" -f "$filename"
            
            # Upload raw file to GCS
            gsutil cp "$filename" gs://$BUCKET_NAME/raw/
            
            # Extract and process if it's a main data file
            if [[ "$filename" == *"Data"* ]] || [[ "$filename" == *"ILSVRC"* ]]; then
                echo "    🔄 Extracting and processing..."
                unzip -q "$filename"
                
                # Find and organize ImageNet structure
                python3 << 'EOF'
import os
import shutil
import glob

def find_and_organize_imagenet():
    """Find ImageNet structure and organize it."""
    paths = ["./ILSVRC/Data/CLS-LOC", "./ILSVRC", "./Data/CLS-LOC", "./Data"]
    
    for path in paths:
        if os.path.exists(path):
            train_path = os.path.join(path, "train")
            val_path = os.path.join(path, "val") 
            if os.path.exists(train_path) and os.path.exists(val_path):
                print(f"✅ Found ImageNet structure at: {path}")
                
                # Count classes and samples
                train_classes = len([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
                print(f"   📊 Training classes: {train_classes}")
                
                # Create organized structure
                if path != "./imagenet_organized":
                    if os.path.exists("./imagenet_organized"):
                        shutil.rmtree("./imagenet_organized")
                    shutil.copytree(path, "./imagenet_organized")
                    print("   📁 Data organized for training")
                
                return True
    
    print("❌ No valid ImageNet structure found")
    return False

if find_and_organize_imagenet():
    print("SUCCESS: ImageNet data ready")
else:
    print("ERROR: Could not organize ImageNet data")
EOF
                
                # Upload organized data to GCS
                if [ -d "./imagenet_organized" ]; then
                    echo "    📤 Uploading organized data to GCS..."
                    gsutil -m cp -r ./imagenet_organized gs://$BUCKET_NAME/processed/
                    rm -rf ./imagenet_organized
                fi
                
                # Clean up extracted files
                rm -rf ./ILSVRC ./Data 2>/dev/null || true
            fi
            
            # Remove downloaded zip to save space
            rm -f "$filename"
        fi
    done
    
else
    echo "🚀 Local processing approach (sufficient disk space)..."
    
    # Download all files first
    echo "  📥 Downloading all competition files..."
    kaggle competitions download -c "$KAGGLE_COMPETITION"
    
    echo "  📦 Extracting files..."
    for zipfile in *.zip; do
        if [ -f "$zipfile" ]; then
            echo "    Extracting: $zipfile"
            unzip -q "$zipfile"
        fi
    done
    
    echo "  🔄 Processing and organizing ImageNet data..."
    python3 << 'EOF'
import os
import shutil
import glob

def process_imagenet():
    """Process and organize ImageNet data."""
    # Look for ImageNet structure
    paths = ["./ILSVRC/Data/CLS-LOC", "./ILSVRC", "./Data/CLS-LOC", "./Data"]
    
    for path in paths:
        if os.path.exists(path):
            train_path = os.path.join(path, "train") 
            val_path = os.path.join(path, "val")
            if os.path.exists(train_path) and os.path.exists(val_path):
                print(f"✅ Found ImageNet at: {path}")
                
                # Count classes and samples
                train_classes = len([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
                
                # Count training samples
                train_samples = 0
                for class_dir in os.listdir(train_path):
                    class_path = os.path.join(train_path, class_dir)
                    if os.path.isdir(class_path):
                        samples = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        train_samples += samples
                
                # Count validation samples
                val_samples = 0
                for class_dir in os.listdir(val_path):
                    class_path = os.path.join(val_path, class_dir)
                    if os.path.isdir(class_path):
                        samples = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        val_samples += samples
                
                print(f"   📊 Classes: {train_classes}")
                print(f"   📊 Training samples: {train_samples:,}")
                print(f"   📊 Validation samples: {val_samples:,}")
                
                # Organize data
                if path != "./imagenet_organized":
                    if os.path.exists("./imagenet_organized"):
                        shutil.rmtree("./imagenet_organized")
                    shutil.copytree(path, "./imagenet_organized")
                    print("   📁 Data organized and ready")
                
                return True
    
    print("❌ No valid ImageNet structure found")
    return False

if process_imagenet():
    print("SUCCESS: ImageNet processing complete")
else:
    print("ERROR: ImageNet processing failed")
EOF
    
    # Upload everything to GCS
    echo "  📤 Uploading to Google Cloud Storage..."
    
    # Upload raw data
    gsutil -m cp *.zip gs://$BUCKET_NAME/raw/
    
    # Upload organized data
    if [ -d "./imagenet_organized" ]; then
        gsutil -m cp -r ./imagenet_organized gs://$BUCKET_NAME/processed/
    else
        echo "⚠️  Organized data not found, uploading raw extracted data..."
        gsutil -m cp -r ./ILSVRC gs://$BUCKET_NAME/processed/ 2>/dev/null || true
        gsutil -m cp -r ./Data gs://$BUCKET_NAME/processed/ 2>/dev/null || true
    fi
fi

# Create dataset info file
echo "📋 Creating dataset information file..."
cat > dataset_info.txt << EOF
ImageNet Dataset Information
===========================

Competition: $KAGGLE_COMPETITION
Processed: $(date)
Project: $PROJECT_ID
Bucket: gs://$BUCKET_NAME

Directory Structure:
- gs://$BUCKET_NAME/raw/          # Original zip files from Kaggle
- gs://$BUCKET_NAME/processed/    # Processed and organized data

Usage:
To use this dataset for training, point your training script to:
gs://$BUCKET_NAME/processed/imagenet_organized/

The data is organized as:
- train/class_name/image.jpg
- val/class_name/image.jpg

Classes: 1000 ImageNet classes
Training samples: ~1.2M images
Validation samples: ~50K images
EOF

gsutil cp dataset_info.txt gs://$BUCKET_NAME/

# Cleanup local files
cd "$HOME"
rm -rf "$WORK_DIR"

echo ""
echo "🎉 ImageNet Dataset Preparation Complete!"
echo ""
echo "📊 Dataset Summary:"
echo "   🪣 Bucket: gs://$BUCKET_NAME"
echo "   📁 Processed data: gs://$BUCKET_NAME/processed/"
echo "   📋 Info file: gs://$BUCKET_NAME/dataset_info.txt"
echo ""
echo "💰 Cost Summary:"
echo "   ✅ This preparation: ~$5-10 (storage + transfer)"
echo "   ✅ Monthly storage: ~$3-5 for ImageNet"
echo "   ❌ No GPU costs incurred!"
echo ""
echo "🚀 Next Steps:"
echo "   1. When ready to train, create a GPU instance:"
echo "      gcloud compute instances create imagenet-training \\"
echo "        --zone=us-central1-a \\"
echo "        --machine-type=n1-standard-8 \\"
echo "        --accelerator=type=nvidia-tesla-v100,count=1 \\"
echo "        --image-family=pytorch-latest-gpu \\"
echo "        --image-project=deeplearning-platform-release"
echo ""
echo "   2. SSH to the instance and download the processed data:"
echo "      gsutil -m cp -r gs://$BUCKET_NAME/processed/imagenet_organized ~/data/"
echo ""
echo "   3. Run your training script"
echo ""
echo "   4. IMPORTANT: Stop the instance when done training!"
echo ""
echo "💡 The dataset is now ready and will persist in GCS until you delete it." 