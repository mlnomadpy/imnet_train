#!/bin/bash

# ImageNet Dataset Preparation for GCS (Cost-Effective)
# Fixed version with better error handling and verification

set -e

PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}
BUCKET_NAME=${BUCKET_NAME:-"${PROJECT_ID}-imagenet-data"}

echo "💾 ImageNet Dataset Preparation (Fixed Version)"
echo "=============================================="
echo "✅ Download ImageNet from Kaggle"
echo "✅ Process and organize data"  
echo "✅ Upload to Google Cloud Storage"
echo "✅ Verify everything worked"
echo "❌ NO expensive GPU instances created"
echo ""

# Check project
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No project ID. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi
echo "✅ Project: $PROJECT_ID"
echo "✅ Bucket: $BUCKET_NAME"

# Install Kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "📦 Installing Kaggle CLI..."
    pip3 install --user kaggle
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials needed!"
    echo "1. Get kaggle.json from https://www.kaggle.com/account"
    echo "2. Upload to Cloud Shell or save to ~/.kaggle/"
    echo "3. Run: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    read -p "Press Enter when ready..."
    
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo "❌ Still no kaggle.json found. Exiting."
        exit 1
    fi
fi

# Enable APIs and create bucket
echo "🔧 Setting up GCS..."
gcloud services enable storage-api.googleapis.com --quiet
if ! gsutil ls -b gs://$BUCKET_NAME 2>/dev/null; then
    gsutil mb gs://$BUCKET_NAME
    echo "✅ Created bucket: gs://$BUCKET_NAME"
else
    echo "✅ Using existing bucket: gs://$BUCKET_NAME"
fi

# Check competition access
echo "🏆 Verifying ImageNet competition access..."
if ! kaggle competitions list 2>/dev/null | grep -q "imagenet-object-localization-challenge"; then
    echo "❌ Cannot access ImageNet competition!"
    echo "1. Go to: https://www.kaggle.com/c/imagenet-object-localization-challenge"
    echo "2. Click 'Join Competition' and accept the rules"
    echo "3. Make sure your kaggle.json is from the correct account"
    read -p "Press Enter after joining the competition..."
    
    # Check again
    if ! kaggle competitions list 2>/dev/null | grep -q "imagenet-object-localization-challenge"; then
        echo "❌ Still cannot access competition. Please check:"
        echo "   - You've joined the competition"
        echo "   - Your kaggle.json is correct"
        echo "   - Your account has API access enabled"
        exit 1
    fi
fi
echo "✅ Competition access verified"

# Check what files are available
echo "📋 Checking available files..."
kaggle competitions files -c imagenet-object-localization-challenge | head -10

# Download and process
echo ""
echo "📥 Starting ImageNet download..."
echo "   This is ~150GB and will take 2-4 hours"
echo "   Using streaming approach due to Cloud Shell space limits"
echo ""

# Create working directory
WORK_DIR="$HOME/imagenet_work"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Track what we've processed
PROCESSED_MAIN_DATA=false

echo "🚀 Downloading files one by one..."
kaggle competitions files -c imagenet-object-localization-challenge | tail -n +2 | while read -r line; do
    filename=$(echo "$line" | awk '{print $1}')
    filesize=$(echo "$line" | awk '{print $2}')
    
    if [[ "$filename" == *.zip ]]; then
        echo ""
        echo "📦 Processing: $filename ($filesize)"
        
        # Download file
        echo "   ⬇️  Downloading..."
        if ! kaggle competitions download -c imagenet-object-localization-challenge -f "$filename"; then
            echo "   ❌ Download failed for $filename"
            continue
        fi
        
        # Verify download
        if [ ! -f "$filename" ]; then
            echo "   ❌ File not found after download: $filename"
            continue
        fi
        
        downloaded_size=$(ls -lh "$filename" | awk '{print $5}')
        echo "   ✅ Downloaded: $downloaded_size"
        
        # Upload raw file to GCS
        echo "   ⬆️  Uploading to GCS..."
        if gsutil cp "$filename" gs://$BUCKET_NAME/raw/; then
            echo "   ✅ Raw file uploaded"
        else
            echo "   ❌ Failed to upload raw file"
        fi
        
        # Process main data files (ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar, etc.)
        if [[ "$filename" == *"ILSVRC2012_img"* ]] || [[ "$filename" == *"Data"* ]]; then
            echo "   🔄 Processing main data file..."
            
            # Extract
            if [[ "$filename" == *.tar ]]; then
                echo "   📦 Extracting tar file..."
                tar -xf "$filename"
            else
                echo "   📦 Extracting zip file..."
                unzip -q "$filename"
            fi
            
            # Look for ImageNet structure
            echo "   🔍 Looking for ImageNet structure..."
            find . -maxdepth 3 -type d -name "train" -o -name "val" | head -5
            
            # Process with Python
            python3 << 'EOF'
import os
import shutil
import glob

def find_and_organize_imagenet():
    """Find ImageNet structure and organize it."""
    print("   🔍 Searching for ImageNet structure...")
    
    # Common ImageNet paths
    search_paths = [
        "./ILSVRC/Data/CLS-LOC",
        "./ILSVRC2012", 
        "./ILSVRC",
        "./Data/CLS-LOC", 
        "./Data",
        "."
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            train_path = os.path.join(path, "train")
            val_path = os.path.join(path, "val")
            
            print(f"   📁 Checking: {path}")
            print(f"      Train exists: {os.path.exists(train_path)}")
            print(f"      Val exists: {os.path.exists(val_path)}")
            
            if os.path.exists(train_path) and os.path.exists(val_path):
                print(f"   ✅ Found ImageNet structure at: {path}")
                
                # Count classes and samples
                try:
                    train_classes = len([d for d in os.listdir(train_path) 
                                       if os.path.isdir(os.path.join(train_path, d))])
                    val_classes = len([d for d in os.listdir(val_path) 
                                     if os.path.isdir(os.path.join(val_path, d))])
                    
                    print(f"   📊 Training classes: {train_classes}")
                    print(f"   📊 Validation classes: {val_classes}")
                    
                    if train_classes > 900:  # Should be 1000 for full ImageNet
                        # Create organized structure
                        organized_path = "./imagenet_organized"
                        if os.path.exists(organized_path):
                            shutil.rmtree(organized_path)
                        
                        shutil.copytree(path, organized_path)
                        print(f"   ✅ Data organized at: {organized_path}")
                        return True
                    else:
                        print(f"   ⚠️  Too few classes ({train_classes}), might be incomplete")
                        
                except Exception as e:
                    print(f"   ❌ Error processing {path}: {e}")
    
    print("   ❌ No valid ImageNet structure found")
    return False

# Run the organization
success = find_and_organize_imagenet()
if success:
    print("SUCCESS: ImageNet data organized")
else:
    print("ERROR: Could not organize ImageNet data")
EOF
            
            # Upload organized data if successful
            if [ -d "./imagenet_organized" ]; then
                echo "   ⬆️  Uploading organized data to GCS..."
                if gsutil -m cp -r ./imagenet_organized gs://$BUCKET_NAME/processed/; then
                    echo "   ✅ Organized data uploaded successfully"
                    PROCESSED_MAIN_DATA=true
                    
                    # Create success marker
                    echo "ImageNet data processed successfully at $(date)" > ./processing_success.txt
                    gsutil cp ./processing_success.txt gs://$BUCKET_NAME/
                else
                    echo "   ❌ Failed to upload organized data"
                fi
                
                # Clean up local data
                rm -rf ./imagenet_organized
            else
                echo "   ⚠️  No organized data found to upload"
            fi
            
            # Clean up extracted files
            echo "   🧹 Cleaning up extracted files..."
            rm -rf ./ILSVRC* ./Data ./train ./val 2>/dev/null || true
        fi
        
        # Remove downloaded file to save space
        rm -f "$filename"
        echo "   🧹 Local file cleaned up"
    fi
done

# Final verification
echo ""
echo "🔍 Verifying upload..."
echo "Bucket contents:"
gsutil ls -lh gs://$BUCKET_NAME/**

# Check if we have processed data
if gsutil ls gs://$BUCKET_NAME/processed/imagenet_organized/ 2>/dev/null; then
    echo "✅ Processed ImageNet data found!"
    
    # Get some stats
    echo "📊 Dataset statistics:"
    gsutil ls gs://$BUCKET_NAME/processed/imagenet_organized/train/ | wc -l | xargs echo "Training class directories:"
    gsutil ls gs://$BUCKET_NAME/processed/imagenet_organized/val/ | wc -l | xargs echo "Validation class directories:"
else
    echo "❌ No processed data found. Check the raw files:"
    gsutil ls -lh gs://$BUCKET_NAME/raw/
fi

# Create detailed info file
cat > dataset_info.txt << EOF
ImageNet Dataset Status
======================
Processed: $(date)
Project: $PROJECT_ID
Bucket: gs://$BUCKET_NAME

Bucket Contents:
$(gsutil ls -lh gs://$BUCKET_NAME/**)

Directory Structure:
- gs://$BUCKET_NAME/raw/          # Original files from Kaggle
- gs://$BUCKET_NAME/processed/    # Organized ImageNet data

Usage for training:
gsutil -m cp -r gs://$BUCKET_NAME/processed/imagenet_organized ~/data/

Expected: 1000 classes, ~1.2M training images, ~50K validation images
EOF

gsutil cp dataset_info.txt gs://$BUCKET_NAME/

# Cleanup
cd "$HOME"
rm -rf "$WORK_DIR"

echo ""
if gsutil ls gs://$BUCKET_NAME/processed/imagenet_organized/ 2>/dev/null; then
    echo "🎉 Dataset Preparation Successful!"
    echo "📊 Bucket: gs://$BUCKET_NAME"
    echo "✅ Processed data: gs://$BUCKET_NAME/processed/imagenet_organized/"
    echo "💰 Cost: ~$5-10 (no GPU costs)"
    echo ""
    echo "🚀 Ready for training! Next step:"
    echo "   ./create_training_instance.sh"
else
    echo "❌ Dataset preparation incomplete!"
    echo "📋 Check bucket contents: gsutil ls -la gs://$BUCKET_NAME/**"
    echo "🔍 Check raw files: gsutil ls -lh gs://$BUCKET_NAME/raw/"
    echo ""
    echo "Common issues:"
    echo "- Kaggle competition not joined"
    echo "- Network interruption during download"
    echo "- Insufficient permissions"
    echo ""
    echo "💡 You can re-run this script to retry"
fi 